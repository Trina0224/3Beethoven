# 多樣化直接列式蒸餾：受控單輪協議

狀態：**訓練前草案；採兩道不可混淆的 freeze。** 靜態 curriculum/final/code preflight 全過並固定 hash 後，才可啟動付費教師；教師完整結果與外部 hash 再通過動態 preflight 後，才可啟動 GPU。教師結果不可能在付費呼叫前存在，因此不得用一個自我矛盾的「全部先凍結」字樣掩蓋時序。

## 1. 唯一目標

從原始 v15 出發，產生一個在固定「清楚統計列式」範圍內全面不弱於 v15、且對未見過的合法參數結構有實質改善的學生。學生只需輸出已代入數值的算式，算術交給程式；不加入陌生語文改寫、自由解釋、選擇題或一般語文能力。評估的數學主分事前固定為 grader 從 raw 中安全抽出的列式之 `math_correct is True and executable is True`；正確列式外帶說明不因文字格式改判為數學 0 分。`whole_raw_executable`（整個 raw 是否僅為 optional `Expression:` 前綴加單一安全 AST）與 `strict_one_line_expression` 都只是逐題、逐類獨立揭露的介面診斷，**不併入** student promotion gate，也不得事後增設格式門檻。SFT 教材則仍必須同時通過 whole-raw 與 strict 合約，不把不完整輸出教給學生。

原始 v15 adapter SHA-256 固定為 `9369d52de4a886df9da0c872cd41bd4e01af0a38bf02ad724b5951c1a6b9f5d3`。不從 V58 或其他後續偏科模型繼續。

## 2. 固定能力範圍

沿用 18 個直接列式類別：

1. 獨立雙事件 `both / neither / exactly_one / at_least_one / same`
2. affine moments `E[Y] / Var(Y) / E[Y**2]`
3. Poisson 隨機變數 `Var(X) / Var(aX+b) / E[X**2]`
4. homogeneous Poisson process `Var(X) / Var(aX+b) / E[X**2]`
5. uniform `E[T] / E[T | T>c]`，所求一律明示為 total `T`
6. binomial `P(X=r)`
7. normal-theory interval 在樣本量改變後的新上界

多樣性是數學結構，不是文風。課程必須以 coverage slot 明確涵蓋：負數、零、分數、不同機率分母、輸入順序、秒／分鐘、非零 uniform 下界、binomial 邊界 `r=0,1,n-1,n`、負或跨零 interval，以及整數與有理數樣本量倍率。

## 3. 資料與分割

- train：18 類各 48，合計 864。
- development：18 類各 6，合計 108；只供 checkpoint 選擇。
- final blind：18 類各 8，合計 144；教師、訓練與 checkpoint 選擇均不可讀。
- 先前已曝光結構診斷：24 題，當時記錄的 manifest SHA-256 為 `9dd9fe1b913824396d98ff4035783ee894cfb37e5a1a07ddc489fb3e7fb6de73`。目前 GitHub 沒有逐題 prompt、semantics 與 raw output，Kaggle V58 保存版及現有草稿亦找不到該 bundle；所以 10/24、16/24 只保留為歷史診斷，**不得作本輪可執行 gate**。本輪若使用 exposed regression，必須另行凍結完整逐題 bundle 並先重跑 v15 baseline，不能冒稱原來同一套 24 題。
- V58 的既有固定模板題保留為 legacy retention suite；它不能代替新 final blind。

相同參數故事的易混淆概念可在同一 split 中成組對照，但完整 story、語意 task key、數值實例及 question 字串不得跨 split。normalized prompt skeleton／template ID 的跨 split overlap 必須重算並如實報告，但不為了追求字面 0 overlap 加入 split-specific 前綴或作文式改寫；三個 split 共享同一個清楚直問 renderer 介面。final 使用獨立 generation stream／seed，並持有事前指定、train 未見的低語文負擔 structural combinations：已知量順序、A/B 順序、括號／分數／負號、秒／分鐘表示及 coverage-slot 組合。holdout 的對象是數學結構與數值組合，不是另造一套文風。V58 的問題不是單獨「同模板」，而是同模板之下幾乎只有安全正整數結構、又缺少廣泛 retention 證據。

train 依 story 對照交錯，不以整類連續區塊餵入。固定 864-row 順序分三段，每段 288 rows、18 個 category 各 16；每 8-row effective batch 的單一 category 不得超過 2。另用凍結 category→domain-family 映射檢查：五個 event category 合計 240 rows，108 batches 中必須精確為 24 個 batch 各 3 筆、84 個各 2 筆；其他 domain family 每 batch最多 2。seed 31415 只做 CPU generator/order invariant fuzz，絕不訓練第二個學生，且本輪不宣稱已證明 training-seed robustness。

final blind 在任何教師請求與推論前生成並封存 hash。教師程式只能取得 train rows；輸出紀錄中若出現任一 development/final ID、question 或 hash，preflight 必須失敗。

## 4. 教師教材

- 教師固定為 `meta-llama/llama-3.3-70b-instruct`，temperature 0。
- 864 個 train 問題以每 packet 12 題請求，首輪最多 72 calls。
- 只重試首輪未通過的項目一次；retry 只提供通用數學 rule card，不提供該題 oracle、reference expression、answer 或 final/development 內容。
- 全部 calls 上限 144；client-reported 累積成本硬上限 US$0.30。完成回覆缺 cost 欄時立即停止後續呼叫。
- 每次 raw、provider/model、usage、cost、parse、逐題判定及重試原因都保存；不可覆寫失敗回答。
- 教師 raw 永久保留並逐題綁定。若 raw expression 通過事前 legal-equivalence bank、symbolic dependency 與其餘 gates，該列標為 `raw_teacher_accepted`，training target 保留教師的原始 expression；不要求與 canonical reference byte 相同，也不以純數值相等放寬。若教師錯誤、未知等價或格式失敗，training target 改採 canonical symbolic oracle，標為 `canonical_corrected`，並保留 raw、錯因與修正前後內容。不得把後者計作 teacher accepted。
- 每個 training target（無論 provenance）都必須同時通過 prompt→semantics、executable、symbolic dependency、數學結構與 `strict_one_line_expression=true`。這是 SFT 教材衛生，不是把學生評估改成語文能力；不合格式的教師 raw 必須走明示的 `canonical_corrected`，不得原樣進訓練。864/864 merged targets 未全數通過、任何 pending 未決、模型身份不符或 receipt hash 不符時，training release 必須為 false。

教師不是 ground truth。每個 reference 另由語意 oracle、bounded exact calculator 與結構 grader 驗證；數值巧合不能取代事件 identity 或公式結構。

## 5. GPU 前不可省略的 preflight

先執行靜態 gate（1–10 與不依賴教師結果的 12–13），把 curriculum whole-file hash、final commitment、grader/runner/code hash 固定後才准呼叫教師；再執行動態 gate（11、教師結果 receipt 與其餘 12–13），把 teacher-results whole-file hash 由 runner 外部參數鎖定後才准載入 GPU。以下全部為硬 gate：

1. 所有新增與既有相關 unit/mutation tests 通過。
2. 每 split 與每 category 數量正確；每個 coverage slot 達到預定下限。
3. 題面由獨立 prompt↔semantics 檢查器核對，不只檢查百分比；斜線分數、moments、process units、uniform conditional flag、binomial counts 與 interval factor 都必須一致。
4. `conditional=False` 的 uniform 必須有 `cutoff==lower`；`conditional=True` 必須有 `lower<=cutoff<upper`。
5. oracle、reference expression、answer、grader 四者一致。
6. 邊界／零值可用於 train 教學，但評估題排除會讓錯誤公式碰巧同值的退化參數；事件題至少排除 `p=q`、`p+q=1`、0 與 1。每個常見混淆在 development/final 至少有一題非退化判別例，且 wrong-formula mutation 與 oracle 不得數值或結構同值。
7. 每個 coverage slot 都有手寫的合法等價式 bank 與典型錯式 bank；grader 必須前者全收、後者全拒且 pending=0。負號、分數與括號不可留到推論後才補 grader。static fixture 至少固定包含：事件 `neither=1-p-q+p*q`、`exactly_one=p+q-2*p*q` 與 `(p+q-p*q)-p*q`、`at_least_one=p+q-p*q`、`same=1-p-q+2*p*q`；affine 二階動差 `a**2*(v+mu**2)+2*a*b*mu+b**2`；Poisson 二階動差 `mu*(mu+1)`；以及 binomial `r=0,1,n-1,n` 的合法簡式。每個 fixture 都必須搭配錯係數、錯次方或漏項的 near-wrong negative；合法式須 `primary_correct=true/review_required=false`，錯式須 `primary_correct=false/review_required=false`。binomial `r=0/n` 可合法省略值為 1 的 `comb`／補事件因子，`comb` 是否被正確使用另在 interior `r` 檢查。
   另固定 raw-whole-expression 反例：`999;<canonical>`、`x=999;<canonical>`、`<canonical> # comment`、code fence 及附加解說的 `whole_raw_executable` 均必須為 false；這個介面診斷不得改寫 student 數學主分。裸的 `<canonical>` 與 `Expression: <canonical>` 皆可數學正確，前者 whole-raw 為 true 但 strict 為 false，後者兩者皆為 true。教師 raw 與 864 個 merged SFT targets 則必須 whole-raw=true 且 strict=true，否則不得列為 `raw_teacher_accepted`或釋出訓練。
8. train/development/final 的 story、task key、question、prompt 及數值情境必須零洩漏；指定 held-out structural combination 不得出現在 train。normalized skeleton／template ID 由凍結 checker 直接從 question/prompt 重算、不可信任 row 自填 signature，並輸出 overlap 報告；shared clear templates 本身不算失敗。
9. 新 development/final 對歷史資料做 exact question/task-key collision audit；碰撞逐筆列出且 final 必須為 0。
10. 教師請求與結果完全不含 development/final 內容。
11. 864 個 training target 都能追溯至教師 raw 與明確 provenance；`raw_teacher_accepted` 不改原式並另綁 canonical reference，`canonical_corrected` 留存錯因與 oracle 修正。runner 實際訓練的是 teacher-results 內經驗證的 merged targets，不可悄悄忽略老師而全部使用 curriculum reference；五道 gate 全過，pending=0。
12. curriculum、三個 split、教師 requests/results、grader、runner、legacy development/final、原 v15、base revision 與環境版本都有 SHA/revision receipt。legacy suite 的確切檔案及 hash 必須在訓練前固定。
13. preflight 程式預設只驗證；缺檔或任何 gate 失敗時必須在載入 CUDA／模型前退出非零。
14. final blind 使用專用的原子排他標記（`O_CREAT|O_EXCL`）；標記內只寫已公開凍結的 commitment，並在第一次對 blind path 進行 `stat`、hash 或 open **之前**完成落盤。若 blind 檔缺失、被竄改或評估中斷，該次仍視為已消耗，不得自動重試。
15. 本輪 baseline、checkpoint、legacy retention 與 final 均直接使用凍結 grader 的 raw-output 判定；任何 `review_required=true` 即停止或失敗。不得在看過模型輸出後加入 review ledger、新等價式或人工改分來挽救本輪結果；必要的 grader 修正必須在教師收集與 GPU 之前完成並重新凍結全部 receipt。

v15 baseline receipt 還必須記錄 adapter config/weights、base revision、tokenizer 與 chat template、完整 prompt、grader fingerprint、環境、load mode、generation config，以及逐題 raw 與 correct vector hashes。學生比較前在相同 runtime fresh reload v15，不沿用記憶中分數。

## 6. 唯一訓練軌跡與選點

- seed/data seed：2027。
- 一個 epoch，864 rows，batch 1、gradient accumulation 8，共 108 optimizer updates。
- learning rate：`1e-5`，constant schedule，無第二個 seed、第二輪、模型混合或補跑。
- 只在 36、72、108 步保存輕量 LoRA adapter；不保存 optimizer checkpoint。選點完成後只封存選中的一份。
- 三個 checkpoint 只可讀 development、另行完整凍結（若有）的 exposed regression 及 legacy development；final blind 的具體 rows 在 selection receipt 核准前不可載入。
- 第一個 optimizer step 前，v15 必須以同一推論介面當場完成新 development、legacy development 與 legacy final 三份 baseline；checkpoint 選點每個都加跑 legacy development。選中後只對該學生跑一次 legacy final，未通過即在讀取 final blind 前硬停，不回退改選其他 checkpoint。

候選 checkpoint 必須同時滿足：

- 108 題 development 的 18 類各不低於同場 v15。
- v15 原本答對的 development 題 paired losses 為 0。
- development 總分至少比 v15 增加 6/108。
- 若本輪另行凍結 exposed regression：須達該 bundle 的事前門檻且保留同場 v15 原本答對的全部題；不得套用已遺失原始 bundle 的 14/24 或「原本 10 題」數字。
- legacy development 逐類不退、paired losses 為 0。
- pending 為 0。

按 36→72→108 的固定順序，取**最早通過全部 gate**者；不在三點間追逐最高分。final 解封前，所有不需 final rows 就能判定的最終條件（legacy retention，以及若有 exposed regression 的最終門檻）亦必須先通過；不能明知某項最終條件失敗仍花費 final 推論。沒有合格 checkpoint 時停止，不解封 final，不追加訓練。

## 7. 事前成功門檻

選中 checkpoint 後，原 v15 與學生才以相同 prompt、greedy decoding、token budget 和 grader 同場跑 final blind。成功必須全部成立：

1. final blind 總分至少比 v15 增加 **12/144**。
2. 18 類每類正確數均不低於 v15，且至少 6 類嚴格改善。
3. v15 原本答對的 final blind 題 paired losses 為 0。
4. legacy final retention 的每類不退且 paired losses 為 0。
5. 若本輪另行凍結 exposed regression，學生不得失去同場 v15 已答對題，且須達其事前固定改善門檻；此項只證明已知缺口回歸，不稱為盲測。遺失 raw 的歷史 24 題不列入成敗。
6. 全部數學評分 pending=0；人工複核只承認事前列出的代數恆等式，不補學生遺漏項。
7. `whole_raw_executable` 與 `strict_one_line_expression` 均逐題、逐類、v15/student paired 另表；它們不進入上述數學主分或 promotion gate，也不得在看過輸出後臨時增加最低比例。若學生介面格式退步，結果文件必須如實揭露，但不能把它重命名成數學失敗或反向用格式改善冒稱數學成功。

任何一項未達即為本輪失敗，保留原 v15 作起點；不得用局部類別、某個 checkpoint 或事後改門檻重新命名為成功。

## 8. 保存與停止

只保存：凍結資料與 hash、教師 raw/usage/cost、preflight/selection/final receipts、每題原始模型輸出、選中 adapter、訓練 trace、結果與撤回／採用結論。暫存 checkpoint 不發布；不建立額外 seed 或重複 Kaggle 成功版本。

無論成功或失敗，結果封存後立即停止 GPU。README、PROJECT_SPEC、目前狀態、恢復文件及歷史決策必須同步；V58 的原始實驗結果保留，但不得再稱為目前成功學生。
