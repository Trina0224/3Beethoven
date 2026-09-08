# 統計多樣性教材：凍結前協議

本協議固定下一輪的教材、驗證與防洩漏（leakage）界面。目的不是再做一輪「哪裡錯就補哪裡」，而是在訓練前把概念覆蓋、數值邊界、混淆式、歷史撞題與盲題隔離全部變成可執行合約。

## 1. 任務範圍

學生只回答清楚、直接、已給定所有數值的統計列式題；輸出一行 `Expression: <fully substituted numerical expression>`，算術交給 bounded exact calculator。三個 split 共用同一個直接問法 renderer，不加入 split cue、陌生改寫或語文難度。`template_signature` 的重疊會據實回報，不把模板重疊誤稱為語文泛化。

固定 18 類：

| family | categories | 同故事正確對照 |
|---|---|---|
| independent events | both, neither, exactly_one, at_least_one, same | 同一組 P(A)、P(B) 問五個不同事件 |
| affine moments | moment_mean, moment_variance, moment_second | 同一個 Y=aX+b 問 E[Y]、Var(Y)、E[Y²] |
| Poisson count | poisson_variance, poisson_scaled, poisson_second | 同一個 X 問 variance、scaled variance、second moment |
| Poisson process | process_variance, process_scaled, process_second | 同一 rate/duration 故事問三個量 |
| uniform | uniform_mean, uniform_conditional | 同一 support 問 unconditional mean 與 conditional total mean |
| binomial | binomial | 同一 n,p 故事輪替 r=0、1、interior、n−1、n |
| interval | interval | 同一原區間使用多個 sample-size multiplier |

所有 contrast 都是正確 target。`misconception_traps` 與 `discriminative_mutations` 只供稽核；錯式絕不作為訓練 target。數值邊界造成錯 identity 偶然同值時，明標 `degenerate_numeric_collision_not_a_negative_target`，不假裝它具有概念判別力。

## 2. 固定數量與物理分割

| split | 每類 | 總數 | 用途 |
|---|---:|---:|---|
| train | 48 | 864 | 唯一訓練／teacher-eligible split |
| development | 6 | 108 | 訓練前診斷與選點；不訓練、不送 teacher |
| final_blind | 8 | 144 | 訓練與選點完成後才解封 |

物理檔案固定如下：

- `docs/STATS_DIVERSE_CURRICULUM.json` 只可有 `train`、`development`、`manifest`。
- `docs/STATS_DIVERSE_FINAL_BLIND.json` 只可有 `final_blind`、`receipt`。

public manifest 只保存 blind 的 count、canonical split hash、receipt hash 與 whole-file hash，不內嵌 blind row、ID、故事或 task key。preselection loader `load_public_curriculum()` 不解析 blind path；runner 另以 OS-level open audit 證明訓練／選點程序沒有 stat/open/read blind 檔。

final 使用 `seeded_rng(SEED, split, family, index, attempt)` 的獨立 generation stream，但與 train/development 共用相同直接問法 renderer。實作以 `sha256(SEED|split|family|index|attempt)` 前 8 bytes 作 RNG seed；manifest 固定每個 split 與 family 的 domain-label hash，blind receipt 另綁 final stream hash。這是結構與數值組合 holdout，不是第二種文風。

## 3. 事前 coverage slots

每列都有 machine-readable `coverage_axes`、`stratum`、`contrast_group`、`group_story_key` 和 `template_signature`；manifest 同時保存每個 split、每個 category、每一 axis 的 counts 與 observation hash。

最低覆蓋合約：

- events 同時含相同與不同分母，並交叉 A→B / B→A 給定順序；train 含 p=0 與 p=1 的 boundary practice。development/final 僅用 0<p,q<1，排除 p=q、p+q=1 及五個 event answer 的偶然同值。
- moments 的 mean、scale、offset 含負數、零與非整數分數；variance 含零、整數與分數。evaluation 的 scale/offset 不為零，避免漏項錯式偶然正確。
- Poisson mean 含分數；μ=0 只可作 train boundary，development/final 不用退化 μ。
- process 同時含秒與分鐘 duration、整數與分數 rate；秒題的 semantics duration 一律換成分鐘後精確比對。
- uniform lower 同時含 zero、nonzero integer、fraction；conditional cutoff 同時以秒與分鐘顯示。train 明含 seconds+nonzero-integer lower 與 minutes+fractional lower 作 compositional bridge，fractional-lower+seconds 才保留給 final。unconditional row 強制 `conditional=false` 且 `cutoff==lower`；conditional row明問 total T，不把剩餘等待時間當 target。
- binomial 每個 split 都含 r=0、1、interior、n−1、n。r=0/n 時省略 `comb(n,r)=1` 是合法簡式，不能拿來當「漏 comb」判別題；同 split 另有 interior discriminator。
- interval 含全負、跨零、全正與 zero-lower；除事前保留的 negative+fractional conjunction 外，train 用其他 geometries 交叉整數／分數 endpoint，避免兩軸整體共線。sample-size multiplier 含整數與有理分數，其 width divisor 必須是 exact rational square root。development 至少含 negative/cross-zero/positive 且整體含兩種 endpoint form，final 再含 zero-lower。

事前保留七種「train 與 development 都沒有、final 有」的組合：same-denominator events 配 B→A 給定順序（interior、五答案互異）；全負分數 affine mean/scale/offset；fractional Poisson mean 配兩個負分數 transform term；seconds process 配 fractional rate 與 negative-fraction scale；fractional uniform lower 配 seconds cutoff；even-n binomial 配 p>1/2（同一 final story 含五個 r slots）；negative fractional interval 配 rational sample multiplier。manifest 固定條件、各 split count 與 final match commitment hash。這些是已見單軸的新 conjunction，不是新文風。

## 4. 訓練順序與遺忘防線

train 由 48 個 18-row cycle 組成，每個 cycle 每類恰一題。同故事 contrast 保持同一 cycle 與相同 `contrast_group`，但跨 family 交錯，不把五個 event row 連續堆在一起。

凍結的 8-row microbatch 合約從 category→domain-family 常數重算，不信任 row 自填 family：

- 108 個 microbatch 中 events 必須恰為 24 batch×3 rows 加 84 batch×2 rows；這是 240 event rows 下的數學最低上限。
- 其他 domain family 每個 microbatch最多 2 rows；單一 category 最多 2 rows。
- 每個 288-row window 每類恰 16 rows；尾端 72/144 rows 的逐類 counts 另存 manifest。

legacy retention 是 runner 的外部、whole-file-hash-locked 輸入，不由本 builder 重造或偽造 hash。現有 24-probe bundle 也是外部輸入；本教材不宣稱產生它們。最終 promotion 仍須逐類不低於原 v15，不能以總分掩蓋遺忘。

## 5. 每列四層驗證

每列在寫檔前依序通過：

1. `parse_question_semantics()` 以凍結 regex grammar 從實際 question 抽回 target、數值與 unit；與 row semantics 作 canonical rational comparison。這個 parser 與 renderer 的 target phrase mapping 分開維護。
2. `oracle(semantics)` 以 Fraction 規則獨立求值，並要求 reference expression 與 answer 都精確相等。
3. `stats_consolidation_grader.score()` 對 reference target 必須 `primary_correct=true`、`strict_one_line_expression=true`；grader version/fingerprint 與 checker source hash 寫入 manifest。
4. 每個 misconception mutation 都實際執行 calculator 與 grader。值不同的 wrong expression 必須 `primary_correct=false`；development/final 每個列出的常見混淆至少有一題非退化 discriminator。

測試另用 18 個手寫 prompt fixture 驗 parser，不從 generator 自我複製；也用手寫 legal-equivalence / near-wrong pairs 驗 events、affine second moment、Poisson second moment、process conversion、uniform total mean、interval width 與 binomial boundary folds。

builder 本身對每列執行 parser/oracle/grader/mutation 驗證並檢查 order/count；測試另以 seed `31415` 在記憶體重建一份即棄 corpus，重跑同一組 invariants。它不落盤、不進訓練、不供選 root seed 或模型，也不據此宣稱 training-seed robustness。

## 6. 歷史撞題掃描

生成前遞迴掃描 `docs/**/*.json` 及 `docs/**/*.json.gz.b64`，後者先 base64 decode 與 gzip decompress；明確包含 `STATS_FINAL_CLEAR_DATA.json.gz.b64`。本輪所有 `STATS_DIVERSE_*` 派生 artifact 排除在歷史來源外，確保 frozen rebuild 不會因 collector/result 檔新增而漂移。

掃描對所有 dict 先收集 string-valued `question`/`prompt`，不要求它具完整 semantics；其後才嘗試 semantic task key 與 primitive story projection。manifest 保存每個 source file 的 encoded/decoded SHA256、exact question/prompt registry hash、task registry hash、primitive story registry hash與接受／拒絕計數。

硬門檻為：跨 train/development/final 的 exact question、exact prompt、semantic task key、group story key與 numeric scenario 均不得碰撞；新列也不得碰歷史 exact text/task/primitive story。normalized template 可重疊且須據實報告。

## 7. Teacher 與 provenance

資料 provenance 固定為 `assistant_authored_verified_rule_expansion`，target authority 固定為 `canonical_symbolic_oracle`。不使用 final_clear 的 `source_match` 來過度宣稱逐列 teacher anchor。

只有 train row 有 deterministic `teacher_binding`（result_id、training_row_id、prompt hash、canonical target hash與 release requirements）。Teacher 只能提供 train 的規則／表述候選，不是答案權威；development/final 的 binding 為 null 且 `teacher_eligible=false`。若 teacher raw expression 錯誤，collector 必須保存 raw 與 hash、記錄 canonical correction provenance，再經 parser/oracle/mutation gate；release row 必須 `review_required=false`。本 builder 本身不呼叫雲端 teacher。

## 8. 重建與驗收

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts \
  python scripts/prepare_stats_diverse_curriculum.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts \
  python scripts/prepare_stats_diverse_curriculum.py --check

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts \
  python scripts/test_stats_diverse_curriculum.py -v
```

任何 count/hash、prompt binding、coverage slot、mutation、history registry、physical split、microbatch order 或 deterministic rebuild 不符即停止，不開始訓練。final_blind 僅能在最後 adapter 已固定且 selection 完成後由 final runner 開啟。
