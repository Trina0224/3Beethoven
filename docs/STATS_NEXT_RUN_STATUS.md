<!-- REPAIR_EXECUTION_LIVE_BEGIN -->
最新：2026-09-07 15:00 PDT（America/Los_Angeles）。本輪固定權重控制、低 LR 修復及封存**全部完成，GPU 已關閉**。報告：[STATS_REPAIR_REPORT.md](STATS_REPAIR_REPORT.md)；機器結果：[STATS_REPAIR_RESULTS.json](STATS_REPAIR_RESULTS.json)；下載定位：[STATS_REPAIR_DOWNLOAD.json](STATS_REPAIR_DOWNLOAD.json)。

- 兩個 seed 各完成 33 updates；LR 5e-6，同 67 新＋192 replay，12／24／33 的六個 checkpoint 都未通過原門檻。原 v15 維持首選。2027 最後舊 28/48、組合 38/48、事件 14/16；31415 為 29/48、38/48、12/16。語義 pending 為 0。
- 原 v15／原批 31415 step24／step33 的 336/336 個選點原答重現；三者教材原題皆 72/72。另事前登記的 prepared-parent 對照 112/112 原答相同。這些對照不等於獨立泛化測試，也沒有隔離說法與參數的影響。
- 實際 518/518 microbatch 的 input、labels、順序及 token 數驗證通過；各 seed 5,516 supervised tokens。續跑 loss 正確記錄 raw 與新增 21 步的分母；仍未做 resume／不中斷的數值等價控制。
- 已修正**未來** baseline／student 的數值準備一致性，並禁止混用缺契約、不同權重／題集／grader／環境或改過分數的基準快取。三項測試通過，commit `423157eaf793a0d98051799d80a6a493caa5ce98`。新比較需新 output root；本輪重現用保存的凍結 `repo/`，不能套新 runner 覆寫歷史結果。
- Kaggle V54／`348070778` 已顯示 Quick version complete。小型 ZIP 189,903,038 bytes；從保存版本重新下載後 SHA、CRC 與 464 個檔案 hash 全部通過。完整 outputs 保留所有中間 adapter 與 optimizer checkpoints，portable ZIP 含两份最後診斷 adapter。
- 本輪 0 新教師呼叫、66 optimizer updates，沒有獨立 holdout／永久 MC。完整教師 gate 及四次比較仍未完成。較小 LR 未建立跨 seed 穩定保留；未見題保留問題尚未解決，不能說模型能力已修復。

下一輪若繼續，先讀上述報告的剩餘限制與固定配置；不要重跑已完成控制／修復，不要把已曝光選點原題回填訓練。以下區塊全是較早歷史，不代表仍有工作在執行。
<!-- REPAIR_EXECUTION_LIVE_END -->

<!-- FIRST_STUDENT_AUDIT_LIVE_BEGIN -->
最新：2026-09-07 PDT，第一批離線異常診斷完成，詳見 [STATS_FIRST_STUDENT_AUDIT.md](STATS_FIRST_STUDENT_AUDIT.md) 與 JSON 證據。

- 兩個 seed 的 259 筆答案監督、資料順序 hash、parent hash、最後權重 hash 與全部驗證歷程已核對；replay 占 supervised tokens 76.9%，無漏遮罩或答案截斷。
- 31415 step24→33 是 14 道舊題由對變錯、1 道變對；退步集中區間 6、等待 5、Poisson 時間 3，都是原始列式錯誤。最後一段仍規劃了 11 筆正確區間 replay。
- 已修正未來 runner/collector 的續跑 loss 回報分母，保留 raw 與新增步數；不改訓練或選點。原最後 loss 0.03034／0.02627 對應本段平均約 0.11125／0.09632。3 項測試通過；歷史結果未覆寫。
- 尚未證明退步的根因；沒有做續跑與不中斷數值控制，不能宣稱已排除所有管線問題。
- 下一段先做固定權重重載重現＋保存 replay 題對照；方案在診斷報告，尚未開跑。本段沒有新增教師／GPU／訓練，維持原 v15。
<!-- FIRST_STUDENT_AUDIT_LIVE_END -->

<!-- FIRST_STUDENT_LIVE_BEGIN -->
第一批已完成（2026-09-07）：filtered-pilot 診斷批次，67 筆新教材＋192 筆 replay。兩個 seed 均完成 33 updates、epoch 1.0；所有待判定項已複核，沒有全門檻過關點，不升級取代原 v15。

- 原 v15 同環境基準已完成並複核：舊 34/48、組合 38/48、事件 12/16。
- 2027 最後：舊 29/48、組合 39/48、事件 14/16；舊 exactly_one 與 poisson_scaled 未過保留門檻。
- 31415 最後：舊 22/48、組合 41/48、事件 11/16；舊 interval、poisson_time、uniform_time，組合 Poisson 與事件 neither 未過門檻。
- Kaggle Version 53 已顯示 Successful，保存所有階段權重與 optimizer checkpoints。小型下載包 185,311,896 bytes，SHA-256 已驗證，含兩份最後診斷 adapter。下載位置見 STATS_FIRST_STUDENT_DOWNLOAD.json。
- 報告見 STATS_FIRST_STUDENT_REPORT.md，逐项結果見 STATS_FIRST_STUDENT_RESULTS.json。完整教師教材 gate 仍未通過，fresh-base 與 v15 的完整四次比較尚未完成；本批是選點驗證，沒有獨立測試結論。

規則見 STATS_FIRST_STUDENT_PROTOCOL.md；下方保留較早紀錄，不代表目前狀態。
<!-- FIRST_STUDENT_LIVE_END -->

# 統計蒸餾下一輪狀態

## 歷史：精簡輸出驗證完成，剩下區間縮放問題

2026-09-07 07:14 PDT。已完成 24 次新教師呼叫、逐題複核與存檔。程式／預期／規則在付費前固定於 `50149e1c5c0ddfcf69dc5fd4fab81497b862d43e`。本輪仍用 Llama 3.3 70B、DeepInfra、temperature 0、400-token 上限；沒有學生訓練。

| 同一批開發題，同一套離線診斷 | 前輪 B | 本輪精簡 C |
|---|---:|---:|
| Poisson 單位／變異數 | 8/8 | 8/8 |
| 區間上端點 | 0/8 | 6/8 |
| 仿射 Poisson 二階動差 | 4/8 | 8/8 |
| 可驗證正確 | 12/24 | 22/24 |
| 輸出截斷 | 11/24 | 0/24 |

同題配對為 12 題兩次皆可驗證、10 題由不可用變可驗證、2 題皆未通過。這是已看過的開發題與事後共同診斷，不是新的泛化測試，也不能宣稱只靠提示就讓正確率升到 22/24。

原凍結自動 gate 是 **10/24、未通過**：13 筆被判 malformed，1 筆等價待判。原答顯示其中 10 筆是把答案放在回聲 `output_structure` 裡；另有多餘右括號、平方根／小數在中間量的解析不一致，以及區間中心／半寬部分求值。這些在獨立診斷中逐項處理，未覆寫原 gate。回聲提取必須同時核對外層題號、題目全文及唯一內層答案；不修補算式。

**兩題真錯已定位，pending 為 0：**

- 區間 `[61,194]`、樣本量 ×25：老師把新半寬寫成 `66.5*sqrt(25)`，應是 `66.5/5`，把縮小寫成放大。
- 區間 `[34,223]`、樣本量 ×4：老師的多段根號乘除等同把舊半寬乘 `sqrt(2)`，應乘 `1/2`。同樣是縮放方向錯誤。

即使採離線診斷的 22/24，區間仍只有 6/8，低於事先固定的每類 7/8；所以不放行 full 教材或 GPU。這次確實消除了已觀察到的截斷，也證實其他兩類都能產生正確答案；下一個需要集中處理的是**樣本量增加時，半寬按平方根反比縮小**，無須再重做所有類別或合併 adapter。

新增費用 US$0.00112608；三輪合計 110 次呼叫、US$0.00806302。57 項離線測試通過。`STATS_TEACHER_COMPACT_EVIDENCE.json` 保留全部 raw、request、usage、finish reason 和原 gate；`STATS_TEACHER_COMPACT_REVIEW.json` 與 `scripts/review_stats_teacher_compact.py` 可重現完整診斷。Kaggle 封裝 `3beethoven_teacher_compact.zip`（39,387 bytes）已通過完整性檢查。

後續採用不同提示、額外縮放規則或 reference-assisted 教材，需要先另記預期／控制；本輪按事先固定的失敗停止規則結束，沒有追加猜測式呼叫。

---

## 歷史：教師 A/B 已完成，當時未放行訓練

2026-09-07 05:29 PDT。已依離線期間授權完成修正、50 項測試、24 道新開發題 × 2 組共 48 次教師呼叫，以及全部原答複核。沒有啟用 GPU、沒有改學生權重、沒有生成 full 教材。下方 04:51 的紀錄保留為歷史，不能再當成最新診斷。

| 開發題型（每組 8 題） | A：直接列式，凍結自動接受 | A：離線語意複核後 | B：中間量，凍結自動接受 | B：完整 JSON 離線提取及語意核對後 |
|---|---:|---:|---:|---:|
| Poisson 單位／計數變異數 | 8 | 8 | 0 | 8 |
| 區間上端點 | 0 | 0 | 0 | 0 |
| 仿射 Poisson 二階動差 | 1 | 3 | 0 | 4 |
| 合計／24 | 9 | 11 | 0 | 12 |

**後兩個診斷數不是新的放行成績。** 原規則要求每類至少 7/8、總計至少 22/24；兩組未過。兩筆 A 的 pending 已確認是正確的部分常數折疊；B 提取後的四筆等價式也已逐式核對。原始 raw、原自動分與原 gate 不覆寫。

### 真正查到的三種問題

1. **A 有實際數學錯誤，不只是 parser。** 區間 `[25,177]`、樣本量 ×9，老師回 `25+(177-25)/3`；正確是 `(25+177)/2+(177-25)/(2*3)`。不能把下端點當成固定中心。八題區間均未得到可接受答案，其中一題還有多餘引號。
2. **B 的 0/24 是輸出流程失敗，不能解讀為零數學能力。** 13 筆有完整 fenced JSON：其中 12 筆的最終式與全部中間量可驗證正確，1 筆把 `(175-57)/2` 算成 58 而非 59。另 11 筆 `finish_reason=length`、恰好 400 tokens；大段說明吃掉輸出額度，沒有完整可接受回應。不能替截斷答案補寫結尾。
3. **數學等價判定仍有覆蓋缺口。** 例如 `(4*78+26)**2+16*78` 的 `16=4**2` 是合理折疊，不是錯式。這次只在獨立診斷註記，不用觀察後新增規則去改原 gate。

因此不宜直接擴充同一生成配方，也不能說 3B 無法學會：本輪根本尚未訓練學生。更不能把 A 與 B 的 11/24、12/24 當成顯著能力差异；B 同時改变提示結構與輸出要求，且受 token 上限影響。

### 下一個值得做的決策

**优先修教師輸出契約，不先加教材量、不合併 adapter、不立刻換模型。** 下一個小實驗應限制只輸出精簡 JSON（無前言、無 Markdown、無展開講解），在開跑前納入保守且唯一的 fenced-JSON 提取與已驗證常數折疊；仍拒絕錯值、歧義與截斷。不直接擴大到 full。若仍需要放寬原 400-token 上限或改用 reference-assisted 教材，須另立實驗與費用規則。這是新的研究決策，依已提交的 A/B protocol 停在此處，沒有擅自加跑。

### 保存與重現

- 預期／門檻先存於 commit `6ef0f87df687d2bebc1d6e77e05255fabf484406`；實際程式與新開發題固定於 `a14d4310fd80c0112e4edac559b5ee387e559008`。
- `STATS_TEACHER_AB_EVIDENCE.json`：48 筆完整 raw、request、provider、finish reason、用量、原自動分及 manifest。
- `STATS_TEACHER_AB_REVIEW.json`：逐題離線診斷，待語意判定為 0；可由 `scripts/review_stats_teacher_ab.py` 在沒有既有輸出檔的乾淨 checkout 重建。
- Kaggle 已產生 `3beethoven_teacher_ab.zip`（75,214 bytes），檔案完整性檢查通過。新費用 US$0.00291616；加 V49 共 86 次呼叫、US$0.00693694。未遇到費用缺值；48 筆皆回報指定模型與 DeepInfra。
- 共用 parser 修正使 V49 離線診斷由 58/65 變 59/65，首試可接受 55/65；六題真錯仍是 Poisson 4、interval 2。先前 affine 那一題的首答其實藏在回聲 envelope 內且正確，不再錯歸為數學能力缺失。`STATS_CONSOLIDATION_PILOT_REGRADE_V3.json` 保留逐題證據；沒有把舊 gate 改成通過。

---

## 歷史快照：04:51 PDT

更新：2026-09-07 04:51 PDT（America/Los_Angeles）。

**現行狀態：Version 49 的完整原答已用修正版 grader 離線重判，結果 58/65，未過 59/65 總門檻；`poisson_process` 2/6、`interval` 0/2 也未過各 family 80% 門檻。付費 full teacher 與 GPU 執行繼續鎖定。**

## Version 49 離線重判結果

| archetype | v2 接受／規劃 | 80% family gate |
|---|---:|---|
| poisson_process | 2/6 | 未過 |
| affine_poisson | 8/9 | 通過 |
| general_moment | 9/9 | 通過 |
| uniform_wait | 4/4 | 通過 |
| binomial | 2/2 | 通過 |
| detection_events | 15/15 | 通過 |
| interval | 0/2 | 未過 |
| chain_poisson_variance | 6/6 | 通過 |
| chain_scaled_variance | 6/6 | 通過 |
| chain_second_moment | 6/6 | 通過 |

- 65 題中 58 題接受、6 題數學真錯、1 題 malformed，待人工等價判定為 0；首輪即接受 49 題。
- 6 題 `chain_scaled_variance` 原始 JSON 只多一個結尾 `}`。重判器只忽略完整 JSON 後多出的右括號，保留原始 bytes，答案本身逐題正確，因此全部恢復信用。
- 另外 10 題屬已審核的結構等價式，包括 `comb(n,1)`、uniform midpoint、Poisson 二階動差的 factored form、以及 affine variance／second-moment 展開式；仍不允許只給最後數字。
- 真失敗集中於三處：Poisson 每分鐘 rate 與秒數換算／variance-vs-second-moment 混淆 4 題；affine 二階動差漏掉 variance term 1 題；interval 有 1 題錯誤縮放、1 題兩次都回聲 request envelope。
- 原始 24 個 pilot request 的 hash 全部相容；本次新增 teacher calls 0。完整摘要見 `STATS_CONSOLIDATION_PILOT_REGRADE_V2.json`。

因此不能把 58/65 四捨五入成通過，也不能因只差一題就啟動 full：兩個 family gate 是獨立失敗。下一個合理動作是只修 `poisson_process` 與 `interval` 的教學／prompt 覆蓋，建立一個新的小 pilot contract 驗證；不重做已經 8 個 family 通過的整包教材，也不先跑 GPU。

## 本次修正與驗收（未增加教師呼叫）

- 原 288 題 holdout 有 36 道非法機率題；根因是用 `101..229` 的共用參數作百分比。現在按題型取樣合法機率，重建 288 題及其中 24 題教師對照；版本及雜湊已更新。舊檔仍可從 commit `d425717` 取回，不修改歷史 v15–v20 成績。
- 新增獨立 Fraction oracle：由分布／有效參數計算期望值，而非把參考式送進同一個計算器自我認證。檢查機率及文字百分比、二項整數條件、變異數非負、Poisson rate／時間、uniform 條件可成立、區間端點與參考值。
- 去重以每個子題的有效參數為準，包括等價時間單位；忽略與所求量無關的 offset。兩個非 pilot 情境重抽；24 個 pilot request 完全不變，可重用原始回答作離線診斷。
- 另抓到選點題 `v17_validation_poisson_scaled_001` 與 replay 的 `3**2*67` 重複；僅在本輪矩陣替換為 `v15_validation_poisson_scaled_000`。112 題規模、類別配額及整數門檻不變，未使用 test 題。
- grader v2 承認經檢核的 `comb(n,1)=n`、一次方、uniform「已等時間＋剩餘時間均值」與 interval 等價式；仍拒絕錯誤單位、剩餘等待取代總等待、未代入變數及只給最後數字。歷史 grader 檔案未改。
- 新 readiness 實際驗證含 replay 的 860 筆題目，合法性錯誤與跨分割有效子題重疊皆為 0；45 項離線測試通過。重跑 readiness 保留既有 38 calls／費用，不再歸零或變成付費開跑就緒。
- 新舊 grader／資料 gate 不可混用；本次完整離線重判已無 pending，但 v2 總門檻與兩個 family gate 都未通過。

重現用命令（不必重新問老師）：

```bash
python scripts/regrade_stats_consolidation_pilot.py \
  --source-root /path/to/extracted/3beethoven_stats_consolidation_teacher \
  --output /path/outside/source/pilot_regrade_v2.json
```

此命令不寫來源檔、不呼叫網路、不產生可放行的訓練 gate；有變更的 request 會標示不可重用。原 42/65 是舊 parser／grader 的自動接受數，v2 的 58/65 才是目前診斷值，但兩者都沒有通過 gate。

限制：修正的是已確認的出題／分割／判分錯誤，不等於已證明自然語言語義全部無誤或跨句式泛化。教材仍是參數化模板；既有順序是故事內排列、故事間混合 replay，不能稱為完整的全體 1→2→3 課程。本次沒有追加教材、改 teacher prompt、改 replay 比例或更動兩起點×兩 seed 研究設計。

## 歷史 pilot 與先前修正紀錄（由上方現行狀態取代）

| 階段 | 狀態 | 固定結果／門檻 |
|---|---|---|
| 高階審核 | 已完成 | `STATS_CONSOLIDATION_REVIEW.md` |
| 96 情境參數組 | 已修正並凍結 | 71 train／25 validation；198／70 targets；每題可獨立作答 |
| 教師 pilot | 已完成、未通過 | 42/65；門檻 59/65；38 calls；reported cost 約 US$0.00402；原始證據在 Kaggle Version 49 |
| 教師 full | 已封鎖 | pilot gate 未過，程式拒絕 full |
| 四次訓練 | 尚未開始 | fresh base 與原 v15，各 seeds 2027／31415；一次曝光、最多 49 updates |
| checkpoint 選點 | 已凍結 | 12、24、最後一步；48 舊＋48 組合＋16 事件；pending 不放行 |
| 最終 holdout | 已生成並凍結 | 96 舊＋96 組合＋96 事件＝288；另有永久 MC 240 responses |
| 教師對照 | 已固定、尚未送出 | holdout 中 24 題；教材封存後才呼叫；不得回填訓練 |
| 最終保存 | 程式已備妥 | 每 run 驗證 adapter tensors、manifest、SHA-256、ZIP |

## 本次修正

- 44 道依賴前文的子題改成自足 prompt；教師單題重試也有完整條件。
- 12 組到達題補上 homogeneous Poisson process 假設。
- `v18_conditional_wait` 維持條件下的**總等待時間**，不再偷換成剩餘等待時間。
- 訓練與驗證使用不同表述家族；分母 100 的事件在文字中實際使用百分比。
- selection 改為同場原 v15 相對保留門檻；事件改用四個不同故事。
- 新 grader 隔離歷史版本，分開記錄數學正確、可執行、嚴格格式；任何 pending 先停。
- immutable contract 在開跑前即包含 parent、資料、grader、selection 與 token hash；採 `SequentialSampler`，只跑一個 epoch，不循環補足步數。
- 教師採持久 ledger；每次請求先預留 US$0.01，pilot 未過不能進 full。

## 歷史 v1 parser 的 pilot 自動接受結果

| archetype | 接受／規劃 | gate |
|---|---:|---|
| poisson_process | 1/6 | 未過 |
| affine_poisson | 4/9 | 未過 |
| general_moment | 9/9 | 通過 |
| uniform_wait | 0/4 | 未過 |
| binomial | 1/2 | 未過 |
| detection_events | 15/15 | 通過 |
| interval | 0/2 | 未過 |
| chain_poisson_variance | 6/6 | 通過 |
| chain_scaled_variance | 0/6 | 未過 |
| chain_second_moment | 6/6 | 通過 |

這張表保留歷史稽核用途，已由上方 v2 完整重判取代。舊 parser 把多一個結尾右括號的完整 JSON、以及合法等價式一起算成拒收；v2 修正後為 58/65，但仍因真錯與 family gate 未過而停止。

Kaggle 證據：[Version 49](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=347939910&select=3beethoven_stats_consolidation_teacher.zip)。該 ZIP 含 38 次原始回答、持久 ledger、逐故事 accepted、pilot gate、usage、manifest 與 SHA-256；沒有教師 benchmark 或 GPU 結果。

## Readiness 證據

- 付費前 `STATS_CONSOLIDATION_READINESS.json` 為 `ready_for_paid_pilot`，9/9 離線檢查通過；pilot 後更新為 `pilot_failed_research_decision_required`。
- 45 項離線測試通過，涵蓋自足 prompt、Poisson 假設、等待時間目標、split／parameter 衝突、112 題選點、pending gate、pilot/full gate、contract 恢復、288 題 holdout、教師子集、等價式與受限 JSON transport repair。
- 固定 SHA-256：candidate stories `87f6a75e...a6061e`；selection `10e7e222...221c89`；holdout `c30e1cf1...ab7183`；teacher benchmark `9fdeea63...95133`。

目前停止點符合協議：不進 full、不啟動 GPU。下一個高階決策只剩要不要批准「兩個失敗 family 的定向教材／prompt 修補＋新小 pilot」；其餘兩起點×兩 seed、112 題選點與 288＋240 最終評估規則維持不變。
