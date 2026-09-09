# 執行交接：step 180 正式 final blind 評測完成

## 已完成

- 唯一訓練軌跡：720 new＋720 replay，180 optimizer updates。
- 最新 adapter SHA-256：`ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`。
- Diverse final blind：180 題、18 類各 10 題；在訓練與選點期間保持封存。
- 原始 3B：45/180，strict format 26/180，66 pending。
- step 180：170/180，strict format 180/180，0 pending。
- 配對結果：`0→0: 8`、`0→1: 127`、`1→0: 2`、`1→1: 43`。
- 學生 10 題失敗與 2 題 paired loss 已逐題檢查，沒有 grader 誤殺。

## 重現規則

使用固定 revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` 的 `meta-llama/Llama-3.2-3B-Instruct`。Baseline 不掛任何 adapter；學生從另一份新載入的相同 base 直接掛 step 180 LoRA。兩者都使用 base model tokenizer。不要使用 adapter ZIP 內舊的 `tokenizer_config.json`，其 `TokenizersBackend` 類別會在 Colab Transformers 環境造成載入錯誤；目前 evaluator 已固定從 base revision 載入 tokenizer。

完整重跑方式見 [Colab 交接](STATS_DIVERSE_FINAL_BLIND_COLAB_HANDOFF.md)。完整輸出已保存於 repo 根目錄 [`final_blind_eval_results.zip`](../final_blind_eval_results.zip)。

## 解讀

這次正式產品比較足以判定限定統計列式蒸餾成功。舊 v15 frozen selection protocol 的失敗結論不追溯修改；它描述當時的選點合規性，本次描述最終學生相對原始 3B 的實際能力。

[正式報告](STATS_DIVERSE_FINAL_BLIND_RESULTS.md) · [機器摘要](STATS_DIVERSE_FINAL_BLIND_RESULTS.json) · [目前狀態](STATS_CURRENT_STATUS.md)
