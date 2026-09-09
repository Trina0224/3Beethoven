# 3Beethoven

## 最新權重：Hugging Face

正式模型頁：[kozakurayuki/3Beethoven-step180](https://huggingface.co/kozakurayuki/3Beethoven-step180)。權重與 `adapter_config.json` 已上傳；遠端 LFS SHA-256 與完成 170/180 評測的 step 180 權重一致。下方 Kaggle ZIP 保留為歷史備份。

- HF revision：`682d5555b0e6115135ac7b9b5d718d2abef186de`
- Adapter SHA-256：`ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`
- [直接下載 adapter_model.safetensors](https://huggingface.co/kozakurayuki/3Beethoven-step180/resolve/682d5555b0e6115135ac7b9b5d718d2abef186de/adapter_model.safetensors?download=true)

這是 LoRA adapter，載入時仍需固定 revision 的原始 Llama 3.2 3B Instruct；不要疊加 v15。Tokenizer 使用原始 base。


## 目前狀態：response distillation 成功

最新 step 180 學生已在先前封存、未用於訓練或選點的 180 題 diverse final blind 上完成正式評測。自然產品比較使用原始 `meta-llama/Llama-3.2-3B-Instruct` 作 baseline：原始 3B 為 **45/180（25.0%）**，最新學生為 **170/180（94.4%）**，淨增加 125 題。學生同時達到 **180/180 嚴格一行可執行算式、0 pending**。

18 個類別中，學生相對原始 3B 有 17 類提高、1 類持平、0 類下降。配對結果有 127 題由錯轉對、2 題由對轉錯；逐一檢查學生剩餘 10 題後，均為真實列式錯誤，沒有 grader 誤殺。這支持本專案的限定主張：學生學會約定範圍內的直接統計列式能力；不延伸成一般語文或未知統計領域能力。

[正式 final blind 結果](docs/STATS_DIVERSE_FINAL_BLIND_RESULTS.md) · [機器可讀摘要](docs/STATS_DIVERSE_FINAL_BLIND_RESULTS.json) · [完整結果 ZIP](final_blind_eval_results.zip)

| 模型 | 目前定位 |
|---|---|
| 原始 Llama 3.2 3B Instruct | 正式自然 baseline：45/180 |
| **多樣教材 step 180** | **目前成功學生：170/180；180/180 strict format；0 pending** |
| 原 v15 | 歷史訓練起點與 selection 錨點；不再作最終產品 headline baseline |
| final_clear / V58 | 歷史窄模板結果；採用資格維持撤回 |

學生由原 v15 接續訓練，使用 720 筆新 train 與 720 筆物化歷史 train replay，固定 1:1 組成 1,440-row 訓練序列，共 180 optimizer updates；最後記錄 loss 約 `0.20015`。step 180 adapter SHA-256 為 `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23`。

## 下載最新學生

權重位於 [Kaggle Version 63](https://www.kaggle.com/code/trinashih/3beethoven-v0-2/output?scriptVersionId=348384546)，檔名 `3Beethoven_latest_step180_weights.zip`。ZIP SHA-256 為 `4a97c1070fd81ff2fb570699eb630043562b9b4968b799dfed6dfb410fdf036e`；其中 adapter SHA-256 如上。把 step 180 LoRA 直接掛在固定 revision 的原始 3B base，不要疊加 v15。

## 實驗解讀

先前凍結的 v15 selection protocol 要求逐題零 paired loss，當時 step 180 在 development 有兩題 loss，因此歷史 selection 結論仍是 `no_checkpoint_passed`。後續依使用者明確指定，正式產品比較改以原始 3B 對最新學生，並解封 final blind；這個結果證明蒸餾目標成功，但不倒推改寫舊 protocol 的執行偏差或合規結論。

## 文件入口

- [正式 final blind 結果](docs/STATS_DIVERSE_FINAL_BLIND_RESULTS.md) · [Colab 重跑說明](docs/STATS_DIVERSE_FINAL_BLIND_COLAB_HANDOFF.md)
- [訓練與 development 歷史結果](docs/STATS_DIVERSE_RUN_RESULTS.md) · [目前狀態](docs/STATS_CURRENT_STATUS.md)
- [受控實驗協議](docs/STATS_DIVERSE_EXPERIMENT_PROTOCOL.md) · [執行交接](docs/STATS_EXECUTION_HANDOFF.md)
- [多樣教材協議](docs/STATS_DIVERSE_CURRICULUM_PROTOCOL.md) · [實際 replay rows](docs/STATS_DIVERSE_REPLAY.json)
- [V58 撤回後記](docs/STATS_V58_POSTMORTEM.md) · [完整研究流程](docs/STATS_V01_V19_RESEARCH_FLOW.md)

原始音樂概念與早期實驗保留為歷史背景。Kaggle 版本號只是 notebook 保存版本；模型身份以 base revision、adapter 路徑與 SHA-256 判定。
