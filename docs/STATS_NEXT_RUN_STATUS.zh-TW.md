# 下一步狀態：正式評測已完成

目前沒有訓練或評測正在執行。最新 step 180 學生在 diverse final blind 得到 **170/180（94.4%）**，原始 3B 為 **45/180（25.0%）**；學生 180/180 遵守嚴格一行算式格式，0 pending。這一輪工作已到可交付段落。

目前固定決定：

- step 180 是本專案目前成功學生，模型身份以 adapter SHA-256 `ff89a22f097e4db457dab287042ae36520ec6b9b36f26e5ed11fe42337c04f23` 固定。
- 正式 headline baseline 是原始 Llama 3.2 3B，而非 v15。
- 不再追加 seed、擴大語文能力或為追求 180/180 繼續燒算力。
- 若未來另開局部修補，優先處理 `process_scaled`（7/10）與 `moment_second`（8/10），並使用新考卷，不能重複用本次 final blind 做模型選擇。
- 舊 frozen v15 selection 的 `no_checkpoint_passed` 是歷史 protocol 結論，與後續產品評測成功並列保留。

[正式結果](STATS_DIVERSE_FINAL_BLIND_RESULTS.md) · [目前狀態](STATS_CURRENT_STATUS.md) · [Colab 重跑](STATS_DIVERSE_FINAL_BLIND_COLAB_HANDOFF.md)
