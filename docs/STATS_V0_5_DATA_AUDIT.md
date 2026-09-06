# v0.5 教材品質檢查

## 完成範圍

- 180 題訓練、24 題驗證，合計 204 題 Llama 教材；另有 36 題保留測試。
- 題目、選項、數學參考答案與分割在教師生成前凍結。三個分割的題型家族在本次實驗內不重疊。
- 初始接受條件：參考答案字母一致、結構和長度合格、Llama 自我複查通過。這些條件不保證解說正確。
- 助理於學生訓練前逐題讀取全部 204 題解說與常見錯誤，標記 27 題數學錯誤或容易誤導的措辭，再讓 Llama 修正，並逐一讀取修正結果。
- 最終 81／204 題曾使用參考答案或檢查回饋；其餘 123 題沒有參考答案條件化。這個比例包含格式修復，不能當作教師初答錯誤率。

## 發現與處理

主要問題包括：把 Poisson 的平均數等於變異數、期望值線性等正確方法稱為錯誤；把二項分布變異數說成平均數平方；加權平均的中間算式不相等；對 Uniform[0,b] 錯誤否認 CDF 與門檻值成正比；混淆獨立性與事件重疊；以及信賴區間縮減「至」與縮減「了」的用語。

27 是需要修正或澄清的教材筆數，不是 27 個獨立的教師答題錯誤。許多問題來自同一家族的相似措辭。檢查改善了教材品質，但不是完整形式化證明。

所有最終學生解說與常見錯誤文字仍來自 `meta-llama/llama-3.3-70b-instruct`。助理僅提供問題診斷與參考檢查，沒有撰寫替代解說作為學生目標。其中 `train_expectation_2_03` 的兩次新修正仍錯誤，因此沿用先前保存、經核對正確的 Llama 回覆，再重新複查；保留其原始回覆及明確更正答案字母的解析紀錄。

## 費用與保存

| 階段 | 請求次數 | 回報費用（美元） |
|---|---:|---:|
| 初始教材生成、複查與 36 題教師測試 | 517 | 0.032661575 |
| 內容修正與複查 | 55 | 0.004915930 |
| 本輪合計 | 572 | 0.037577505 |

費用取自 API 回覆的 usage 欄位，572 次回覆均有費用資料；這不是帳單，也不是帳戶餘額。全程沿用同一份 600 次請求上限及預先寫入的請求帳本，沒有重置額度。

訓練前 Kaggle **第 8 版，script version ID 347596444** 已確認 Successful，且 `3beethoven_stats_v0_5.zip` 出現在輸出清單。ZIP 保存原始 API 快取、請求帳本、複查、修正前資料 `before_content_review/`、最終資料與來源程式。GitHub 的 [教師資料](STATS_V0_5_TEACHER_DATA.json) 保存最終教材、逐筆來源標籤和教師測試；原始回覆來源一致性已由程式逐筆驗證。

## 教師保留測試

教師只測原始選項排列，16 token 上限；不得與學生四種排列總分直接當作相同條件比較。

| 主題 | 答對／題數 |
|---|---:|
| Poisson | 5／6 |
| 期望值 | 4／6 |
| 均勻分布 | 6／6 |
| 第一類錯誤 | 2／6 |
| 第二類錯誤 | 6／6 |
| 信賴區間 | 4／6 |
| 合計 | 27／36（75%） |

教師測試答案沒有作為學生訓練目標，也沒有以教師答案取代數學參考答案。

## 修正清單

- `train_poisson_1_02`：For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.
- `train_poisson_1_03`：For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.
- `train_poisson_1_05`：For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.
- `train_poisson_1_06`：For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.
- `train_poisson_1_07`：For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.
- `train_poisson_1_09`：For independent Poisson variables, summing means gives the same correct result as summing variances. Do not call that identity incorrect. Describe a genuinely wrong calculation.
- `train_poisson_2_00`：Binomial variance is np(1-p), not the squared mean. Remove the false distribution claim.
- `train_poisson_2_04`：Var(3X+6)=Var(3X)+Var(6) is valid because 6 is constant. Do not label this valid identity a mistake.
- `train_poisson_2_07`：For a Poisson variable its mean equals its variance; using that mean in 9*Var(X) is valid. Explain a genuinely wrong scaling operation.
- `train_poisson_2_08`：For a Poisson variable its mean equals its variance; using that mean in 9*Var(X) is valid. Explain a genuinely wrong scaling operation.
- `train_poisson_2_09`：For a Poisson variable its mean equals its variance; using that mean in 9*Var(X) is valid. Explain a genuinely wrong scaling operation.
- `train_expectation_0_01`：Linearity E[3X-c]=E[3X]-E[c]=3E[X]-c is valid, including for a constant. Do not call any of these identities incorrect. Clearly identify an actual omitted term or wrong sign.
- `train_expectation_0_03`：Linearity E[3X-c]=E[3X]-E[c]=3E[X]-c is valid, including for a constant. Do not call any of these identities incorrect. Clearly identify an actual omitted term or wrong sign.
- `train_expectation_0_04`：Linearity E[3X-c]=E[3X]-E[c]=3E[X]-c is valid, including for a constant. Do not call any of these identities incorrect. Clearly identify an actual omitted term or wrong sign.
- `train_expectation_0_09`：Linearity E[3X-c]=E[3X]-E[c]=3E[X]-c is valid, including for a constant. Do not call any of these identities incorrect. Clearly identify an actual omitted term or wrong sign.
- `train_expectation_2_03`：The intermediate equality 24+4=26 is false. Recompute both weighted terms and verify their sum.
- `train_uniform_2_00`：For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.
- `train_uniform_2_01`：For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.
- `train_uniform_2_02`：For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.
- `train_uniform_2_03`：For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.
- `train_uniform_2_05`：For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.
- `train_uniform_2_06`：For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.
- `train_uniform_2_07`：For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.
- `train_uniform_2_08`：For Uniform[0,b], P(X<x)=x/b is proportional to x for fixed b. Calling proportionality to the threshold or dividing by the maximum incorrect is false here. Give a genuinely incorrect probability calculation, clearly distinguished from the correct one.
- `train_type_ii_2_05`：The studies are independent. Adding rejection probabilities fails because the events overlap, not because of dependence. Correct the misconception.
- `train_confidence_1_01`：Reducing width BY one third means keeping two thirds; this question asks reducing width TO one third. Use precise wording consistent with the question.
- `validation_confidence_3_00`：Here standard error and critical value both happen to equal 2. Explain why blindly equating them is not a valid general method without falsely denying their numerical equality in this case.
