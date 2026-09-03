"""Llama 3.3 70B teacher competency diagnostic v0.1.

Purpose:
- Test whether the proposed cloud teacher is actually strong enough in statistics/probability
  before generating any distillation dataset.
- This is a teacher-only benchmark. Do not train on these questions.

Environment:
- Kaggle notebook
- OPENROUTER_API_KEY stored in Kaggle Secrets

Scoring:
- 40 multiple-choice questions
- 5 categories x 8 questions
- balanced correct-answer positions: A/B/C/D = 10 each
"""

import re
import time
from collections import Counter

import pandas as pd
import requests
from kaggle_secrets import UserSecretsClient

MODEL = "meta-llama/llama-3.3-70b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# (category, question, choices, answer)
BENCHMARK = [
    # probability_bayes
    ("probability_bayes", "A disease affects 1% of a population. A test has 95% sensitivity and 90% specificity. If a randomly selected person tests positive, approximately what is P(disease | positive)?", ["A. 95%", "B. 8.8%", "C. 50%", "D. 90%"], "B"),
    ("probability_bayes", "Two fair six-sided dice are rolled. What is the probability that their sum is 9?", ["A. 1/9", "B. 1/6", "C. 1/12", "D. 5/36"], "A"),
    ("probability_bayes", "Events A and B are independent with P(A)=0.6 and P(B)=0.5. What is P(A union B)?", ["A. 0.30", "B. 1.10", "C. 0.80", "D. 0.50"], "C"),
    ("probability_bayes", "A standard deck is known to contain a face card. Given that information, what is the probability the card is red?", ["A. 1/4", "B. 1/3", "C. 2/3", "D. 1/2"], "D"),
    ("probability_bayes", "A fair coin is flipped 3 times. What is the probability of exactly 2 heads?", ["A. 3/8", "B. 1/8", "C. 1/2", "D. 5/8"], "A"),
    ("probability_bayes", "P(A)=0.4, P(B)=0.5, and P(A and B)=0.2. Which statement is correct?", ["A. A and B are mutually exclusive", "B. A and B are independent", "C. P(A|B)=0.2", "D. P(A union B)=1.1"], "B"),
    ("probability_bayes", "A factory has machines M1 and M2 producing 60% and 40% of items. Their defect rates are 1% and 3%. Given a defective item, approximately what is P(M2 produced it)?", ["A. 25%", "B. 40%", "C. 67%", "D. 75%"], "C"),
    ("probability_bayes", "If X and Y are mutually exclusive events with P(X)=0.3 and P(Y)=0.4, what is P(X|Y)?", ["A. 0.7", "B. 0.3", "C. 0.4", "D. 0"], "D"),

    # distributions_expectation
    ("distributions_expectation", "If X is Bernoulli(p=0.3), what is Var(X)?", ["A. 0.21", "B. 0.30", "C. 0.09", "D. 0.70"], "A"),
    ("distributions_expectation", "For X ~ Binomial(n=20,p=0.4), what is E[X]?", ["A. 4", "B. 8", "C. 12", "D. 20"], "B"),
    ("distributions_expectation", "For a Poisson random variable with lambda=5, which pair gives its mean and variance?", ["A. mean=5, variance=25", "B. mean=25, variance=5", "C. mean=5, variance=5", "D. mean=sqrt(5), variance=5"], "C"),
    ("distributions_expectation", "If Z is standard normal, approximately what fraction of observations lie within two standard deviations of the mean?", ["A. 68%", "B. 75%", "C. 99.7%", "D. 95%"], "D"),
    ("distributions_expectation", "If Y=3X+2 and Var(X)=4, what is Var(Y)?", ["A. 36", "B. 14", "C. 12", "D. 18"], "A"),
    ("distributions_expectation", "Which distribution is memoryless?", ["A. Normal", "B. Exponential", "C. Uniform", "D. Beta"], "B"),
    ("distributions_expectation", "For independent random variables X and Y with Var(X)=2 and Var(Y)=3, what is Var(X+Y)?", ["A. 1", "B. 6", "C. 5", "D. sqrt(5)"], "C"),
    ("distributions_expectation", "A normal distribution has mean 100 and standard deviation 15. What z-score corresponds to x=130?", ["A. 1", "B. 1.5", "C. 3", "D. 2"], "D"),

    # inference_testing
    ("inference_testing", "A two-sided hypothesis test yields p=0.03 with alpha=0.05. What is the standard decision?", ["A. Reject H0", "B. Accept H0 as proven true", "C. Increase alpha until p>alpha", "D. Conclude the effect is practically large"], "A"),
    ("inference_testing", "What does a 95% confidence interval procedure mean in frequentist statistics?", ["A. There is a 95% probability this fixed interval contains the fixed parameter", "B. Over repeated samples, 95% of intervals constructed this way contain the true parameter", "C. 95% of the population lies inside the interval", "D. The null hypothesis has 5% probability"], "B"),
    ("inference_testing", "Holding effect size and alpha fixed, what generally happens to statistical power as sample size increases?", ["A. It decreases", "B. It stays exactly constant", "C. It increases", "D. It becomes equal to alpha"], "C"),
    ("inference_testing", "A Type I error is:", ["A. Failing to reject a false null hypothesis", "B. Estimating the wrong effect size", "C. Using a biased sample", "D. Rejecting a true null hypothesis"], "D"),
    ("inference_testing", "Which test is designed to compare means from the same subjects measured before and after an intervention?", ["A. Paired t-test", "B. Independent-samples t-test", "C. Chi-square goodness-of-fit test", "D. One-way ANOVA for independent groups"], "A"),
    ("inference_testing", "If 20 independent hypotheses are each tested at alpha=0.05 with no correction and all null hypotheses are true, the expected number of false positives is:", ["A. 0.05", "B. 1", "C. 5", "D. 10"], "B"),
    ("inference_testing", "Which statement about p-values is correct?", ["A. p is the probability H0 is true", "B. p is the probability the result occurred by chance", "C. p is the probability, assuming H0, of data at least as extreme as observed", "D. p is the probability the alternative is true"], "C"),
    ("inference_testing", "All else equal, changing a test from two-sided to a justified one-sided test typically does what to power in the specified direction?", ["A. Always halves the sample size exactly", "B. Makes Type I error zero", "C. Has no effect", "D. Increases power"], "D"),

    # regression_causality
    ("regression_causality", "In simple linear regression y=b0+b1*x+error, what does b1 represent?", ["A. Expected change in y for a one-unit increase in x", "B. Correlation squared", "C. Residual standard deviation", "D. Probability x causes y"], "A"),
    ("regression_causality", "A correlation of r=0 between X and Y implies:", ["A. X and Y are independent in all cases", "B. There is no linear association, though nonlinear dependence may exist", "C. X cannot cause Y", "D. The variables have identical distributions"], "B"),
    ("regression_causality", "Which issue occurs when a predictor in a regression model is strongly linearly related to other predictors?", ["A. Heteroskedasticity", "B. Autocorrelation", "C. Multicollinearity", "D. Survivorship bias"], "C"),
    ("regression_causality", "Random assignment in an experiment primarily helps with causal inference because it:", ["A. Guarantees a large effect", "B. Eliminates all measurement error", "C. Guarantees perfect compliance", "D. Balances confounders in expectation across treatment groups"], "D"),
    ("regression_causality", "If residual variance increases systematically with fitted values in a linear regression, this is called:", ["A. Heteroskedasticity", "B. Multicollinearity", "C. Simpson's paradox", "D. Complete separation"], "A"),
    ("regression_causality", "Which statement best describes omitted-variable bias?", ["A. Bias caused only by too many predictors", "B. Bias in an estimated effect when an omitted variable is related to both the included predictor and outcome", "C. Random sampling noise", "D. Bias caused by centering predictors"], "B"),
    ("regression_causality", "In logistic regression, exp(beta) for a predictor is commonly interpreted as:", ["A. A probability", "B. A risk difference", "C. An odds ratio for a one-unit predictor increase, holding other variables fixed", "D. A correlation coefficient"], "C"),
    ("regression_causality", "Simpson's paradox refers to a situation where:", ["A. A sample mean equals the population mean", "B. Two variables are perfectly correlated", "C. A p-value equals alpha", "D. An association in aggregated data reverses or disappears after conditioning on groups"], "D"),

    # data_reasoning
    ("data_reasoning", "A dataset has values [1,2,3,4,100]. Which measure of center is most resistant to the outlier?", ["A. Median", "B. Mean", "C. Range", "D. Variance"], "A"),
    ("data_reasoning", "A survey about workplace satisfaction is sent voluntarily to all employees, but mainly very happy and very unhappy employees respond. The main concern is:", ["A. Instrument calibration", "B. Self-selection bias", "C. Regression dilution", "D. Ecological fallacy"], "B"),
    ("data_reasoning", "A classifier has 90 true positives, 10 false negatives, and 30 false positives. What is recall?", ["A. 0.60", "B. 0.75", "C. 0.90", "D. 0.97"], "C"),
    ("data_reasoning", "A classifier sees 1% positive cases and predicts every case as negative. Its accuracy is 99%. What is the best conclusion?", ["A. The classifier is excellent", "B. Accuracy proves high recall", "C. The dataset must be wrong", "D. Accuracy is misleading under severe class imbalance"], "D"),
    ("data_reasoning", "If every observation in a dataset is increased by 10, what happens to the standard deviation?", ["A. It stays the same", "B. It increases by 10", "C. It is multiplied by 10", "D. It becomes zero"], "A"),
    ("data_reasoning", "Which scenario most directly violates the assumption of independent observations?", ["A. Measuring one value from each randomly sampled person", "B. Treating repeated measurements from the same person as if they came from different independent people", "C. Increasing sample size", "D. Standardizing a variable"], "B"),
    ("data_reasoning", "Precision is TP/(TP+FP). If TP=40 and FP=10, precision is:", ["A. 0.20", "B. 0.40", "C. 0.80", "D. 0.90"], "C"),
    ("data_reasoning", "A model performs extremely well on its training data but poorly on new test data. The most likely general diagnosis is:", ["A. Underfitting", "B. Perfect calibration", "C. Randomization", "D. Overfitting"], "D"),
]


def ask_teacher(api_key, item):
    category, question, choices, expected = item
    prompt = f"""Solve this statistics/probability multiple-choice question carefully.

Question:
{question}

{chr(10).join(choices)}

Return ONLY one letter: A, B, C, or D.
"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are being objectively benchmarked. Calculate carefully. Output only the requested answer letter."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 8,
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else "INVALID", text


def main():
    secrets = UserSecretsClient()
    api_key = secrets.get_secret("OPENROUTER_API_KEY")

    expected_dist = Counter(x[3] for x in BENCHMARK)
    print("Expected answer distribution:", dict(sorted(expected_dist.items())))
    print("\nRunning Llama 3.3 70B statistics teacher diagnostic...\n")

    rows = []
    for idx, item in enumerate(BENCHMARK, 1):
        category, question, choices, expected = item
        try:
            predicted, raw = ask_teacher(api_key, item)
        except Exception as exc:
            predicted, raw = "ERROR", str(exc)
        correct = predicted == expected
        rows.append({
            "id": idx,
            "category": category,
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "raw": raw,
        })
        print(f"{idx:02d}/{len(BENCHMARK)} {'✅' if correct else '❌'} {category} expected={expected} got={predicted}")
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    print("\n==============================")
    print("70B TEACHER RESULTS")
    print("==============================")
    print(f"Overall accuracy: {df['correct'].mean():.1%}")
    print("\nAccuracy by category:")
    print(df.groupby("category")["correct"].mean().sort_values())
    print("\nPredicted answer distribution:")
    print(df["predicted"].value_counts().sort_index())
    print("\nIncorrect answers:")
    print(df.loc[~df["correct"], ["id", "category", "question", "expected", "predicted"]].to_string(index=False))


if __name__ == "__main__":
    main()
