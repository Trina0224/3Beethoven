"""Evaluation-only questions authored after training; never student targets."""
import json
from collections import Counter
from stats_v0_3_common import digest, normalize_question, prompt_for

# Correct option text is separate from distractors so option rotation cannot
# silently change the gold answer. Six concepts, ten task forms each.
ITEMS = [
 ("poisson", "A homogeneous Poisson process has rate 4 arrivals per minute. What is the variance of the number of arrivals in a three-minute window?", "12", ["4", "36", "144"], "Poisson parameter over three minutes is 4*3=12; variance equals this parameter."),
 ("poisson", "Independent counts X and Y have Poisson distributions with parameters 2 and 5. What is Var(X+Y)?", "7", ["3", "10", "49"], "Independence gives Var(X+Y)=Var(X)+Var(Y)=2+5=7."),
 ("poisson", "Nine independent observations each have a Poisson distribution with mean 18. What is the variance of their sample mean?", "2", ["18", "162", "6"], "Variance of an independent sample mean is 18/9=2."),
 ("poisson", "A count distribution has exact theoretical mean 8 and exact theoretical variance 20. Can this distribution be Poisson with any parameter?", "No, because a Poisson distribution has equal theoretical mean and variance", ["Yes, with parameter 8", "Yes, with parameter 20", "Yes, with parameter 12"], "Unequal theoretical mean and variance rule out a Poisson distribution; these are not noisy sample estimates."),
 ("expectation", "Random variables X and Y have finite means E[X]=4 and E[Y]=7. Independence is not assumed. What is E[3X-2Y+5]?", "3", ["31", "26", "Cannot be determined without independence"], "Linearity does not require independence: 3*4-2*7+5=3."),
 ("expectation", "A random variable X has finite mean and E[7-2X]=1. What is E[X]?", "3", ["4", "6", "-3"], "7-2E[X]=1 implies E[X]=3."),
 ("expectation", "The only information about X and Y is E[X]=2 and E[Y]=3, and all required expectations exist. What must E[XY] equal?", "It cannot be determined from these means alone", ["6", "5", "1"], "E[XY] depends on dependence/covariance; independence was not given."),
 ("expectation", "X and Y both have finite mean 11, but may be dependent and have different distributions. What is E[X-Y]?", "0", ["11", "22", "It is undetermined unless X and Y have identical distributions"], "E[X-Y]=E[X]-E[Y]=0; dependence and different distributions do not change linearity."),
 ("uniform", "X is continuously uniform on [-6,10]. What is E[X]?", "2", ["8", "4", "-2"], "The midpoint is (-6+10)/2=2."),
 ("uniform", "X is continuously uniform on [a,14] and E[X]=9. What is a?", "4", ["5", "9", "18"], "(a+14)/2=9 implies a=4."),
 ("uniform", "X is uniform on [0,10] and Y is uniform on [4,6]. Which statement is correct?", "They have equal means, but X has larger variance", ["They have equal means and equal variances", "X has a larger mean, but equal variance", "Y has a larger mean and larger variance"], "Both midpoints are 5, while variances are 100/12 and 4/12."),
 ("uniform", "A total waiting time X is uniform on [0,12] minutes. Given that X exceeds 4 minutes, what is the expected remaining wait X-4?", "4 minutes", ["6 minutes", "8 minutes", "12 minutes"], "Conditional X is uniform on (4,12), with mean 8; remaining mean is 8-4=4."),
 ("type_i", "A drug trial tests H0: the drug has no effect. The drug truly has no effect, but the trial rejects H0 and announces an effect. What happened?", "A Type I error", ["A Type II error", "A correct rejection", "A correct non-rejection"], "Rejecting a true null is a Type I error."),
 ("type_i", "A test of a simple null hypothesis has exact size alpha=0.02. When that null is true, what is the probability of rejecting it?", "0.02", ["0.98", "0.50", "It equals the test's power"], "For a simple null, exact test size is the rejection probability under that null."),
 ("type_i", "Two hundred tests each have a true null and an exact Type I error probability of 0.05. What is the expected number of false rejections?", "10", ["5", "20", "190"], "Linearity of expectation gives 200*0.05=10; independence is unnecessary for this expectation."),
 ("type_i", "For the same test statistic and sample size, a researcher uses a strictly smaller rejection region. What happens to the Type I error probability?", "It cannot increase", ["It must increase", "It always becomes exactly zero", "It becomes equal to the Type II error probability"], "Under a true null, a subset rejection event cannot have greater probability."),
 ("type_ii", "A test uses H0: a machine's defect rate has not increased. In fact the rate has increased, but the test fails to reject H0. What happened?", "A Type II error", ["A Type I error", "A correct rejection", "A correct non-rejection"], "Failing to reject a false null is a Type II error."),
 ("type_ii", "At a specified alternative, a test has power 0.85. What is its Type II error probability at that alternative?", "0.15", ["0.85", "0.05", "0.95"], "Power=1-beta, so beta=1-0.85=0.15."),
 ("type_ii", "One hundred tests are conducted in situations where each null is false. Each test has Type II error probability 0.20 at its actual alternative. What is the expected number of failures to reject?", "20", ["80", "5", "100"], "Expected failures to reject false nulls equal 100*0.20=20."),
 ("type_ii", "A report only says that a test failed to reject H0. It does not reveal whether H0 is true. Can this decision already be classified as a Type II error?", "No; that classification also requires H0 to be false", ["Yes; every failure to reject is a Type II error", "No; every failure to reject is a Type I error", "Yes; failure to reject proves H0 is false"], "A failure to reject is correct when H0 is true and Type II when H0 is false."),
 ("confidence", "For a normal-theory mean interval with known population standard deviation, keep confidence level fixed and quadruple sample size. What happens to the interval width?", "It is halved", ["It doubles", "It is quartered", "It is unchanged"], "Width is proportional to 1/sqrt(n), so multiplying n by 4 halves width."),
 ("confidence", "A two-sided normal-theory confidence interval uses standard error 3 and critical value 2.576. What is its total width?", "15.456", ["7.728", "5.152", "11.76"], "Total width is 2*2.576*3=15.456, not the one-sided margin of error."),
 ("confidence", "Which is the frequentist meaning of a valid 95% confidence-interval procedure under its model assumptions?", "Across repeated samples, about 95% of the intervals cover the fixed true parameter", ["There is a 95% probability that the fixed parameter is in this particular observed interval", "95% of individual observations must be inside the interval", "The next observation has a 95% chance of equaling the sample mean"], "Coverage concerns the repeated-sampling random intervals, not a posterior probability for a fixed parameter."),
 ("confidence", "A confidence interval has endpoints 10 and 18. Keeping its center and critical value fixed, the standard error doubles. What is the new total width?", "16", ["4", "8", "32"], "Original width=18-10=8; width is proportional to standard error, so new width=16."),
]

ITEMS += [
 ("poisson", "A homogeneous Poisson process averages 12 events per hour. What is the expected count in a half-hour observation?", "6", ["12", "24", "3"], "The expected count scales with duration: 12*0.5=6."),
 ("poisson", "X has a Poisson distribution with parameter 3. What is P(X=0)?", "exp(-3)", ["3*exp(-3)", "1-exp(-3)", "1/3"], "The Poisson mass at zero is exp(-lambda)*lambda^0/0!=exp(-3)."),
 ("poisson", "Independent X and Y are Poisson with means 4 and 9. What is Var(X-Y)?", "13", ["5", "-5", "169"], "Variances add for a difference of independent variables: 4+9=13."),
 ("poisson", "X is Poisson with mean 9 and Y=2X. What is Var(Y)?", "36", ["18", "9", "81"], "Var(2X)=4Var(X)=4*9=36."),
 ("poisson", "A Poisson random variable has probability 0.2 of being zero. What is its mean?", "-ln(0.2)", ["0.2", "ln(0.2)", "5"], "P(X=0)=exp(-lambda)=0.2, hence lambda=-ln(0.2)."),
 ("poisson", "Four independent Poisson counts each have parameter 3. What are the mean and variance of their total?", "Mean 12, variance 12", ["Mean 3, variance 3", "Mean 12, variance 48", "Mean 12, variance 144"], "The sum is Poisson with parameter 4*3=12."),
 ("expectation", "E[X]=2 and E[X squared]=7. What is E[(X+1) squared]?", "12", ["9", "8", "10"], "Expanding gives E[X^2]+2E[X]+1=7+4+1=12."),
 ("expectation", "Independent X and Y have finite means 2 and 5. What is E[XY]?", "10", ["7", "3", "It is undetermined despite independence"], "Independence gives E[XY]=E[X]E[Y]=2*5=10."),
 ("expectation", "X is strictly positive and E[X]=4. No distribution or other moments are given. Must E[1/X] equal 1/4?", "No; the mean alone does not determine E[1/X]", ["Yes, always", "No; it must equal 4", "No; it must equal zero"], "Expectation cannot generally be moved through a reciprocal; a constant X and a nonconstant positive X with the same mean differ."),
 ("expectation", "An indicator I equals 1 when event A occurs and 0 otherwise. If P(A)=0.3, what is E[I]?", "0.3", ["0.7", "1", "0"], "E[I]=1*0.3+0*0.7=0.3."),
 ("expectation", "Someone claims a real random variable has E[X]=3 and E[X squared]=5. All moments are finite. Is this possible?", "No, because it would imply negative variance", ["Yes, with variance 2", "Yes, with variance 5", "Yes, with variance 14"], "Var(X)=E[X^2]-E[X]^2=5-9=-4 is impossible."),
 ("expectation", "A random output is drawn from group A with probability 0.25 and group B otherwise. Their conditional means are 8 and 4. What is the overall mean?", "5", ["6", "12", "3"], "Total expectation gives 0.25*8+0.75*4=5."),
 ("uniform", "X is continuously uniform on [1,9]. What is P(X>5)?", "1/2", ["5/9", "1/4", "3/4"], "The favorable interval length is 9-5=4 out of total length 8."),
 ("uniform", "X is continuously uniform on [-3,3]. What is E[absolute value of X]?", "1.5", ["0", "3", "6"], "Absolute X is uniform on [0,3], with mean 1.5."),
 ("uniform", "X is continuously uniform on [2,8]. What is Var(X)?", "3", ["5", "6", "36"], "Uniform variance is (8-2)^2/12=36/12=3."),
 ("uniform", "X is continuously uniform on [0,12]. What is its standard deviation?", "sqrt(12)", ["12", "6", "144"], "Variance is 12^2/12=12; standard deviation is sqrt(12)."),
 ("uniform", "X is continuously uniform on [-2,2]. What is E[X squared]?", "4/3", ["0", "2", "4"], "Mean is zero, so E[X^2]=Var(X)=4^2/12=4/3."),
 ("uniform", "X is continuously uniform on [0,2] and Y=X+5. What is E[Y]?", "6", ["5", "7", "10"], "Uniform mean is 1, and adding 5 gives expectation 6."),
 ("type_i", "Which conditional probability defines the Type I error rate for a simple null?", "P(reject H0 given H0 is true)", ["P(H0 is true given H0 was rejected)", "P(fail to reject H0 given H0 is false)", "P(H0 is false given H0 was not rejected)"], "Type I error conditions on null truth; reversing the conditional is not valid."),
 ("type_i", "Under a true null, a continuous p-value is exactly uniform on [0,1]. A test rejects when p<0.04. What is its Type I error probability?", "0.04", ["0.96", "0.5", "0.16"], "For a uniform p-value, P(p<0.04)=0.04."),
 ("type_i", "Ten independent tests each have a true null and Type I error probability 0.05. What is the probability of at least one false rejection?", "1-0.95^10", ["0.05", "0.95^10", "0.05^10"], "Independence gives probability of no false rejection 0.95^10; take its complement."),
 ("type_i", "Twenty hypotheses are each tested at level 0.0025. Without assuming independence, which upper bound follows for the probability of one or more false rejections?", "0.05", ["0.0025", "0.95", "0.5"], "The union bound is at most 20*0.0025=0.05, and remains valid if fewer nulls are true."),
 ("type_i", "A procedure never rejects any null hypothesis. What is its Type I error probability?", "0", ["1", "0.5", "It depends on the sample size"], "A true null can never be rejected under this procedure."),
 ("type_i", "A significance threshold is 0.05. Does this by itself imply that 5% of all rejected hypotheses are actually true?", "No; that is a different conditional probability", ["Yes, exactly 5%", "Yes, at least 95%", "Yes, provided the sample size exceeds 30"], "Alpha controls rejection conditional on null truth, not null truth conditional on rejection."),
 ("type_ii", "For a fixed nonzero effect and the usual z test with known variance and fixed significance level, increasing sample size has what effect on Type II error probability?", "It decreases", ["It increases", "It must stay exactly constant", "It becomes identical to the significance level"], "A larger sample increases the standardized nonzero effect and power, reducing beta for this fixed-effect z-test setup."),
 ("type_ii", "A test always rejects H0, regardless of data. At a specified false null, what is its Type II error probability?", "0", ["1", "0.5", "It equals the effect size"], "There are no failures to reject a false null when rejection is automatic."),
 ("type_ii", "Two tests have powers 0.60 and 0.90 at the same specified alternative. Which has smaller Type II error probability?", "The test with power 0.90", ["The test with power 0.60", "They have equal Type II error probability", "Neither can have a Type II error"], "Their beta values are 0.40 and 0.10 respectively."),
 ("type_ii", "A medical screening model uses H0: the patient does not have the condition. The patient has it, but the test does not reject H0. Which term matches this Type II error?", "False negative", ["False positive", "True positive", "True negative"], "Failure to detect a condition that is present is a false negative and a failure to reject this false null."),
 ("type_ii", "A study fails to reject H0 and reports low power for an important specified alternative. Which conclusion is justified?", "The nonsignificant result does not rule out that alternative", ["H0 has been proved true", "The alternative has been proved false", "A Type I error certainly occurred"], "Low power means a meaningful alternative can often be missed; nonsignificance is not proof of the null."),
 ("type_ii", "At a fixed alternative, a test has Type II error probability 0.25. In 80 repetitions at that alternative, what is the expected number of rejections?", "60", ["20", "25", "80"], "Power is 0.75, so expected rejections are 80*0.75=60."),
 ("confidence", "For a normal-theory mean interval at fixed confidence and known standard deviation, what sample-size factor is needed to make the width one third as large?", "9", ["3", "1/3", "27"], "Width scales as n^-1/2; reducing it by 3 requires n to increase by 3^2=9."),
 ("confidence", "A valid 90% confidence-interval procedure is applied to 1000 repeated samples under its assumptions. What is the expected number of intervals that miss the fixed true parameter?", "100", ["900", "90", "10"], "Noncoverage probability is 0.10, giving 1000*0.10=100 expected misses."),
 ("confidence", "A two-sided 95% z interval for a mean excludes a specified null mean. Using the matching two-sided z test and the same assumptions, what decision follows at level 0.05?", "Reject that null mean", ["Fail to reject that null mean", "Accept the null with probability 0.95", "The decision cannot be linked to the matching interval"], "Matching two-sided confidence intervals and tests invert one another: exclusion implies rejection."),
 ("confidence", "A normal-theory interval keeps its confidence level and sample size fixed, but the known population standard deviation doubles. What happens to its width?", "It doubles", ["It is halved", "It is unchanged", "It quadruples"], "Width is 2*z*sigma/sqrt(n), proportional to sigma."),
 ("confidence", "A 95% confidence interval for a population mean is [40,50]. Does it claim that 95% of individual population observations lie between 40 and 50?", "No; a mean-confidence interval is not an interval for individual observations", ["Yes, by definition", "Yes, if the sample mean is 45", "Yes, because both endpoints are positive"], "Uncertainty about a mean and dispersion of individual observations are different quantities."),
 ("confidence", "For a two-sided normal-theory interval, the desired half-width is 2 and the critical value is 2. What standard error is required?", "1", ["2", "4", "0.5"], "Half-width equals critical value times standard error: 2=2*SE, so SE=1."),
]


def questions():
    result=[]
    for i,(concept,q,correct,wrong,reason) in enumerate(ITEMS):
        choices=list(wrong)
        position=(i*3+1)%4
        choices.insert(position,correct)
        result.append({"id":f"holdout_{i+1:02d}","category":concept,"question":q,
                       "choices":choices,"answer_letter":"ABCD"[position],"reference_reason":reason})
    return result


def validate(excluded):
    rows=questions()
    assert len(rows)==60
    assert Counter(r["answer_letter"] for r in rows)==dict.fromkeys("ABCD",15)
    assert set(Counter(r["category"] for r in rows).values())=={10}
    keys=[normalize_question(r["question"]) for r in rows]
    assert len(set(keys))==60
    assert not set(keys)&{normalize_question(r["question"]) for r in excluded}
    for r in rows:
        assert len(set(r["choices"]))==4
        assert len(prompt_for(r).encode())<=4000
    return {"questions":60,"per_concept":10,"positions":dict(Counter(r["answer_letter"] for r in rows)),
            "sha256":digest(rows),"exact_overlap":0,"status":"evaluation-only, authored and frozen after training; not externally authored"}


if __name__=="__main__":
    print(json.dumps({"audit":validate([]),"questions":questions()},indent=2))
