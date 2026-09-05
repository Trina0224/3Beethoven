"""Evaluation-only transfer probes; never used as student training targets.

24 new items in six parameterized families, frozen before v0.4 training.
These are internally authored probes, not 24 independent task families.
"""
from collections import Counter
from fractions import Fraction
from stats_v0_3_common import normalize_question, digest


def questions():
    rows=[]
    def add(category, question, correct, wrong, reason):
        choices=list(map(str, wrong)); pos=len(rows)%4
        choices.insert(pos,str(correct))
        rows.append(dict(id=f"transfer_{len(rows)+1:02d}",category=category,
                         question=question,choices=choices,answer_letter="ABCD"[pos],
                         reference_reason=reason))
    for rate in (5,7,11,13):
        add("poisson",f"A counter records a homogeneous Poisson process at {rate} events per hour. Let S be the total count over two non-overlapping intervals of two hours each. What is Var(S)?",
            4*rate,[rate,2*rate,16*rate],"Independent increments: variance equals rate times total duration, four hours.")
    for mean in (3,5,7,9):
        add("expectation",f"X has mean {mean} and variance 4. What is E[(X-1)^2]?",
            4+(mean-1)**2,[(mean-1)**2,4+mean**2,mean-1],
            "E[(X-1)^2]=Var(X)+(E[X]-1)^2.")
    for width in (6,12,18,24):
        add("uniform",f"X is continuously uniform on [0,{width}]. What is E[X^2]?",
            width**2//3,[width**2//12,width**2//2,width**2],
            "E[X^2]=Var(X)+E[X]^2=w^2/12+w^2/4=w^2/3.")
    for n in (3,4,5,6):
        add("type_i",f"There are {n} independent tests, each with a true null and exact Type I error probability 0.10. What is the probability that exactly one test rejects?",
            f"{n}*0.10*0.90^{n-1}",[f"0.10^{n}",f"1-0.90^{n}","0.10"],
            "Exactly one rejection has binomial probability n*p*(1-p)^(n-1).")
    for beta in (10,20,30,40):
        b=str(Fraction(beta,100)); power=str(Fraction(100-beta,100))
        add("type_ii",f"Two independent studies test the same specified false null. Each has Type II error probability {beta/100:.2f}. What is the probability that at least one study rejects?",
            f"1-({b})^2",[f"({power})^2",f"({b})^2",power],
            "Both studies miss with probability beta squared; take the complement.")
    for se in (2,3,4,5):
        add("confidence",f"A normal-theory interval has standard error {se} and critical value 2. Keeping confidence and population standard deviation fixed, sample size is multiplied by four. What is the new total interval width?",
            2*se,[4*se,se,8*se],
            "Quadrupling n halves SE. New total width is 2*2*(SE/2)=2*SE.")
    return rows


def validate(excluded):
    rows=questions()
    keys=[normalize_question(r['question']) for r in rows]
    assert len(rows)==len(set(keys))==24
    assert not set(keys)&{normalize_question(r['question']) for r in excluded}
    assert Counter(r['answer_letter'] for r in rows)==dict.fromkeys('ABCD',6)
    assert set(Counter(r['category'] for r in rows).values())=={4}
    assert all(len(set(r['choices']))==4 for r in rows)
    return dict(n=24,families=6,sha256=digest(rows),exact_overlap=0)
