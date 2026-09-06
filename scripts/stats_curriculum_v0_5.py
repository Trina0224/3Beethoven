"""Expanded statistics curriculum with disjoint within-run task families.

Code defines exact references, never teacher explanation targets. All fractions
are exact. Related principles and earlier exposed diagnostics are acknowledged.
"""
from fractions import Fraction as F
from collections import Counter
from stats_v0_3_common import digest,normalize_question

TOPICS=('poisson','expectation','uniform','type_i','type_ii','confidence')


def item(topic,family,v):
    k=v+2
    if topic=='poisson':
        if family==0:
            return f'A Poisson process has {k+1} events per hour. What is the variance of its count over {k} hours?',(k+1)*k,[(k+1),(k+1)*k*k,(k+1)**2*k],'A Poisson count has variance equal to rate times duration.'
        if family==1:
            return f'Independent X and Y are Poisson with means {k+2} and {2*k+1}. What is Var(X+Y)?',3*k+3,[k-1,(k+2)*(2*k+1),(3*k+3)**2],'Independence makes the variances add; each Poisson variance equals its mean.'
        if family==2:
            return f'X is Poisson with mean {k+3}. For Y=3X+{k}, what is Var(Y)?',9*(k+3),[3*(k+3),9*(k+3)+k,k+3],'Adding a constant does not change variance; multiplying by 3 multiplies variance by 9.'
        if family==3:
            return f'A sample contains {k} independent Poisson observations, each with mean {3*k}. What is the variance of the sample mean?',3,[3*k,3*k*k,9*k*k],'The variance of an independent sample mean is the individual variance divided by sample size.'
        return f'Independent Poisson counts X and Y have means {k+1} and {k+4}. What is Var(2X-3Y)?',4*(k+1)+9*(k+4),[2*(k+1)+3*(k+4),4*(k+1)-9*(k+4),13*(2*k+5)],'Independent variances add even for a difference, and coefficients are squared.'
    if topic=='expectation':
        if family==0:
            return f'E[X]={k+1}. What is E[3X-{k+2}]?',3*(k+1)-(k+2),[3*(k+1)+(k+2),3*(k+1),(k+1)-(k+2)],'Linearity: multiply the mean by 3, then subtract the constant.'
        if family==1:
            return f'X has mean {k} and variance {k+3}. What is E[X^2]?',k*k+k+3,[k*k,k+3,(k+k+3)**2],'The second moment equals the variance plus the square of the mean.'
        if family==2:
            return f'A mixture selects group A with probability 1/4 and group B otherwise. Their conditional means are {4*k} and {4*k+8}. What is the overall mean?',4*k+6,[4*k+4,8*k+8,4*k+2],'Weight the two conditional means by 1/4 and 3/4.'
        if family==3:
            return f'E[X]={k}, E[Y]={k+1}, and Cov(X,Y)=2. What is E[XY]?',k*(k+1)+2,[k*(k+1),2*k+3,k*(k+1)-2],'Covariance equals E[XY]-E[X]E[Y]; rearrange.'
        return f'Independent X and Y have means {k} and 2, and variances 3 and 5. What is E[(X+Y)^2]?',8+(k+2)**2,[8,8+k*k+4,(8+k+2)**2],'The sum has mean k+2 and variance 8; second moment is variance plus mean squared.'
    if topic=='uniform':
        if family==0:
            return f'X is continuously uniform on [{k},{k+10}]. What is E[X]?',k+5,[k,k+10,2*k+10],'The uniform mean is the midpoint of the endpoints.'
        if family==1:
            w=6*k
            return f'X is continuously uniform on [2,{2+w}]. What is Var(X)?',F(w*w,12),[F(w*w,2),F(w*w,6),w*w],'Uniform variance is interval length squared divided by 12.'
        if family==2:
            return f'X is continuously uniform on [0,{4*k}]. What is P(X<{k})?',F(1,4),[F(3,4),F(1,2),F(1,4*k)],'Uniform probability is favorable length divided by total length.'
        if family==3:
            w=3*k+1
            return f'X is continuously uniform on [0,{w}]. What is E[X^2]?',F(w*w,3),[F(w*w,12),F(w*w,2),w*w],'Integrating x squared over the uniform interval gives w squared divided by 3.'
        return f'X is continuously uniform on [0,{6*k}]. Given X>{2*k}, what is E[X] under that condition?',4*k,[3*k,2*k,5*k],'Conditioning truncates the uniform interval to (2k,6k); its midpoint is 4k.'
    if topic=='type_i':
        if family==0:
            return f'{20*k} tests each have a true null and exact Type I error probability 0.05. What is the expected number of false rejections?',k,[20*k,2*k,F(k,2)],'Linearity of expectation gives the number of tests times each rejection probability; independence is not needed.'
        if family==1:
            return f'{k} true null hypotheses are each tested at level 0.01. Without independence, what upper bound on the probability of at least one false rejection follows directly from the union bound?',F(k,100),[F(1,100),F(100-k,100),F(k,10)],'The union bound sums the per-test Type I error bounds.'
        if family==2:
            return f'{k} independent tests have true nulls and exact Type I error probability 1/10 each. What is the probability of no false rejections?',F(9,10)**k,[1-F(9,10)**k,F(1,10)**k,F(9,10)],'Each test avoids rejection with probability 9/10; independence gives the product.'
        if family==3:
            return f'{k} independent tests each have a true null and exact Type I error probability 1/5. What is the probability of exactly one false rejection?',k*F(1,5)*F(4,5)**(k-1),[F(1,5)**k,F(1,5),1-F(4,5)**k],'Choose which single test rejects, multiply its rejection probability by non-rejection probabilities for the others.'
        p=F(v+1,10)
        return f'Three independent tests each have a true null and exact Type I error probability {p}. What is the probability that at least two reject?',3*p*p*(1-p)+p**3,[p**3,1-(1-p)**3,3*p*p],'Add the probabilities of exactly two and exactly three rejections.'
    if topic=='type_ii':
        if family==0:
            p=F(60+2*v,100)
            return f'At a fixed alternative, a test has power {p}. What is its Type II error probability?',1-p,[p,F(1,20),1+p],'Type II error probability is one minus power at the specified alternative.'
        if family==1:
            return f'At a specified false null, {10*k} studies each have Type II error probability 1/5. What is the expected number of failures to reject?',2*k,[8*k,k,10*k],'The expected number of misses is the number of studies times beta.'
        if family==2:
            beta=F(v+1,25)
            return f'Two independent studies each have Type II error probability {beta} at the same specified alternative. What is the probability that at least one rejects?',1-beta**2,[(1-beta)**2,beta**2,1-beta],'Both studies fail to reject with probability beta squared; take the complement.'
        if family==3:
            p=F(v+2,10); q=F(4,5)
            return f'Two independent studies have powers {p} and {q} at their specified alternatives. What is the probability that exactly one rejects?',p*(1-q)+(1-p)*q,[p*q,1-(1-p)*(1-q),p*p],'Exactly one rejection comprises two disjoint outcomes: first only or second only.'
        p=F(v+2,10)
        return f'Three independent studies each have power {p} at a specified alternative. What is the probability that at least two reject?',3*p*p*(1-p)+p**3,[p**3,1-(1-p)**3,3*p*p],'Use the binomial probabilities for exactly two and exactly three rejections.'
    if family==0:
        return f'A two-sided normal-theory confidence interval has critical value 2 and standard error {k}. What is its total width?',4*k,[2*k,k,8*k],'Total width is twice critical value times standard error.'
    if family==1:
        return f'Keep confidence level and population standard deviation fixed. By what factor must sample size increase so a normal-theory mean interval has 1/{k} of its original width?',k*k,[k,F(1,k),k**3],'Width scales with the inverse square root of sample size; square the desired reduction factor.'
    if family==2:
        return f'A normal-theory mean interval has total width {2*k}. With confidence and sample size fixed, the known population standard deviation triples. What is its new width?',6*k,[2*k,F(2*k,3),18*k],'Width is proportional to population standard deviation when confidence and sample size are fixed.'
    if family==3:
        return f'A two-sided normal-theory confidence interval needs total width {4*k} with critical value 2. What standard error gives this width?',k,[2*k,4*k,F(k,2)],'Standard error equals total width divided by twice the critical value.'
    return f'A normal-theory mean interval is [{10*k},{10*k+8}]. Its center, confidence level and population standard deviation stay fixed while sample size quadruples. What is the new lower endpoint?',10*k+2,[10*k,10*k+3,10*k+4],'Quadrupling sample size halves the width from 8 to 4. Keep the center and subtract the new half-width 2.'


def build():
    result={split:[] for split in ('train','validation','test')}
    for split,families,n in (('train',range(3),10),('validation',[3],4),('test',[4],6)):
        rows=result[split]
        for topic in TOPICS:
            for family in families:
                for v in range(n):
                    q,correct,wrong,reason=item(topic,family,v)
                    correct=str(correct); choices=list(map(str,wrong))
                    pos=(len(rows)*3+1)%4; choices.insert(pos,correct)
                    rows.append(dict(id=f'{split}_{topic}_{family}_{v:02d}',category=topic,
                        family=f'{topic}_{family}',question=q,choices=choices,
                        answer_letter='ABCD'[pos],reference_reason=reason,split=split))
    return result


def audit(excluded=()):
    data=build(); allrows=sum(data.values(),[])
    keys=[normalize_question(r['question']) for r in allrows]
    assert len(keys)==len(set(keys))==240
    assert not set(keys)&{normalize_question(r['question']) for r in excluded}
    for r in allrows:
        assert len(set(r['choices']))==4, r['id']
        # All options are numeric, so also catch equivalent fraction spellings.
        assert len({F(c) for c in r['choices']})==4, r['id']
    families={k:{r['family'] for r in v} for k,v in data.items()}
    assert not families['train']&families['validation']
    assert not families['train']&families['test']
    assert not families['validation']&families['test']
    for split,n in (('train',180),('validation',24),('test',36)):
        rows=data[split]; assert len(rows)==n
        assert Counter(r['answer_letter'] for r in rows)==dict.fromkeys('ABCD',n//4)
    return {k:dict(n=len(v),families=len(families[k]),sha256=digest(v)) for k,v in data.items()}


if __name__=='__main__':
    import json
    print(json.dumps(dict(audit=audit(),data=build()),indent=2))
