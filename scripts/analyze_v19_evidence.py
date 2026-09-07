"""Post-hoc paired analysis; data-level uncertainty is not training-seed variance."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
D=Path(__file__).resolve().parents[1]/'docs'
x=json.loads((D/'STATS_V0_19_RESULTS.json').read_text());out=x['outputs']
credits={(a['file'],a['id']) for a in x['semantic_review']['decisions'] if a['credit']}
def correct(f,r):return bool(r['correct'] or (f,r['id']) in credits)
def compare(a,b,category=None):
 left={r['id']:r for r in out[a]};right=[r for r in out[b] if category is None or r['category']==category]
 changes=[int(correct(b,r))-int(correct(a,left[r['id']])) for r in right]
 gains=changes.count(1);losses=changes.count(-1)
 return dict(n=len(right),baseline=sum(correct(a,left[r['id']]) for r in right),candidate=sum(correct(b,r) for r in right),gains=gains,losses=losses,net=gains-losses,p_exact_two_sided=float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.,changes=changes)
a='v15_old_test.json';b='epoch_1_old_test.json'
families={k:compare(a,b,k) for k in sorted({r['category'] for r in out[b]})}
ps=np.array([r['p_exact_two_sided'] for r in families.values()]);order=np.argsort(ps);padj=np.empty(len(ps));padj[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
for r,p in zip(families.values(),padj):r['p_holm_8']=float(p)
rng=np.random.default_rng(190019)
bootstrap=sum(rng.choice(r['changes'],size=(20000,r['n']),replace=True).sum(axis=1) for r in families.values())
old=compare(a,b);old['stratified_paired_bootstrap_95_net_questions']=np.quantile(bootstrap,[.025,.975]).tolist()
new=compare('v15_new_test.json','epoch_1_new_test.json')
qs={q['id']:q for q in x['frozen_questions.json']['new_test']}
l={r['id']:r for r in out['v15_new_test.json']};groups={}
for r in out['epoch_1_new_test.json']:
 key=qs[r['id']]['story_id'];groups.setdefault(key,[]).append(int(correct('epoch_1_new_test.json',r))-int(correct('v15_new_test.json',l[r['id']])))
assert len(groups)==32 and all(len(v)==3 for v in groups.values())
v=np.array([sum(g) for g in groups.values()]);bs=rng.choice(v,size=(20000,len(v)),replace=True).sum(axis=1)
new['story_cluster_count']=32;new['story_cluster_bootstrap_95_net_questions']=np.quantile(bs,[.025,.975]).tolist()
result=dict(post_hoc=True,old=old,by_family=families,new=new,limits=['Exact tests treat sampled pairs as independent within each tested set; template dependence limits generalization.','Bootstrap intervals condition on these authored question families and one training run; not training-seed uncertainty.','Holm corrects the eight old-family tests; wider historical adaptive exploration is not corrected.','Failure to reject equality does not establish equivalence.','New-chain question-level exact p is descriptive; use story-cluster interval for dependence.'])
(D/'STATS_V19_PAIRED_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{j:v for j,v in r.items() if j!='changes'} for k,r in families.items()},indent=2));print('oldCI',old['stratified_paired_bootstrap_95_net_questions'],'newCI',new['story_cluster_bootstrap_95_net_questions'])
