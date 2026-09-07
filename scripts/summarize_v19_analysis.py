"""Summarize completed fixed-weight probes with explicit clustered units."""
import json,hashlib,base64,gzip
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
D=Path(__file__).resolve().parents[1]/'docs';x=json.loads((D/'STATS_V19_ANALYSIS_RESULTS.json').read_text());qmap={q['id']:q for q in x['questions.json']}
pending=x['verification.json']['pending'];assert len(pending)==1
r=pending[0];assert r['raw']=='Expression: (3 * X + 10 - 3 * X) ** 2' and r['reference']=='3**2*216'
review=dict(decisions=[dict(model=r['model'],id=r['id'],raw=r['raw'],raw_sha256=hashlib.sha256(r['raw'].encode()).hexdigest(),reference=r['reference'],credit=False,reason='X 相消後得到 10**2=100；參考式 3**2*216=1944。即使容許符號相消，答案仍錯。')],pending_remaining=0,note='All frozen scores unchanged; the sole pending expression is mathematically wrong.')
models={};rng=np.random.default_rng(190038)
for name in ('v15','v19'):
 rows=x[f'{name}_formula.json'];byid={r['id']:r for r in rows};groups={}
 for r in rows:
  q=qmap[r['id']]
  if q['probe']=='concept':groups.setdefault(q['pair_id'],{})[q['variant']]=r
 comparisons={}
 for variant in ('paraphrase','magnitude'):
  gains=sum(g[variant]['correct'] and not g['original']['correct'] for g in groups.values());losses=sum(not g[variant]['correct'] and g['original']['correct'] for g in groups.values());both=sum(g[variant]['correct'] and g['original']['correct'] for g in groups.values())
  clusters={}
  for key,g in groups.items():clusters.setdefault(qmap[g['original']['id']]['story_id'],[]).append(int(g[variant]['correct'])-int(g['original']['correct']))
  v=np.array([sum(a) for a in clusters.values()]);bs=rng.choice(v,size=(20000,len(v)),replace=True).sum(axis=1)
  comparisons[variant]=dict(gains=gains,losses=losses,both_correct=both,paired_questions=24,story_clusters=len(clusters),cluster_bootstrap_net_95=np.quantile(bs,[.025,.975]).tolist(),note='Eight stories only; descriptive conditional uncertainty, not training-seed variance.')
 bycell={}
 for r in rows:
  q=qmap[r['id']]
  if q['probe']=='concept':
   key=q['track']+'_d'+str(q['depth']);bycell.setdefault(key,{v:0 for v in ('original','paraphrase','magnitude')});bycell[key][q['variant']]+=int(r['correct'])
 variants={k:dict(v,automatic_pending=v['pending'],pending=0) for k,v in x['summary.json'][name]['by_variant'].items()}
 models[name]=dict(variants=variants,paired=comparisons,all_three_correct=sum(all(r['correct'] for r in g.values()) for g in groups.values()),by_cell=bycell,mc_correct=x['summary.json'][name]['mc']['overall']['correct'],mc_all_four=x['summary.json'][name]['mc']['all_four_correct'])
left={(r['id'],r['shift']):r for r in x['v15_mc.json']};right=x['v19_mc.json'];clusters={}
for r in right:clusters.setdefault(r['id'],[]).append(int(r['correct'])-int(left[(r['id'],r['shift'])]['correct']))
v=np.array([sum(a) for a in clusters.values()]);bs=rng.choice(v,size=(20000,60),replace=True).sum(axis=1)
l4={q['id']:q['correct_rotations']==4 for q in x['summary.json']['v15']['mc']['question_details']};r4={q['id']:q['correct_rotations']==4 for q in x['summary.json']['v19']['mc']['question_details']};g=sum(r4[k] and not l4[k] for k in l4);l=sum(l4[k] and not r4[k] for k in l4)
mc=dict(question_clusters=60,responses_per_model=240,net_rotation_correct=int(v.sum()),cluster_bootstrap_net_95=np.quantile(bs,[.025,.975]).tolist(),all_four_gains=g,all_four_losses=l,all_four_p_exact=float(binomtest(g,g+l,.5).pvalue) if g+l else 1.)
# Same prompts from the original v19 run: score reproducibility, not seed replication.
old=json.loads((D/'STATS_V0_19_RESULTS.json').read_text());reproduction={}
for name,file in [('v15','v15_new_test.json'),('v19','epoch_1_new_test.json')]:
 prior={r['id']:r for r in old['outputs'][file]};same=0;rawsame=0
 for r in x[name+'_formula.json']:
  q=qmap[r['id']]
  if q['variant']=='original':
   p=prior[q['pair_id']];assert p['prompt']==r['prompt'];same+=p['correct']==r['correct'];rawsame+=p['raw']==r['raw']
 reproduction[name]=dict(n=24,same_correctness=same,same_raw=rawsame)
failures=[dict(model=name,id=r['id'],question=qmap[r['id']]['question'],track=qmap[r['id']].get('track'),depth=qmap[r['id']].get('depth'),raw=r['raw'],reference=qmap[r['id']]['expression']) for name in ('v15','v19') for r in x[name+'_formula.json'] if qmap[r['id']]['variant']=='paraphrase' and not r['correct']]
summary=dict(models=models,mc_paired=mc,original_reproduction=reproduction,review=review,paraphrase_failures=failures,kaggle_version=38,script_version_id=347843399,receipt=x['receipt'],verified_responses=656)
(D/'STATS_V19_ANALYSIS_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');x['semantic_review']=review;(D/'STATS_V19_ANALYSIS_RESULTS.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');(D/'STATS_V19_ANALYSIS_RESULTS.json.gz.b64').write_text(base64.b64encode(gzip.compress(json.dumps(x,ensure_ascii=False).encode())).decode()+'\n')
print(json.dumps(dict(models=models,mc=mc,reproduction=reproduction),indent=2))
print('V19 PARAPHRASE ERRORS')
for r in failures:
 if r['model']=='v19':print(r['track'],r['depth'],r['raw'],'REF',r['reference'])
