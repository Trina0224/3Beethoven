"""Match observed optimizer microbatches to the saved ordered supervision."""
import argparse,hashlib,json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 from transformers import AutoTokenizer
 out={}
 for seed in (2027,31415):
  folder=a.root/f'low_lr/original_v15_continued_lora_seed_{seed}'
  if not (folder/'training_complete.json').exists():continue
  complete=read(folder/'training_complete.json')
  if complete['selection']['stop_reason']=='manual_review_required':continue
  step=complete['selection']['selected_step'] or complete['selection']['diagnostic_step']
  tok=AutoTokenizer.from_pretrained(folder/f'step_{step}/adapter',local_files_only=True)
  order=read(folder/'training_order.json');trace=[json.loads(l) for l in (a.root/f'low_lr/microbatches_{seed}.jsonl').read_text().splitlines()]
  assert len(trace)==min(step*8,len(order)),(seed,len(trace),step)
  for i,(row,actual) in enumerate(zip(order,trace)):
   msgs=[{'role':'user','content':row['prompt']}]
   pre=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=False)
   full=tok.apply_chat_template(msgs+[{'role':'assistant','content':row['target']}],tokenize=True,add_generation_prompt=False,return_dict=False)
   labels=[-100]*len(pre)+full[len(pre):]
   expected={'global_step_before':i//8,'input_ids_sha256':digest([full]),'labels_sha256':digest([labels]),'supervised_tokens':len(full)-len(pre)}
   assert expected==actual,(seed,i,expected,actual)
  out[seed]={'matched_all_actual_microbatches':True,'rows':len(trace),'updates':step,'supervised_tokens':sum(x['supervised_tokens'] for x in trace),'planned_rows_sha256':digest(order),'trace_sha256':hashlib.sha256((a.root/f'low_lr/microbatches_{seed}.jsonl').read_bytes()).hexdigest()}
 (a.root/'ACTUAL_TRACE_VERIFIED.json').write_text(json.dumps(out,indent=2)+'\n')
 print('ACTUAL_TRACE_VERIFIED',json.dumps(out),flush=True)
if __name__=='__main__':main()
