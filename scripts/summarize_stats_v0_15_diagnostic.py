"""Summarize raw diagnostic grades without changing pending or failed outputs."""
import json
from collections import defaultdict
from pathlib import Path

def summarize(data):
    result={}
    for model,rows in data['models'].items():
        groups=defaultdict(list)
        for row in rows:groups[(row['family'],row['stage'])].append(row)
        result[model]={family:{stage:dict(n=len(rs),correct=sum(r['math_correct'] is True for r in rs),
            wrong=sum(r['math_correct'] is False for r in rs),pending=sum(r['math_correct'] is None for r in rs),
            token_limit=sum(r['generated_tokens']==160 for r in rs))
            for (f,stage),rs in groups.items() if f==family} for family in sorted({f for f,_ in groups})}
    return result

if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'docs/STATS_V0_15_DIAGNOSTIC_RESULTS.json'
    print(json.dumps(summarize(json.loads(path.read_text())),indent=2))
