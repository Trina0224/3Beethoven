"""Post-run curriculum correction only; never modifies trained data or scores."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from exact_calculator import calculate
from stats_consolidation_pilot import digest


def one(node):
    return isinstance(node,ast.Constant) and type(node.value) is int and node.value==1


def simplify(expression):
    original_value=calculate(expression)
    count=0
    class Reduce(ast.NodeTransformer):
        def visit_BinOp(self,node):
            nonlocal count
            node=self.generic_visit(node)
            if (isinstance(node.op,ast.Sub) and one(node.left) and
                isinstance(node.right,ast.BinOp) and isinstance(node.right.op,ast.Sub) and one(node.right.left)):
                count+=1
                return node.right.right
            return node
    tree=Reduce().visit(ast.parse(expression,mode='eval'))
    result=ast.unparse(ast.fix_missing_locations(tree)) if count else expression
    assert calculate(result)==original_value
    return result,count


def main(source,output):
    data=json.loads(source.read_text());rows=copy.deepcopy(data['arms']['semantic_course'])
    changes=[]
    for row in rows:
        old=row['target'];expr,count=simplify(old.removeprefix('Expression: '))
        if not count:continue
        row['target']='Expression: '+expr
        row['target_sha256']=hashlib.sha256(row['target'].encode()).hexdigest()
        row['provenance']='post_run_assistant_correction_by_identity_1_minus_1_minus_x_equals_x'
        changes.append(dict(id=row['source_id'],before=old,after=row['target'],removed_pairs=count))
    assert all(simplify(r['target'].removeprefix('Expression: '))[1]==0 for r in rows)
    result=dict(status='post_run_corrected_training_data_not_yet_trained',train=rows,
        source_training_sha256=data['manifest']['arm_sha256']['semantic_course'],
        corrected_training_sha256=digest(rows),changed_rows=len(changes),changes=changes,
        rationale='V57 candidate produced two unfinished complement loops. This diagnoses a representation hazard; '
                  'it does not isolate causality. Remove unnecessary nested complements without changing meanings, '
                  'question numbers, row order or evaluation data. Both students also lost retention; that remains unresolved.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('train','changes')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source,a.output)
