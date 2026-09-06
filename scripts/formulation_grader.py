"""Safe, representation-tolerant grading for the bounded statistics curriculum.

Never uses eval, never fills missing bindings from the reference. Auto-credit
requires a reviewed expression structure AND exact numerical agreement. Other
equivalent expressions are pending review, not automatically mathematically wrong.
"""
import ast
import copy
import math
import re
from fractions import Fraction as F
from exact_calculator import calculate


def fraction_node(value):
    return ast.parse(str(value),mode='eval').body


def parse_expression(text,env):
    if len(text)>2048:raise ValueError('Expression too long')
    tree=ast.parse(text.strip().replace('^','**'),mode='eval').body
    if len(list(ast.walk(tree)))>128:raise ValueError('Expression too large')
    class Resolve(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id not in env:raise ValueError('Unbound name: '+node.id)
            return copy.deepcopy(env[node.id])
        def visit_Constant(self,node):
            if type(node.value) not in (int,float):raise ValueError('Non-numeric literal')
            return fraction_node(F(str(node.value)))
        def visit_Call(self,node):
            if not isinstance(node.func,ast.Name) or node.keywords:raise ValueError('Unsupported call')
            name=node.func.id
            if name not in ('comb','sqrt'):raise ValueError('Unsupported function')
            args=[self.visit(a) for a in node.args]
            if name=='sqrt':
                if len(args)!=1:raise ValueError('sqrt arity')
                v=F(calculate(ast.unparse(args[0])))
                if v<0:raise ValueError('Negative square root')
                n,d=math.isqrt(v.numerator),math.isqrt(v.denominator)
                if n*n!=v.numerator or d*d!=v.denominator:raise ValueError('Non-rational square root requires review')
                return fraction_node(F(n,d))
            if len(args)!=2:raise ValueError('comb arity')
            return ast.Call(func=ast.Name(id='comb',ctx=ast.Load()),args=args,keywords=[])
        def visit_BinOp(self,node):
            left,right=self.visit(node.left),self.visit(node.right)
            if isinstance(node.op,ast.Pow) and F(calculate(ast.unparse(right)))==F(1,2):
                return self.visit_Call(ast.Call(func=ast.Name(id='sqrt',ctx=ast.Load()),args=[left],keywords=[]))
            return ast.BinOp(left=left,op=node.op,right=right)
        def generic_visit(self,node):
            if not isinstance(node,(ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.USub,ast.UAdd,ast.Load)):
                raise ValueError('Unsupported syntax')
            return super().generic_visit(node)
    resolved=ast.fix_missing_locations(Resolve().visit(tree))
    expr=ast.unparse(resolved)
    calculate(expr)  # bounds the expanded tree and arithmetic
    return resolved


def shape(node):
    # Fraction/decimal spelling is presentation; don't collapse arbitrary
    # arithmetic subtrees (which would reduce every expression to its answer).
    if isinstance(node,ast.Constant):return ('q',str(F(node.value)))
    if isinstance(node,ast.UnaryOp):
        return ('q',calculate(ast.unparse(node))) if isinstance(node.operand,ast.Constant) else (type(node.op).__name__,shape(node.operand))
    if isinstance(node,ast.Call):return (node.func.id,*(shape(a) for a in node.args))
    if isinstance(node,ast.BinOp):
        if isinstance(node.op,ast.Div) and isinstance(node.left,ast.Constant) and isinstance(node.right,ast.Constant):
            return ('q',calculate(ast.unparse(node)))
        op=type(node.op).__name__
        if op in ('Add','Mult'):
            def flatten(n):
                if isinstance(n,ast.BinOp) and type(n.op).__name__==op:return flatten(n.left)+flatten(n.right)
                return [shape(n)]
            return (op,*sorted(flatten(node),key=repr))
        return (op,shape(node.left),shape(node.right))
    raise ValueError('Unsupported shape')


def references(q):
    b={k:'('+v+')' for k,v in q['bindings'].items()};c=q['category']
    refs=[q['expression']]
    if c=='uniform_time':
        u,t=b['upper_minutes'],b['cutoff_minutes']
        refs += [f'(1/({u}-{t}))*(({u}**2-{t}**2)/2)',
                 f'({u}**2-{t}**2)/(2*({u}-{t}))']
    elif c=='moment':
        m,v,a,z=(b[k] for k in ('mean','variance','scale','offset'))
        refs += [f'{a}**2*({v}+{m}**2)+2*{a}*{m}*{z}+{z}**2',
                 f'{a}**2*{v}+{a}**2*{m}**2+2*{a}*{m}*{z}+{z}**2']
    elif c=='interval':
        l,u,k=(b[x] for x in ('lower','upper','width_divisor'))
        center=f'({l}+{u})/2';half=f'({u}-{l})/2'
        refs += [f'{center}+{half}/{k}',f'{center}+({u}-{center})/{k}',
                 f'{l}+({u}-{l})/2+({u}-{l})/(2*{k})']
    return {shape(parse_expression(x,{})) for x in refs}


def grade(raw,q):
    result=dict(math_correct=None,format_exact=False,bindings_match_schema=None,
                executable=False,review_required=True,normalized_expression=None)
    result['format_exact']=bool(re.fullmatch(r'\s*Bindings: [^\n]+\nExpression: [^\n]+\s*',raw))
    env={};candidates=[];errors=[];seen={}
    for line in raw.strip().splitlines():
        line=re.sub(r'^\s*(Bindings|Expression)\s*:\s*','',line,flags=re.I).strip()
        if line.startswith('```'):continue
        for segment in line.split(';'):
            segment=segment.strip()
            if not segment:continue
            parts=segment.split('=')
            lhs=parts[0].strip() if len(parts)>1 else None
            expression=parts[-1].strip()
            try:
                node=parse_expression(expression,env)
                if lhs and re.fullmatch(r'[A-Za-z_]\w*',lhs):
                    env[lhs]=node;seen[lhs]=calculate(ast.unparse(node))
                    # For interval questions, upper may be a computed OUTPUT,
                    # not an erroneous binding of the old endpoint.
                    if q['category']=='interval' and lhs in ('upper','new_upper','new_upper_endpoint'):
                        candidates.append((segment,node))
                    elif lhs not in q['bindings']:
                        candidates.append((segment,node))
                else:candidates.append((segment,node))
            except (ValueError,SyntaxError,RecursionError,OverflowError) as exc:
                errors.append(dict(segment=segment,error=str(exc)))
                # An unparseable final line must not silently select an earlier
                # correct result. Unknown output needs review.
                candidates.append((segment,None))
    result['bindings_match_schema']=(all(seen.get(k)==calculate(v) for k,v in q['bindings'].items()))
    result['parse_notes']=errors
    if not candidates:return result
    text,node=candidates[-1];result['selected_source']=text
    if node is None:return result
    expression=ast.unparse(node);value=calculate(expression)
    result.update(executable=True,normalized_expression=expression,computed=value)
    if value!=q['answer']:
        return dict(result,math_correct=False,review_required=False,reason='The model expression yields the wrong value; no repair applied.')
    if shape(node) in references(q):
        return dict(result,math_correct=True,review_required=False,reason='Correct fully substituted expression with reviewed mathematical structure; labels are scored separately.')
    return dict(result,reason='Numerically equal but structure not yet verified; needs semantic review, not an automatic failure.')
