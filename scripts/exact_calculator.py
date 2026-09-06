"""Bounded rational-expression executor; never evaluates arbitrary Python.

Accepts a model-authored expression, not a question or reference answer.
"""
import ast
import math
from fractions import Fraction


def calculate(expression):
    if not isinstance(expression, str) or len(expression) > 1024:
        raise ValueError('Expression must be a string of at most 1024 characters')
    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('Invalid expression') from exc
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError('Expression is too complex')

    def bounded(value):
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
            raise ValueError('Result exceeds arithmetic limit')
        return value

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return bounded(Fraction(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            elif isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right) > 32:
                    raise ValueError('Exponent must be an integer between -32 and 32')
                result = left ** int(right)
            else:
                raise ValueError('Unsupported operator')
            return bounded(result)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ('comb', 'gcd') and len(node.args) == 2
                and not node.keywords):
            values = [visit(arg) for arg in node.args]
            if any(v.denominator != 1 for v in values):
                raise ValueError('Function arguments must be integers')
            a, b = map(int, values)
            if node.func.id == 'comb':
                if not 0 <= b <= a <= 1000:
                    raise ValueError('comb requires 0 <= k <= n <= 1000')
                return bounded(Fraction(math.comb(a, b)))
            return bounded(Fraction(math.gcd(a, b)))
        raise ValueError('Unsupported expression')

    try:
        return str(visit(tree.body))
    except ZeroDivisionError as exc:
        raise ValueError('Division by zero') from exc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expression')
    print(calculate(parser.parse_args().expression))
