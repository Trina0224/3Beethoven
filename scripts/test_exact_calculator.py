import unittest
from exact_calculator import calculate


class CalculatorTests(unittest.TestCase):
    def test_exact_arithmetic(self):
        for expression, result in [
            ('1/10+2/10', '3/10'), ('144/216', '2/3'),
            ('gcd(144,216)', '72'), ('17**4', '83521'),
            ('comb(4,2)*(7/50)**2*(43/50)**2', '271803/3125000'),
            ('(1-7/50)*(11/50)+(7/50)*(1-11/50)', '373/1250'),
            ('2**(-3)', '1/8'), ('+3-5', '-2'),
        ]:
            with self.subTest(expression=expression):
                self.assertEqual(calculate(expression), result)

    def test_reject_unsafe_or_unbounded(self):
        for expression in [
            '__import__("os")', '(1).__class__', '[1,2]', 'True',
            '0.1+0.2', '2^3', '2**33', '2**(2**32)',
            'comb(1001,2)', 'comb(2,3)', 'gcd(1/2,3)',
            'comb(n=3,k=2)', '1/0', 'unknown', '1+' * 200 + '1',
        ]:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    calculate(expression)


if __name__ == '__main__':
    unittest.main()
