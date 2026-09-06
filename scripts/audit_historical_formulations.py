"""Reproduce the retrospective, assistant-reviewed formulation ledger.

Verdicts are explicit review annotations, NOT a general automatic math grader.
No inference, paid API calls, changes to original scores, or model updates.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from exact_calculator import calculate

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SOURCE_HASHES = {
    'STATS_V0_9_RESULTS.json': 'bae8c278fafdc824878e97fa30be16fd43c3804244903a9004bc5de3c7d5a501',
    'STATS_V0_10_RESULTS.json': 'd2cf7f521897d4c4f08f18d2e40a678a1d6f01fd769ef2b2bb06fcd171a494c3',
    'STATS_V0_11_RESULTS.json': 'ba372f17880c2273a20008c761e0dc625856126baca54a0e902845a54bb8b094',
}

# Row indices refer to immutable ordered source response arrays. True = an
# applicable governing formula exists; grounded = its problem-specific setup
# is correct before arithmetic. Do not infer either from final-answer accuracy.
SPECS = {
    ('STATS_V0_9', 'baseline_numeric'): {
        'formula': list(range(8)) + [11,12,13,19] + list(range(24,32)) + [32,34,37,38,39],
        'grounded': list(range(8)) + [11,12,13,19] + list(range(24,32)) + [34,37,38,39],
    },
    ('STATS_V0_9', 'v05_numeric'): {
        'formula': list(range(8)) + [11,12,13] + list(range(24,32)) + [35,36],
        'grounded': list(range(8)) + list(range(24,32)) + [35,36],
    },
    ('STATS_V0_9', 'v09_numeric'): {
        'formula': list(range(48)),
        'grounded': [i for i in range(48) if i not in (40,41,47)],
        'exact_first_expression': True,
    },
    ('STATS_V0_10', 'baseline_numeric'): {
        'formula': [i for i in range(48) if i != 27],
        'grounded': [i for i in range(48) if i not in (6,27,30,34,35,39,40,43)],
    },
    ('STATS_V0_10', 'v09_numeric'): {
        'formula': list(range(48)),
        'grounded': [i for i in range(48) if i not in (5,19,26,35,37,46)],
        'exact_first_expression': True,
    },
    ('STATS_V0_10', 'v10_numeric'): {
        'formula': list(range(48)),
        'grounded': [i for i in range(48) if i not in (32,35,38)],
        'exact_first_expression': True,
    },
    ('STATS_V0_11', 'v10_transfer'): {
        'formula': list(range(48)),
        'grounded': [i for i in range(48) if i not in (25,26,30,31,32,35,39,44)],
        'exact_first_expression': True,
    },
    ('STATS_V0_11', 'v11_transfer'): {
        'formula': list(range(48)),
        'grounded': [i for i in range(48) if i not in (25,30,31,32,35,36,37,40,45)],
        'exact_first_expression': True,
    },
}

NOTES = {
    ('STATS_V0_9','baseline_numeric',18): 'Numerically correct by cancellation of conceptual mistakes: wrong conditional integrand and normalization. No formulation credit.',
    ('STATS_V0_9','baseline_numeric',32): 'Correct two-case formula, but detection/miss probabilities are substituted incorrectly.',
    ('STATS_V0_9','v09_numeric',40): 'Correct C+h/k rule, but center 50 is wrong for [36,54]; it should be 45.',
    ('STATS_V0_9','v09_numeric',41): 'Correct C+h/k rule, but center 65 is wrong for [51,69]; it should be 60.',
    ('STATS_V0_9','v09_numeric',47): '38-(38-32)/3 happens to equal 36 for this k=3 case, but is not C+h/k. Coincidental numeric agreement does not validate the setup.',
    ('STATS_V0_10','baseline_numeric',6): 'Binomial structure present, but p=11/40 replaces the question\'s 9/40.',
    ('STATS_V0_10','v09_numeric',5): 'C(6,4) is represented as 6 rather than 15. Generic binomial formula credit only.',
    ('STATS_V0_10','v09_numeric',19): 'C(5,3) is represented as 5 rather than 10. Generic binomial formula credit only.',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit():
    inventory, records, summaries = [], [], []
    sources = {}
    for path in sorted(DOCS.glob('*RESULTS.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        for key, rows in data.items():
            if key == 'teacher' or not isinstance(rows, list) or not rows:
                continue
            if not isinstance(rows[0], dict) or 'raw' not in rows[0]:
                continue
            counts = Counter()
            for row in rows:
                mode = row.get('mode')
                if re.fullmatch(r'\s*[ABCD][.\s]*', row['raw']):
                    status = 'letter_only_no_formulation_evidence'
                elif (path.stem.removesuffix('_RESULTS'), key) in SPECS:
                    status = 'reviewed_primary_numeric_response'
                elif mode in ('guided','arithmetic','compact_arithmetic') or key.endswith(('_arithmetic','_variants')):
                    status = 'formula_or_operation_supplied_by_prompt_not_independent_formulation'
                elif mode == 'free':
                    status = 'direct_answer_without_executable_setup'
                else:
                    status = 'diagnostic_reasoning_reviewed_separately_not_ranked'
                counts[status] += 1
            inventory.append({'source': path.name, 'set': key, 'n': len(rows), 'classification': dict(counts)})

    for (experiment, key), spec in SPECS.items():
        path = DOCS / (experiment + '_RESULTS.json')
        sources[path.name] = sha(path.read_bytes())
        if sources[path.name] != SOURCE_HASHES[path.name]:
            raise ValueError(f'Source changed; row annotations require fresh review: {path.name}')
        data = json.loads(path.read_text())
        rows = data[key]
        assert len(rows) == 48
        formula, grounded = set(spec['formula']), set(spec['grounded'])
        assert grounded <= formula <= set(range(48))
        current = []
        for i, row in enumerate(rows):
            raw = row['raw']
            evidence = '\n'.join(line for line in raw.splitlines()
                                 if line.startswith(('Formula:', 'Calculation:')))
            item = {
                'source': path.name, 'set': key, 'index': i, 'id': row['id'],
                'category': row['category'], 'prompt': row['prompt'],
                'raw_sha256': sha(raw.encode()), 'raw': raw,
                'expected': row['expected'], 'original_correct': row['correct'],
                'hit_token_limit': row.get('hit_token_limit', False),
                'applicable_formula_correct': i in formula,
                'grounded_setup_correct': i in grounded,
                'evidence': evidence or raw,
            }
            reason = ('Correct governing formula and problem-specific setup; later arithmetic is not graded.'
                      if i in grounded else
                      'Correct governing formula, but wrong or incomplete problem-specific substitution.'
                      if i in formula else
                      'Wrong governing operation/event, or incomplete necessary formula; final number does not confer credit.')
            item['reason'] = NOTES.get((experiment,key,i),reason)
            if spec.get('exact_first_expression'):
                expression = raw.split('Calculation:',1)[1].split('=',1)[0].strip()
                normalized = expression.replace('^','**')
                value = calculate(normalized)
                item.update(extracted_expression=expression, normalized_expression=normalized,
                            recomputed_value=value, numeric_expression_matches=value == row['expected'])
                # Every pass is checked arithmetically, but a numeric match alone
                # does NOT override semantic review (v0.9 confidence item 47).
                assert i not in grounded or value == row['expected']
                if i not in grounded and value == row['expected']:
                    assert (experiment,key,i) == ('STATS_V0_9','v09_numeric',47)
            current.append(item)
        summary = {'experiment': experiment, 'model_set': key, 'n':len(current),
                   'original_strict_correct':sum(r['original_correct'] for r in current),
                   'applicable_formula_correct':len(formula),
                   'grounded_setup_correct':len(grounded),
                   'grounded_but_original_failed':sum(r['grounded_setup_correct'] and not r['original_correct'] for r in current),
                   'by_category':{}}
        for category in sorted({r['category'] for r in current}):
            subset=[r for r in current if r['category']==category]
            summary['by_category'][category] = {'n':len(subset),
                'formula_correct':sum(r['applicable_formula_correct'] for r in subset),
                'grounded_correct':sum(r['grounded_setup_correct'] for r in subset)}
        summaries.append(summary)
        records.extend(current)

    return {'method':'Retrospective assistant review, not blinded or independently human-validated. Explicit per-row annotations plus exact rational execution checks for 240 trained responses.',
            'scope':'384 primary free-response answers across matched v0.9, v0.10 and v0.11 evaluations. All other saved student response arrays inventoried; diagnostic modes are not pooled into the training ranking.',
            'rubric':{'formula':'Applicable governing formula, even when substitution or subsequent arithmetic is wrong. A tautological restatement of the question is insufficient.',
                      'grounded':'Correct problem-specific formulation before arithmetic, including a multi-step dependency chain when explicit. Wrong parameter bindings and coincidental numeric agreement do not pass.',
                      'original':'Original strict final-answer score unchanged. Differences include both arithmetic and formatting, not arithmetic alone.'},
            'source_sha256':sources,'inventory':inventory,'summaries':summaries,'records':records}


def report(data):
    lines = ['# 歷次學生回答：列式能力回溯審核', '',
        '結論：v0.9 是目前證據最清楚的「統計列式能力突破」；v0.10 是較合理的後續工具整合候選。v0.11 在其同題對照中沒有改善列式，不能因版本較新就優先採用。v0.12 完整回答尚未取回，不能排名。', '',
        '## 同題比較', '',
        '公式分＝適用的通用公式或完整題目算式正確；列式＋代入分＝題目數字、事件及必要步驟也正確，後續算錯仍得分。不同測試列不可直接當成排行榜。', '',
        '| 測試批次 | 模型 | 原始嚴格數值分 | 公式分 | 列式＋代入分 |',
        '|---|---|---:|---:|---:|']
    for s in data['summaries']:
        lines.append(f"| {s['experiment']} | {s['model_set']} | {s['original_strict_correct']}/48 | {s['applicable_formula_correct']}/48 | {s['grounded_setup_correct']}/48 |")
    lines += ['', '## 如何解讀「哪次最好」', '',
        '- v0.9 六類統計題同場比較：未訓練學生 24/48、v0.5 18/48、v0.9 45/48（列式＋代入）。v0.9 的改善是 21 題與 27 題；通用公式 48/48。這支持特定課程的能力轉移，不代表一般統計能力滿分。',
        '- v0.10 的較難兩類機率題：v0.9 42/48 → v0.10 45/48；兩者通用公式均 48/48。v0.10 有小幅、同題的代入改善，不能稱為大幅新增概念能力。',
        '- v0.11 新遷移題：v0.10 40/48 → v0.11 39/48；兩者通用公式均 48/48。差一題不足以斷言穩定退步，但沒有支持 v0.11 優於 v0.10 的列式證據。',
        '- 單純以「公式對就得分」看，v0.9、v0.10、v0.11 在這些各自的測試都已達 48/48，無法用這個飽和指標再分高下；應保留代入分和原始分。', '',
        '## 已確認的算錯、但列式正確案例', '',
        '1. v0.9 的 v09_test_expectation_006：列出 `3**2*1+(3*9+1)**2`，正確值 793，卻算成 865。列式得分。',
        '2. v0.10 的 v10_test_type_i_001：列出 `6*(13/40)**1*(27/40)**5`，卻把五次方算錯。列式得分，計算器可直接執行該算式。',
        '3. v0.11 的 v11_transfer_test_type_i_000：列出 `4*(9/40)**3*(31/40)**1`，卻在乘法及約分出錯。列式得分。', '',
        '反例：v0.9 的最後一題信賴區間寫 `38-(38-32)/3=36`，恰好得到正確數字，但這不是固定中心後縮小半寬的正確代入式；不因答案相同而給列式＋代入分。其通用公式另外得分。', '',
        '## 全歷史覆蓋與限制', '',
        '- 逐筆重新計分 384 筆主要數值回答；240 筆受訓學生回答的第一個數值算式另以精確有理數重新執行核對。所有主要對照包含同場未訓練或前版本學生，不只挑成功模型。',
        '- v0.3 holdout/rotation、v0.4、v0.5、v0.6 主要評測的學生輸出均只有選項字母；沒有可供重新評分的列式。不是列式 0 分，而是缺少證據。',
        '- v0.7、v0.8 是同一 v0.5 與基礎模型的診斷，不是新訓練版本。已檢視其自由回答、短步驟與長步驟：v0.5 的短格式可列對條件均勻分布與部分二項式，長格式常能列對二階矩和恰好一個檢出事件；仍可見漏組合係數、把 E[X²] 誤作 E[X] 或 Var(X)、信賴區間縮放錯誤。不能把這些不同提示／token 上限混入上表排名。',
        '- 給定公式／算式的診斷題及 v0.11 算術題不作獨立統計列式證據：把題目提供的算式照抄，不能用來證明蒸餾學會選公式。完整清冊見 JSON inventory。',
        '- 本次為知道模型標籤的回溯審核，非盲測、非獨立人工複核；原始題目已曝光。應由另一位審查者複核逐題紀錄，並以新凍結題目驗證後再作確認性結論。',
        '- 未覆蓋未保存於 GitHub 的早期開發回答或 v0.12 最終輸出。沒有重新訓練、沒有教師 API 費用，也沒有將重新計分冒充新生成的 3B＋工具分數。', '',
        '原始 RESULTS.json 全部保留；本報告與新 JSON 是新增的評分視角，不修改舊的成功門檻或宣稱所有蒸餾實驗均成功。原始嚴格分與本次分數的差距可能包含輸出格式問題，不全是算術錯誤。', '',
        '## 可重現', '',
        '執行 `python scripts/audit_historical_formulations.py`。逐題 ledger 含原始回答、題目、來源、雜湊、兩層判分、理由與可執行算式；判分標註明列於腳本，並非通用自動閱卷器。來源雜湊固定，資料改動時須重新審核。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    data = audit()
    (DOCS / 'STATS_FORMULATION_AUDIT.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (DOCS / 'STATS_FORMULATION_AUDIT.md').write_text(report(data))
    print(json.dumps(data['summaries'],ensure_ascii=False,indent=2))
