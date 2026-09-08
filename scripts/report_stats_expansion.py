"""Produce an evidence-only report after all selections and reviews complete."""
import json,argparse,datetime,zoneinfo
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    def read(p):return json.loads(p.read_text())
    summary=read(root/'EXPANSION_SUMMARY.json');assert not summary['pending']
    teacher=read(root/'teacher/results.json');usage=read(root/'teacher/usage.json')
    corpus=read(Path(__file__).resolve().parents[1]/'docs/STATS_EXPANSION_CORPUS.json')
    assert set(summary['probe'])=={'v15','prior_v55_2027','2027','31415'}
    now=datetime.datetime.now(zoneinfo.ZoneInfo('America/Los_Angeles')).isoformat()
    lines=['# 新教材蒸餾結果','',f'更新時間：{now}（洛杉磯）。','',
        f"新增 {teacher['accepted']} 道通過檢查的教師目標，每題兩種敘述；保留原有 516 筆教材，共 {len(corpus['train'])} 筆。",
        '兩個學生均從原始 v15 出發，固定 seeds 2027、31415，學習率 5e-6，各訓練一輪。','',
        '## 教材與限制','',f"教師啟動呼叫 {usage['calls']} 次；已回報費用 USD {usage['cost']:.8f}，另保守預留 USD {usage.get('unknown_billing_reserved_usd',0):.2f} 處理未取得用量的呼叫。",
        '原始格式失敗、一次人工中斷與一次 HTTP 429 均保留。中斷與限流的嘗試沒有重送。整數／分數格式要求是在檢視失敗後補明；後續包裝修復與事件符號等價審核有獨立紀錄，不是教師嚴格格式成功率。',
        '變異數題原先多要求了題目無法決定的平均數中間量；該欄位及其臆測值保留但不採用，只驗證題目可決定的變異數與最終算式，不宣稱完整教師工作正確。',
        '中間量以獨立有理數參考核對；教材答案保留教師算式字元，不以參考答案替換。學生評分器與通過門檻沒有放寬。','',
        '| 教材目標 | 合格題數 |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in teacher['by_target'].items()]
    lines += ['', '## 學生與原有門檻','','| Seed | 更新步數 | 固定評測步數 | 完整門檻通過 |','|---|---:|---:|---|']
    for seed,item in summary['seeds'].items():
        selection=read(root/f'seed_{seed}/selection.json');c=item['training_complete']
        lines.append(f"| {seed} | {c['global_steps']} | {selection['fixed_step']} | {'是' if selection['selected_step'] is not None else '否'} |")
    lines+=['','| Seed／步數 | 舊題 /48 | 推理鏈 /48 | 事件 /16 | 完整門檻 |','|---|---:|---:|---:|---|']
    for seed,item in summary['seeds'].items():
        for h in item['history']:
            m=h['metrics'];lines.append(f"| {seed}／{h['step']} | {m['old']['correct']} | {m['chain']['correct']} | {m['event']['correct']} | {'通過' if h['gate']['passed'] else '未通過'} |")
    lines+=['','## 固定候選後的新案例評測','','64 題包含 8 組事件情境 × 5 種目標，以及 8 組動差情境 × 3 種目標。教師未見這些題目；候選在評測前固定，沒有依結果重選。','',
        '| 模型 | 正確 /64 |','|---|---:|']
    labels={'v15':'原始 v15','prior_v55_2027':'上一批 V55 seed 2027／65','2027':'新 seed 2027','31415':'新 seed 31415'}
    for name in labels:lines.append(f"| {labels[name]} | {summary['probe'][name]['overall']['correct']} |")
    lines+=['','| 題型（各 8 題） | v15 | V55 | 新 2027 | 新 31415 |','|---|---:|---:|---:|---:|']
    for cell in summary['probe']['v15']['cells']:
        lines.append('| '+cell+' | '+' | '.join(str(summary['probe'][name]['cells'][cell]['correct']) for name in labels)+' |')
    lines+=['','## 判讀','',
        '這是小型、篩選後的教材擴充試驗。新案例、教材比例與訓練曝光量同時改變，且教師輸出格式曾修正；不能單獨歸因於案例數量。每個領域僅 8 組新情境，同情境的多種目標彼此相關。',
        '本輪不等於完整教師門檻、四組起點／種子比較與正式保留題升級程序。原始 v15 仍保留；新權重作為可比較的候選學生。','',
        '兩輪皆檢查實際 token／label／教材順序雜湊與微批次數量；所有保存的學生權重另由打包程序檢查張量是否有限，並產生 SHA-256 清單。','']
    (root/'STATS_EXPANSION_REPORT.md').write_text('\n'.join(lines))
    print('EXPANSION_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
