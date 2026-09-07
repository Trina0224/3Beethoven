"""Archive completed bounded repairs, excluding bulky optimizer checkpoints."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if (root / 'PREPARED_PARENT_CONTROL_PROTOCOL.md').exists():
        supplemental = json.loads((root / 'prepared_parent/COMPLETE.json').read_text())
        assert supplemental['complete'] and supplemental['pending'] == 0
    trace = json.loads((root / 'ACTUAL_TRACE_VERIFIED.json').read_text())
    selected = {}
    for seed in (2027, 31415):
        folder = root / f'low_lr/original_v15_continued_lora_seed_{seed}'
        result = json.loads((folder / 'training_complete.json').read_text())
        assert result['selection']['stop_reason'] != 'manual_review_required'
        assert trace[str(seed)]['matched_all_actual_microbatches']
        selected[folder.name] = result['selection']['selected_step'] or result['selection']['diagnostic_step']
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in ('checkpoints', '__pycache__', '.git') for part in rel.parts):
            continue
        if path.suffix in ('.zip', '.pyc'):
            continue
        if 'adapter' in rel.parts:
            if len(rel.parts) < 4 or rel.parts[0] != 'low_lr':
                continue
            if rel.parts[2] != f'step_{selected[rel.parts[1]]}':
                continue
        files.append(path)
    manifest = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(root)))
        archive.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
    result = {'path': args.output.name, 'bytes': args.output.stat().st_size,
              'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'files': len(files), 'adapter_steps': selected,
              'note': 'All validation raw answers; final/selected portable adapters. All intermediate adapters and optimizer checkpoints remain in full notebook outputs.'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('REPAIR_ARCHIVE_VERIFIED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
