"""Portable entry point for the five recorded YOLO26 experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/yolo26'
RUNS = ('n640', 's640', 'm640', 'n960', 's960')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=2, default=str) + '\n').encode())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fingerprint(records):
    pairs = sorted((r['split'], r['sha256'], r['label_sha256']) for r in records)
    return hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()


def scan(base, splits):
    from PIL import Image
    records = []
    for split in splits:
        images = sorted(p for p in (base / 'images' / split).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        labels = list((base / 'labels' / split).glob('*.txt'))
        if len({p.stem for p in images}) != len(images) or {p.stem for p in images} != {p.stem for p in labels}:
            raise ValueError(f'Image/label mismatch: {base}/{split}')
        for index, image in enumerate(images, 1):
            label = base / 'labels' / split / (image.stem + '.txt')
            with Image.open(image) as im:
                im.load()
            for line in label.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if len(values) != 5 or not all(map(math.isfinite, values)):
                    raise ValueError(f'Invalid label: {label}')
                c, x, y, w, h = values
                if c not in (0, 1, 2) or min(w, h) <= 0 or min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1.00001:
                    raise ValueError(f'Invalid box: {label}')
            if index % 1000 == 0:
                print(f'Checking {base.name}/{split}: {index}/{len(images)}', flush=True)
            records.append(dict(split=split, image=str(image.resolve()), label=str(label.resolve()), sha256=sha(image), label_sha256=sha(label)))
    return records


def prepare(args):
    import yaml
    expected = read(CONFIG / 'datasets.json')
    names = ['legacy_c4d'] + (['bollard_val_2146', 'nightowls_rec34'] if args.external else [])
    groups = {}
    for name in names:
        records = scan(args.data_root / name, expected[name]['split_counts'])
        if fingerprint(records) != expected[name]['content_sha256']:
            raise ValueError(f'Dataset differs from the recorded experiment: {name}')
        groups[name] = records
        print(f'Verified {name}: {len(records)} images', flush=True)
    train = {r['sha256'] for r in groups['legacy_c4d'] if r['split'] == 'train'}
    val = {r['sha256'] for r in groups['legacy_c4d'] if r['split'] == 'val'}
    if train & val:
        raise ValueError('Train/validation overlap')
    if args.external:
        dev_stems = {Path(r['image']).stem for r in groups['legacy_c4d'] if r['split'] == 'val'}
        groups['bollard_dev_disjoint'] = [r for r in groups['bollard_val_2146'] if r['sha256'] not in val and Path(r['image']).stem not in dev_stems]
        for name in ('bollard_val_2146', 'nightowls_rec34'):
            if train & {r['sha256'] for r in groups[name]}:
                raise ValueError(f'Train/external overlap: {name}')
        train_stems = {Path(r['image']).stem.removeprefix('no_') for r in groups['legacy_c4d'] if r['split'] == 'train'}
        if train_stems & {Path(r['image']).stem for r in groups['nightowls_rec34']}:
            raise ValueError('Train/NightOwls source ID overlap')
    # Write only to the chosen workspace after all datasets pass.
    for name, records in groups.items():
        if fingerprint(records) != expected[name]['content_sha256']:
            raise ValueError(f'Dataset fingerprint mismatch: {name}')
        dest = args.workspace / 'prepared' / name
        dest.mkdir(parents=True, exist_ok=True)
        for split in expected[name]['split_counts']:
            (dest / f'{split}.txt').write_bytes((''.join(r['image'].replace('\\', '/') + '\n' for r in records if r['split'] == split)).encode())
        spec = {'path': dest.as_posix(), 'train': 'train.txt' if name == 'legacy_c4d' else 'DO_NOT_TRAIN', 'val': 'val.txt', 'names': {0: 'person', 1: 'stairs', 2: 'bollard'}}
        (dest / 'data.yaml').write_bytes(yaml.safe_dump(spec, sort_keys=False).encode())
        write(dest / 'audit.json', {'content_sha256': fingerprint(records), 'records': records})
    print('Preparation passed; no images copied or modified.')


def check_data(workspace, dataset):
    dest = workspace / 'prepared' / dataset
    audit = read(dest / 'audit.json')
    expected = read(CONFIG / 'datasets.json')[dataset]['content_sha256']
    if fingerprint(audit['records']) != expected:
        raise ValueError('Audit does not match the historical dataset')
    for r in audit['records']:
        if sha(r['image']) != r['sha256'] or sha(r['label']) != r['label_sha256']:
            raise ValueError(f'Data changed after preparation: {r["image"]}')
    # Prevent edited input lists from bypassing the audit.
    for split in {r['split'] for r in audit['records']}:
        actual = (dest / f'{split}.txt').read_text(encoding='utf-8').splitlines()
        wanted = [r['image'].replace('\\', '/') for r in audit['records'] if r['split'] == split]
        if actual != wanted:
            raise ValueError('Prepared image list changed; run prepare again')
    import yaml
    spec = yaml.safe_load((dest / 'data.yaml').read_text())
    expected_spec = {'path': dest.as_posix(), 'train': 'train.txt' if dataset == 'legacy_c4d' else 'DO_NOT_TRAIN', 'val': 'val.txt', 'names': {0: 'person', 1: 'stairs', 2: 'bollard'}}
    if spec != expected_spec:
        raise ValueError('Prepared dataset YAML changed; run prepare again')
    return dest / 'data.yaml', expected


def runtime(config):
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.runtime'))
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import torch
    import ultralytics
    for key, actual in [('torch', torch.__version__), ('ultralytics', ultralytics.__version__)]:
        if actual != config['environment'][key]:
            raise RuntimeError(f'{key}: expected {config["environment"][key]}, found {actual}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is disabled')
    return torch, ultralytics


def e2e(validator):
    if not getattr(validator, 'end2end', False):
        raise RuntimeError('Expected an end-to-end validator')


def execute(args):
    config = read(CONFIG / f'{args.run}.json')
    torch, ultralytics = runtime(config)
    data, data_hash = check_data(args.workspace, 'legacy_c4d' if args.phase == 'train' else args.dataset)
    dest = args.workspace / 'runs' / args.run
    weights = (args.pretrained_dir / f'{config["model"]}.pt') if args.phase == 'train' else args.weights
    if weights is None or not weights.is_file():
        raise ValueError('A local weights file is required')
    weight_hash = sha(weights)
    if args.phase == 'train' and weight_hash != config['pretrained_sha256']:
        raise ValueError('Pretrained checkpoint differs from the historical run')
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if not model.model.end2end:
        raise ValueError('Expected a YOLO26 end-to-end model')
    model.add_callback('on_val_batch_end', e2e)
    settings = dict(config['resolved_args']) if args.phase == 'train' else read(CONFIG / 'evaluation.json')
    for key in ('model', 'data', 'project', 'name', 'save_dir', 'mode'):
        settings.pop(key, None)
    if args.phase == 'train':
        settings.update(project=str(dest.parent), name=dest.name, exist_ok=False)
    else:
        dest = args.output
        if dest is None:
            raise ValueError('--output is required for evaluation')
        settings.update(project=str(dest.parent), name=dest.name, exist_ok=False)
    settings['data'] = str(data)
    if dest.exists():
        raise ValueError(f'Output already exists; choose a fresh workspace/output: {dest}')
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=dict(settings))
    print(json.dumps({'phase': args.phase, 'run': args.run, 'gpu': torch.cuda.get_device_name(0), 'weights_sha256': weight_hash, 'data_content_sha256': data_hash, 'settings': settings}, indent=2), flush=True)
    if args.check_only:
        print('Check passed. No training or evaluation started.')
        return
    started = time.monotonic()
    record = {'run': args.run, 'weights_sha256': weight_hash, 'data_content_sha256': data_hash, 'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'settings': settings}
    if args.phase == 'train':
        def on_start(trainer):
            write(dest / 'run_manifest.json', {**record, 'resolved_args': vars(trainer.args)})
        def on_epoch(trainer):
            write(dest / 'progress.json', {'state': 'training', 'epoch': trainer.epoch + 1, 'epochs': trainer.epochs, 'metrics': {k: float(v) for k,v in trainer.metrics.items()}})
        model.add_callback('on_train_start', on_start)
        model.add_callback('on_fit_epoch_end', on_epoch)
        model.train(**settings)
        progress = read(dest / 'progress.json')
        progress.update(state='complete', epoch=settings['epochs'])
        write(dest / 'progress.json', progress)
        write(dest / 'training_complete.json', {'state': 'complete', 'elapsed_seconds': time.monotonic()-started, 'best_sha256': sha(dest / 'weights/best.pt'), 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
    else:
        metrics = model.val(**settings)
        classes = {0,1,2} if args.dataset == 'legacy_c4d' else {0} if args.dataset == 'nightowls_rec34' else {0,2}
        per_class = {model.names[int(c)]: {'ap50': float(metrics.box.ap50[i]), 'ap50_95': float(metrics.box.ap[i]), 'precision_at_library_selected_point': float(metrics.box.p[i]), 'recall_at_library_selected_point': float(metrics.box.r[i])} for i,c in enumerate(metrics.box.ap_class_index) if int(c) in classes}
        write(dest / 'evaluation.json', {**record, 'dataset': args.dataset, 'per_class': per_class, 'speed_ms': metrics.speed})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='phase', required=True)
    prep = sub.add_parser('prepare', help='Validate existing resampled YOLO data; create local input lists')
    prep.add_argument('--data-root', type=Path, required=True)
    prep.add_argument('--external', action='store_true')
    for phase in ('train', 'evaluate'):
        cmd = sub.add_parser(phase)
        cmd.add_argument('--run', choices=RUNS, required=True)
        cmd.add_argument('--check-only', action='store_true')
        if phase == 'train':
            cmd.add_argument('--pretrained-dir', type=Path, required=True)
        else:
            cmd.add_argument('--weights', type=Path, required=True)
            cmd.add_argument('--output', type=Path, required=True)
            cmd.add_argument('--dataset', choices=tuple(read(CONFIG / 'datasets.json')), default='legacy_c4d')
    for cmd in (prep, *[sub.choices[n] for n in ('train','evaluate')]):
        cmd.add_argument('--workspace', type=Path, required=True)
    args = p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value, Path):
            setattr(args,key,value.expanduser().resolve())
    (prepare if args.phase == 'prepare' else execute)(args)


if __name__ == '__main__':
    main()
