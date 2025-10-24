import argparse
import json
import os
import re
from collections import defaultdict
from typing import Optional


def strip_ext(vid: str) -> str:
    return vid[:-4] if vid.lower().endswith('.mp4') else vid


def guess_video_id(inst: dict) -> str:
    # Prefer explicit field
    if 'video_id' in inst and inst['video_id']:
        return strip_ext(str(inst['video_id']))
    # Else derive from url/path
    url = inst.get('url', '')
    base = os.path.basename(url)
    if base:
        base = strip_ext(base)
        return base
    raise ValueError('Cannot determine video_id for instance: {}'.format(inst))


def guess_user_from_video_id(video_id: str) -> Optional[str]:
    m = re.search(r'(user\d{2})', video_id)
    return m.group(1) if m else None


def count_frames(video_root: str, video_id: str) -> int:
    try:
        import cv2  # type: ignore
    except Exception:
        return 0

    path = os.path.join(video_root, video_id + '.mp4')
    if not os.path.exists(path):
        return 0
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n


def load_arabic_json(path: str):
    data = json.load(open(path, 'r', encoding='utf-8'))
    # Supported flavors:
    # - list of {label, instances:[{video_id,url,...}]}
    if isinstance(data, list):
        return data
    raise ValueError('Unsupported JSON format. Expect a list of label entries with instances.')


def to_nslt(arabic_entries, video_root: str, strategy: str, test_user: str | None,
            train_ratio: float, seed: int):
    import random

    # Collect all items and build class map
    labels = []
    items = []  # (video_id, label)
    for entry in arabic_entries:
        label = entry.get('label') or entry.get('gloss')
        if label is None:
            continue
        labels.append(label)
        for inst in entry.get('instances', []):
            vid = guess_video_id(inst)
            items.append((vid, label))

    classes = sorted(set(labels))
    cls_to_id = {c: i for i, c in enumerate(classes)}

    # Group by user for LOO
    by_user = defaultdict(list)
    for vid, label in items:
        user = guess_user_from_video_id(vid) or 'unknown'
        by_user[user].append((vid, label))

    out = {}
    rng = random.Random(seed)

    if strategy == 'loo':
        if not test_user:
            raise ValueError('For leave-one-out, provide --test_user like user01')
        for user, vids in by_user.items():
            subset = 'test' if user == test_user else 'train'
            for vid, label in vids:
                n = count_frames(video_root, vid)
                out[vid] = {
                    'subset': subset,
                    'action': [cls_to_id[label], 0, n],
                    'url': vid + '.mp4'
                }
    elif strategy == 'ratio':
        all_items = list(items)
        rng.shuffle(all_items)
        split = int(len(all_items) * train_ratio)
        train_items = set(all_items[:split])
        for vid, label in items:
            subset = 'train' if (vid, label) in train_items else 'test'
            n = count_frames(video_root, vid)
            out[vid] = {
                'subset': subset,
                'action': [cls_to_id[label], 0, n],
                'url': vid + '.mp4'
            }
    else:
        raise ValueError('Unknown strategy: {}'.format(strategy))

    return out, cls_to_id


def to_wlasl(arabic_entries, strategy: str, test_user: str | None, train_ratio: float, seed: int):
    import random

    rng = random.Random(seed)

    # Build entries with split per instance
    out = []
    for entry in arabic_entries:
        label = entry.get('label') or entry.get('gloss')
        if label is None:
            continue
        new_instances = []
        insts = list(entry.get('instances', []))
        if strategy == 'ratio':
            rng.shuffle(insts)
            split_idx = int(len(insts) * train_ratio)
            train_insts = set(insts[:split_idx])
        for inst in insts:
            vid = guess_video_id(inst)
            user = guess_user_from_video_id(vid) or 'unknown'
            if strategy == 'loo':
                subset = 'test' if (test_user and user == test_user) else 'train'
            else:
                subset = 'train' if inst in train_insts else 'test'
            new_instances.append({
                'video_id': vid,
                'split': subset,
                'frame_start': int(inst.get('frame_start', 0)),
                'frame_end': int(inst.get('frame_end', 0))
            })
        out.append({'gloss': label, 'instances': new_instances})
    return out


def main():
    p = argparse.ArgumentParser(description='Convert Arabic dataset JSON to NSLT or WLASL formats')
    p.add_argument('--input', required=True, help='Path to arabic_sign_language_dataset.json (list format)')
    p.add_argument('--video_root', default='videos', help='Folder where <video_id>.mp4 resides')
    p.add_argument('--output', required=True, help='Output JSON path')
    p.add_argument('--format', choices=['nslt', 'wlasl'], default='nslt', help='Target format')
    p.add_argument('--strategy', choices=['loo', 'ratio'], default='loo', help='Split strategy')
    p.add_argument('--test_user', default=None, help='userXX to hold out for loo')
    p.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio for ratio split')
    p.add_argument('--seed', type=int, default=0, help='Random seed for ratio split')
    args = p.parse_args()

    arabic_entries = load_arabic_json(args.input)

    if args.format == 'nslt':
        out, _ = to_nslt(arabic_entries, args.video_root, args.strategy, args.test_user, args.train_ratio, args.seed)
    else:
        out = to_wlasl(arabic_entries, args.strategy, args.test_user, args.train_ratio, args.seed)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'Wrote {args.format} JSON to {args.output}')


if __name__ == '__main__':
    main()
