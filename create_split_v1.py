"""Create a reproducible APTOS split from audited CSVs; never edit images."""
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    root = Path.cwd()
    audit = root / 'environment' / 'data_audit'
    output = audit / 'split_v1'
    names = ['train', 'validation', 'test']
    weights = [70, 15, 15]
    seed = 42
    rng = random.Random(seed)

    def read(name):
        with (audit / name).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    rows = sorted(read('eligible_images_v1.csv'), key=lambda r: r['id_code'])
    decisions = read('data_decisions_v1.csv')
    links = read('near_duplicate_constraints_v1.csv')
    check(len(rows) == 3504, 'Expected 3504 eligible images')
    check(len(decisions) == 3662, 'Expected 3662 decision records')
    check(len({r['id_code'] for r in decisions}) == 3662, 'Repeated decision ID')
    check(len({r['id_code'] for r in rows}) == 3504, 'Repeated eligible ID')
    check(len({r['pixel_sha256'] for r in rows}) == 3504, 'Repeated eligible pixel hash')
    kept = {r['id_code']: r for r in decisions if r['status'] == 'keep'}
    check({r['id_code']: r for r in rows} == kept, 'Eligible and keep records differ')
    check(set(r['diagnosis'] for r in rows) == set('01234'), 'Unexpected labels')
    by_id = {r['id_code']: r for r in decisions}
    check(len(links) == 12, 'Expected 12 precautionary links')
    for link in links:
        a, b = by_id[link['id_a']], by_id[link['id_b']]
        check(a['split_group_id'] == b['split_group_id'], 'Broken near-duplicate constraint')

    groups = defaultdict(list)
    for row in rows:
        groups[row['split_group_id']].append(row)
    check(len(groups) == 3494, 'Unexpected group count')

    # Integer largest-remainder allocation avoids floating-point tie ambiguity.
    # Ties are resolved in train, validation, test order, fixed in advance.
    targets = {}
    for label, count in sorted(Counter(r['diagnosis'] for r in rows).items()):
        quotas = [count * w // 100 for w in weights]
        order = sorted(range(3), key=lambda i: (-(count * weights[i] % 100), i))
        for i in order[:count - sum(quotas)]:
            quotas[i] += 1
        targets[label] = dict(zip(names, quotas))
    remaining = {label: quota.copy() for label, quota in targets.items()}
    assignments = {}

    # Place indivisible multi-image groups first. Weighted random allocation
    # uses the requested ratios and only considers partitions with capacity.
    multi = [g for g in groups.values() if len(g) > 1]
    rng.shuffle(multi)
    multi.sort(key=len, reverse=True)
    for group in multi:
        counts = Counter(r['diagnosis'] for r in group)
        options = [s for s in names if all(remaining[c][s] >= n for c, n in counts.items())]
        check(bool(options), 'Class quotas cannot accommodate an intact group')
        split = rng.choices(options, weights=[weights[names.index(s)] for s in options])[0]
        assignments[group[0]['split_group_id']] = split
        for label, count in counts.items():
            remaining[label][split] -= count

    # Singleton groups fill the remaining class quotas, without breaking groups.
    for label in targets:
        singles = sorted((g[0] for g in groups.values()
                          if len(g) == 1 and g[0]['diagnosis'] == label),
                         key=lambda r: r['id_code'])
        rng.shuffle(singles)
        start = 0
        for split in names:
            count = remaining[label][split]
            for row in singles[start:start + count]:
                assignments[row['split_group_id']] = split
            start += count
        check(start == len(singles), 'Singleton allocation mismatch')

    result = [{**r, 'split': assignments[r['split_group_id']]} for r in rows]
    for field in ['id_code', 'pixel_sha256', 'split_group_id']:
        sets = [{r[field] for r in result if r['split'] == s} for s in names]
        check(all(not (sets[i] & sets[j]) for i in range(3) for j in range(i)),
              'Cross-partition overlap: ' + field)
    for label in targets:
        for split in names:
            check(sum(r['diagnosis'] == label and r['split'] == split for r in result)
                  == targets[label][split], 'Class quota mismatch')
    check(len(result) == len(kept), 'Incomplete allocation')

    def csv_text(items):
        buf = io.StringIO(newline='')
        writer = csv.DictWriter(buf, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(items)
        return buf.getvalue()

    totals = {s: sum(r['split'] == s for r in result) for s in names}
    metadata = {
        'seed': seed, 'ratios': [0.70, 0.15, 0.15],
        'method': 'Intact multi-image groups allocated first; singleton groups fill per-class largest-remainder quotas',
        'class_quota_tie_order': names, 'totals': totals, 'class_counts': targets,
        'eligible_images': len(result), 'split_groups': len(groups),
        'input_file_sha256': {n: hashlib.sha256((audit / n).read_bytes()).hexdigest()
            for n in ['eligible_images_v1.csv', 'data_decisions_v1.csv', 'near_duplicate_constraints_v1.csv']},
        'checks': 'PASS: eligible coverage, excluded IDs absent, class quotas, ID/hash/group isolation, 12 recorded constraints',
        'limitation': 'Uses recorded hashes and reviewed relationships; not a fresh PNG audit, proof of absence of near duplicates, or patient-level isolation.'
    }
    files = {'split_manifest_v1.csv': csv_text(result),
             'split_summary_v1.json': json.dumps(metadata, indent=2) + '\n'}
    for split in names:
        files[split + '_v1.csv'] = csv_text([r for r in result if r['split'] == split])
    # Refuse to replace a different existing split. Identical reruns are allowed.
    for name, content in files.items():
        path = output / name
        check(not path.exists() or path.read_bytes() == content.encode('utf-8'),
              'Existing different output; not overwritten: ' + str(path))
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output / name).write_bytes(content.encode('utf-8'))

    print('Seed:', seed)
    print('Eligible images:', len(result), '| Split groups:', len(groups))
    print('Label | Train | Validation | Test')
    for label, values in targets.items():
        print(label, '|', values['train'], '|', values['validation'], '|', values['test'])
    for split, count in totals.items():
        print(f'{split}: {count} ({100 * count / len(result):.2f}%)')
    print('SPLIT CHECK: PASS')
    print('Saved:', output)
    print('Original images and labels unchanged.')


if __name__ == '__main__':
    main()
