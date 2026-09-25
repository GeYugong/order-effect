"""Post-completion descriptive audit and explicitly post-hoc comparisons; no API calls."""
import collections
import itertools
import json
from pathlib import Path

from analyze import contrast, holm

ROOT = Path(__file__).resolve().parent


def main():
    a = json.loads((ROOT / 'reports/statistics_A.json').read_text(encoding='utf-8'))
    b = json.loads((ROOT / 'reports/statistics_B.json').read_text(encoding='utf-8'))
    groups = {r['order']: r['runs'] for r in b['groups'] if r['method'] == 'merge'}
    pairs = {f'{left} - {right}': contrast(groups, {left: 1, right: -1})
             for left, right in itertools.combinations(groups, 2)}
    holm(pairs)
    invalid = collections.Counter()
    merge_steps = lost_steps = 0
    for path in (ROOT / 'results/B').glob('merge_*/merge_[0-9]*.json'):
        row = json.loads(path.read_text(encoding='utf-8'))
        merge_steps += 1
        if not row['generation'].get('update_valid', True):
            invalid[row['generation']['error']] += 1
        before = {s for rule in row['before'] for s in rule['sources']}
        after = {s for rule in row['after'] for s in rule['sources']}
        lost_steps += bool(before - after)
    canonical = [v for row in b['nested_evaluations'] if row['method'] == 'canonical_all'
                 for v in row['evaluation_scores']]
    result = {
        'analysis_status': 'post-hoc, after seeing all results; original primary analyses unchanged',
        'B_pairwise_family': 'All six unordered comparisons among four merge orders; two-sided Welch tests; Holm over six tests',
        'unit': 'One independently generated playbook, averaging its three evaluations; n=5 per merge order',
        'B_all_pairwise_tests': pairs,
        'ci_note': 'Each ci95 is an unadjusted per-comparison interval, not a simultaneous confidence interval',
        'A_equal_order_weighted_means_descriptive': {
            method: sum(r['mean'] for r in a['groups'] if r['method'] == method) / 4
            for method in ('no_memory', 'reasoning_memory')},
        'canonical_fixed_context_repeats': canonical,
        'merge_steps': merge_steps, 'invalid_merge_updates': dict(invalid),
        'steps_losing_source_labels': lost_steps,
        'limitations': [
            'Post-hoc family selection does not replace independent confirmation or the prespecified primary contrast.',
            'Small samples and discrete scores limit parametric inference.',
            'All conclusions conditional on one fixed experience pool and heldout set; not all possible permutations.',
            'Source tags are model-generated; loss or retention does not establish semantic forgetting or preservation.',
            'Invalid-update skip policy is part of the tested algorithm; mechanism cannot be isolated from it.',
        ],
    }
    (ROOT / 'reports/posthoc_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'pairs': pairs, 'merge_steps': merge_steps, 'invalid_updates': sum(invalid.values()), 'source_loss_steps': lost_steps}))


if __name__ == '__main__':
    main()
