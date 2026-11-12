import json, glob, math, statistics as st
files = sorted(glob.glob('/home/hejingyu/projects/SIPPO-npu-nrpsc-20261112-r1/_runs/c*/frozen/grad_compare.json'))
rows = []
for f in files:
    tag = f.split('/_runs/')[1].split('/')[0]
    for m in json.load(open(f))['minibatches']:
        m['tag'] = tag
        rows.append(m)
print('files:', [f.split('/_runs/')[1].split('/')[0] for f in files], 'minibatches:', len(rows))
both = [m for m in rows if m.get('grad_norm/psc', 0) > 0 and m.get('grad_norm/nrpsc', 0) > 0]
nr_only = [m for m in rows if m.get('grad_norm/nrpsc', 0) > 0]
psc_only = [m for m in rows if m.get('grad_norm/psc', 0) > 0]
print('minibatches with PSC grad:', len(psc_only), '| NR-PSC grad:', len(nr_only), '| both:', len(both))
for k in ('nrpsc', 'psc', 'nrpsc_no_reward_eligibility'):
    print(k, 'planner-only violations (tokens outside planner rows / response mask):',
          sum(m[f'{k}/nonzero_tokens_outside_planner_rows'] for m in rows), sum(m[f'{k}/nonzero_tokens_outside_response_mask'] for m in rows))
def pooled(a, b):
    return math.sqrt(sum(m[a] ** 2 for m in both)) / math.sqrt(sum(m[b] ** 2 for m in both))
if both:
    w = 0.02
    print('pooled ||g_psc||/||g_nrpsc|| =', round(pooled('grad_norm/psc', 'grad_norm/nrpsc'), 4), '-> norm-matched weight', round(w * pooled('grad_norm/psc', 'grad_norm/nrpsc'), 5))
    per = [m['matched_weight_norm/nrpsc'] for m in both]
    print('per-minibatch matched weight: median', round(st.median(per), 5), 'min', round(min(per), 5), 'max', round(max(per), 5))
    proj = [m['matched_weight_projection/nrpsc'] for m in both if 'matched_weight_projection/nrpsc' in m]
    print('projection weight: median', round(st.median(proj), 5))
    print('cos(psc,nrpsc): median', round(st.median(m['cosine/psc__nrpsc'] for m in both), 4), 'range', round(min(m['cosine/psc__nrpsc'] for m in both), 4), round(max(m['cosine/psc__nrpsc'] for m in both), 4))
    print('cos(base,nrpsc): median', round(st.median(m['cosine/base__nrpsc'] for m in both), 4), '| cos(base,psc): median', round(st.median(m['cosine/base__psc'] for m in both), 4))
    print('weighted/base @0.02: psc median', round(st.median(m['weighted_to_base_ratio_at_0.02/psc'] for m in both), 4), ' nrpsc median', round(st.median(m['weighted_to_base_ratio_at_0.02/nrpsc'] for m in both), 4))
    if any('grad_norm/nrpsc_no_reward_eligibility' in m for m in both):
        off = [m for m in both if m['grad_norm/nrpsc_no_reward_eligibility'] > 0]
        print('no-eligibility variant: minibatches with grad', len(off), '| ||g|| ratio to eligible NR-PSC median', round(st.median(m['grad_norm/nrpsc_no_reward_eligibility'] / m['grad_norm/nrpsc'] for m in off), 3) if off else None)
for m in nr_only:
    print(m['tag'], m['minibatch'], {k: round(m[k], 4) for k in ('grad_norm/base', 'grad_norm/psc', 'grad_norm/nrpsc', 'cosine/psc__nrpsc') if k in m}, 'w*', round(m.get('matched_weight_norm/nrpsc', float('nan')), 5))