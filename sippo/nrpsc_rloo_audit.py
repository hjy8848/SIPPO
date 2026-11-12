import json, glob, math, statistics as st
W_PSC, W_NR = 0.02, 0.04
files = sorted(glob.glob('/home/hejingyu/projects/SIPPO-npu-nrpsc-20261112-r1/_runs/r[0-9]/replay/grad_compare.json'))
rows = []
for f in files:
    for m in json.load(open(f))['minibatches']:
        m['tag'] = f.split('/_runs/')[1].split('/')[0]
        rows.append(m)
def q(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p; lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
def stats(xs):
    return f'median {st.median(xs):.4f} | p90 {q(xs,0.9):.4f} | max {max(xs):.4f} | min {min(xs):.4f}'
ok = [m for m in rows if m.get('grad_norm/planner_rloo', 0) > 0 and m.get('grad_norm/psc', 0) > 0 and m.get('grad_norm/nrpsc', 0) > 0]
print('replay files:', len(files), '| minibatches:', len(rows), '| with RLOO, PSC and NR-PSC all non-zero:', len(ok))
def rel(m, name, w):
    r, a = m['grad_norm/planner_rloo'], m[f'grad_norm/{name}']
    dot = m[f'dot/planner_rloo__{name}']
    cos = dot / (r * a)
    orth = math.sqrt(max(a * a - dot * dot / (r * r), 0.0))
    return cos, orth / r, w * a / r, w * orth / r
tab = {}
for name, w in (('psc', W_PSC), ('nrpsc', W_NR)):
    vals = [rel(m, name, w) for m in ok]
    tab[name] = vals
print()
print('| metric | PSC (w=0.02) | NR-PSC (w=0.04) |')
print('|---|---|---|')
labels = ['cos(g_aux, g_Planner-RLOO)', 'orthogonal norm / ||g_RLOO|| (unweighted)', 'weighted ||aux|| / ||g_RLOO||', 'weighted orthogonal / ||g_RLOO||']
for i, lab in enumerate(labels):
    a = [v[i] for v in tab['psc']]; b = [v[i] for v in tab['nrpsc']]
    print(f'| {lab} | {stats(a)} | {stats(b)} |')
print()
print('informative questions (PSC, summed over minibatches):', sum(m.get('psc/informative_question_count', 0) for m in rows))
print('informative class pairs (NR-PSC, summed):', sum(m.get('nrpsc_metric/aux/nrpsc/informative_class_pairs', 0) for m in rows))
print('minibatches with PSC grad:', sum(1 for m in rows if m.get('grad_norm/psc', 0) > 0), '| with NR-PSC grad:', sum(1 for m in rows if m.get('grad_norm/nrpsc', 0) > 0))
print('cos(PSC, NR-PSC) on these:', stats([m['cosine/psc__nrpsc'] for m in ok]))
big = [(m['tag'], m['minibatch'], round(rel(m, 'nrpsc', W_NR)[2], 3)) for m in ok if rel(m, 'nrpsc', W_NR)[2] > 1.0]
print('minibatches where 0.04*||g_NR|| > ||g_RLOO||:', len(big), big)
print()
print('per minibatch (tag mb | cos_PSC cos_NR | w*orth_PSC/R w*orth_NR/R | w*PSC/R w*NR/R):')
for m in ok:
    p, n = rel(m, 'psc', W_PSC), rel(m, 'nrpsc', W_NR)
    print(m['tag'], m['minibatch'], '|', round(p[0], 3), round(n[0], 3), '|', round(p[3], 3), round(n[3], 3), '|', round(p[2], 3), round(n[2], 3))