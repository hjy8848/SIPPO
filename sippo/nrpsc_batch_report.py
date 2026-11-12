"""Offline report on a frozen real n=8 batch (CPU): complete groups, parse rate, class pairs, Planner token masks."""
import pickle, sys, json, collections
import numpy as np, torch
from transformers import AutoTokenizer
from verl.trainer.ppo.nrpsc_objective import validate_sibling_groups, FrozenTfidf, PlanObservation, batch_node_coefficients, load_frozen_tfidf

path, model, idf = sys.argv[1:4]
d = pickle.load(open(path, 'rb'))
t, n, meta = d['tensors'], d['non_tensors'], d['meta']
valid = t['actor_dp_valid_rows'].numpy().astype(bool)
roles = np.asarray(n['role'], dtype=object)
planner = (roles == 'PlanningAgent') & valid
rep = {'rows': int(len(valid)), 'valid_rows': int(valid.sum()),
       'roles': dict(collections.Counter(roles[valid].tolist())), 'planner_rows': int(planner.sum())}
uids = np.asarray(n['uid'], dtype=object)
traj = np.asarray(n['trajectory_id'], dtype=object)
groups = collections.defaultdict(set)
for u, tr in zip(uids[planner], traj[planner]):
    groups[u].add(tr)
rep['questions'] = len(groups)
rep['siblings_per_question'] = dict(collections.Counter(len(g) for g in groups.values()))
rep['complete_questions_n8'] = int(sum(len(g) == 8 for g in groups.values()))
# the batch carries the exact views the trainer built
views = n['nrpsc_plan_view']
pv = [views[i] for i in np.flatnonzero(planner)]
rep['planner_rows_with_parsed_view'] = int(sum(v is not None for v in pv))
rep['planner_parse_rate'] = rep['planner_rows_with_parsed_view'] / max(1, len(pv))
rep['unique_planner_trajectories'] = len(set(traj[planner].tolist()))
# per-question outcome mix
out = np.asarray(n['outcome_exact'], dtype=float)
mix = collections.Counter()
for u in groups:
    rows = np.flatnonzero(planner & (uids == u))
    seen = {}
    for r in rows:
        seen[traj[r]] = out[r]
    s = sum(seen.values())
    mix[f'{int(s)}/{len(seen)}'] += 1
rep['questions_by_successes/siblings'] = dict(sorted(mix.items()))
rep['mixed_outcome_questions'] = int(sum(v for k, v in mix.items() if not k.startswith('0/') and k.split('/')[0] != k.split('/')[1]))
# class pairs and Planner token masks, scored with the frozen IDF at the collection policy (old log-probs)
scorer, sha = load_frozen_tfidf(idf)
rep['idf_sha256'] = sha
obs, seen_traj = [], set()
for r in np.flatnonzero(planner):
    if views[r] is None or traj[r] in seen_traj:
        continue
    seen_traj.add(traj[r])
    obs.append(PlanObservation(uids[r], int(r), views[r], int(out[r]), 1, True))
coef, counts = batch_node_coefficients(t['old_log_probs'].float(), t['response_mask'], obs, scorer,
                                       float(meta['nrpsc_similarity_threshold']), question_count=max(1, len(groups)), beta=1.0)
rep['class_pair_counts'] = {k: (v if not isinstance(v, (list, dict)) else str(v)[:200]) for k, v in counts.items() if k != 'candidate_counts'}
nz = coef.abs() > 0
rep['coef_nonzero_tokens'] = int(nz.sum())
rep['coef_nonzero_outside_planner_rows'] = int((nz & ~torch.as_tensor(planner).unsqueeze(-1)).sum())
rep['coef_nonzero_outside_response_mask'] = int((nz & ~t['response_mask'].bool()).sum())
rep['coef_nonzero_rows'] = int((nz.any(-1)).sum())
rep['coef_abs_sum'] = float(coef.abs().sum())
rep['planner_response_tokens_total'] = int(t['response_mask'][torch.as_tensor(planner)].sum())
print(json.dumps(rep, indent=1, default=str))