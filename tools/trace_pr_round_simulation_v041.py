"""Isolated synthetic PR selection demo; never calls an executor or an LLM API.

This is an executable design illustration, NOT the production TRACE backend.
Run with the existing .venv-optimizer interpreter after activating conda.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from functools import partial
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)

# Full proposed field layout. Continuous bounds [0,1] are SYNTHETIC fixtures.
FIELDS = [
    ("filter_order", "cat", ["before", "after", "both"]),
    ("vector_impl", "cat", ["single", "failure-entry", "multi"]),
    ("graph_impl", "cat", ["BFS", "best-first", "beam", "MCTS", "evolutionary"]),
    ("filter_pre_impl", "cat", ["LLM", "relational"]),
    ("filter_post_impl", "cat", ["LLM", "relational"]),
    ("rank_impl", "cat", ["ColBERT", "LLM", "cross-encoder"]),
    ("retrieval_unit", "cat", ["whole", "actions", "window"]),
    ("injection_timing", "cat", ["start", "periodic", "event"]),
    ("seed_count", "ord", [4, 8, 16, 32, 64, 128]),
    ("failure_slots", "ord", [0, 1, 2]),
    ("depth", "ord", [2, 3, 4]),
    ("fanout", "ord", [4, 8, 16, 32, 64]),
    ("edge_filter_enabled", "cat", [False, True]),
    ("edge_threshold", "cont", [0.0, 1.0]),
    ("beam_width", "ord", [1, 2, 4, 8]),
    ("sem_threshold_pre", "ord", [0.4, 0.6, 0.8]),
    ("sem_threshold_post", "ord", [0.4, 0.6, 0.8]),
    ("retrieval_interval", "ord", [5, 10]),
    ("returned_k", "ord", [3, 5, 8, 10]),
    ("trajectory_category", "cat", ["Success", "All"]),
    ("graph_direction", "cat", ["Backward", "Forward"]),
]
MASKS = [3, 4, 9, 13, 14, 15, 16, 17]
CAT = [i for i, (_, t, _) in enumerate(FIELDS) if t == "cat"] + list(range(21, 29))
DEFAULT = {name: values[0] for name, _, values in FIELDS}
DEFAULT.update(seed_count=16, fanout=8, beam_width=2, edge_threshold=0.5,
               returned_k=5, sem_threshold_pre=0.6, sem_threshold_post=0.6)


def active(c):
    pre, post = c["filter_order"] != "after", c["filter_order"] != "before"
    return {3: pre, 4: post, 9: c["vector_impl"] == "failure-entry",
            13: c["edge_filter_enabled"], 14: c["graph_impl"] == "beam",
            15: pre and c["filter_pre_impl"] == "LLM",
            16: post and c["filter_post_impl"] == "LLM",
            17: c["injection_timing"] == "periodic"}


def canonical(c):
    out = {name: c.get(name) if c.get(name) is not None else DEFAULT[name]
           for name, _, _ in FIELDS}
    a = active(out)
    for i, (name, typ, values) in enumerate(FIELDS):
        if not a.get(i, True):
            out[name] = None
        elif typ == "cont":
            assert math.isfinite(out[name]) and 0 <= out[name] <= 1
            out[name] = float(out[name])
        else:
            assert out[name] in values, (name, out[name])
    return out


def key(c):
    return hashlib.sha256(json.dumps(canonical(c), sort_keys=True).encode()).hexdigest()


def levels(i):
    _, typ, values = FIELDS[i]
    if typ == "cat":
        return torch.arange(len(values), dtype=torch.double)
    z = torch.tensor(values)
    if i in (8, 11, 14):
        z = z.log2()
    return (z-z[0])/(z[-1]-z[0])


def encode(c):
    c = canonical(c)
    a = active({k: DEFAULT[k] if v is None else v for k, v in c.items()})
    x = []
    for i, (name, typ, values) in enumerate(FIELDS):
        v = c[name]
        x.append(0.0 if v is None else v if typ == "cont"
                 else float(levels(i)[values.index(v)]))
    return torch.tensor(x + [float(a[i]) for i in MASKS])


def decode(x):
    c = {}
    for i, (name, typ, values) in enumerate(FIELDS):
        c[name] = float(x[i]) if typ == "cont" else values[int((levels(i)-x[i]).abs().argmin())]
    return canonical(c)


def plan(c):
    parts = [f"V({c['vector_impl']})"]
    if c['filter_pre_impl'] is not None:
        parts.append(f"F_pre({c['filter_pre_impl']})")
    parts.append(f"G({c['graph_impl']})")
    if c['filter_post_impl'] is not None:
        parts.append(f"F_post({c['filter_post_impl']})")
    parts.append(f"Rank({c['rank_impl']})")
    return " → ".join(parts)


def sobol(n, seed, avoid=()):
    engine = torch.quasirandom.SobolEngine(21, scramble=True, seed=seed)
    found, blocked = {}, set(avoid)
    for row in engine.draw(4096).double():
        c = {name: float(row[i]) if typ == "cont"
             else values[min(int(row[i]*len(values)), len(values)-1)]
             for i, (name, typ, values) in enumerate(FIELDS)}
        c = canonical(c)
        if key(c) not in blocked:
            found.setdefault(key(c), c)
        if len(found) == n:
            return list(found.values())
    return list(found.values())


def synthetic_metrics(c):
    """Invented quality and costs. No relationship to measured ScienceWorld results."""
    seed, depth, fan = c['seed_count'], c['depth'], c['fanout']
    edge = c['edge_threshold'] if c['edge_filter_enabled'] else 0.0
    filters = sum(c[n] == 'LLM' for n in ('filter_pre_impl', 'filter_post_impl'))
    graph_bonus = {'BFS': 0, 'best-first': 4, 'beam': 8, 'MCTS': 10, 'evolutionary': 6}[c['graph_impl']]
    quality = (13 + 6*math.log2(seed) + 2*depth + graph_bonus + 4*filters
               + 5*(c['vector_impl']=='multi') + 3*(c['rank_impl']=='cross-encoder')
               - 16*(edge-0.55)**2 - 3*max(0, math.log2(fan)-4)
               + 5*(c['graph_impl']=='beam' and seed >= 32))
    load = seed*depth*(1+0.15*fan)*(1-0.5*edge)
    timing = {'start':1, 'periodic':1.5, 'event':1.2}[c['injection_timing']]
    tokens = (2300+7*load+550*filters+220*c['returned_k'])*timing
    latency = (13+0.035*load+3*filters+{'ColBERT':1,'LLM':8,'cross-encoder':3}[c['rank_impl']])*timing
    return [quality, tokens, latency]


def raw_sample_to_x(raw):
    """Canonical tensor mapping preserves the pathwise edge-threshold gradient."""
    a = {3: raw[:,0] != 1, 4: raw[:,0] != 0, 9: raw[:,1] == 1,
         13: raw[:,12] == 1, 14: raw[:,2] == 2,
         15: (raw[:,0] != 1) & (raw[:,3] == 0),
         16: (raw[:,0] != 0) & (raw[:,4] == 0), 17: raw[:,7] == 1}
    cols=[]
    for i, (_, typ, _) in enumerate(FIELDS):
        col = raw[:,i] if typ == 'cont' else levels(i)[raw[:,i].long()]
        cols.append(col * a[i].double() if i in a else col)
    return torch.stack(cols+[a[i].double() for i in MASKS], -1), a


class PRState:
    def __init__(self, source, c):
        self.source, self.start, self.params = source, c, []
        for name, typ, values in FIELDS:
            v = c[name] if c[name] is not None else DEFAULT[name]
            if typ == 'cat':
                q = torch.full((len(values),), 0.4/(len(values)-1))
                q[values.index(v)] = 0.6
                p = 0.5 + 0.1*q.log()
            elif typ == 'ord':
                idx = values.index(v)
                p = torch.tensor(idx + (0.4 if idx < len(values)-1 else -0.4))
            else:
                p = torch.tensor(v)
            self.params.append(torch.nn.Parameter(p))
        self.opt = torch.optim.Adam(self.params, lr=0.05)
        self.candidates, self.history = {}, []
        self.stale, self.stopped = 0, False
        self.ema = 0.0

    def sample(self, n):
        vals, logps = [], []
        for p, (_, typ, values) in zip(self.params, FIELDS):
            if typ == 'cat':
                dist = torch.distributions.Categorical(logits=(p-0.5)/0.1)
                v = dist.sample((n,))
                vals.append(v.double()); logps.append(dist.log_prob(v))
            elif typ == 'ord':
                low = p.detach().floor().clamp(0, len(values)-2)
                dist = torch.distributions.Bernoulli(logits=(p-low-0.5)/0.1)
                z = dist.sample((n,))
                vals.append(low+z); logps.append(dist.log_prob(z))
            else:
                vals.append(p.expand(n)); logps.append(torch.zeros(n))
        x, a = raw_sample_to_x(torch.stack(vals, -1))
        lp = torch.stack([v*a[i] if i in a else v for i,v in enumerate(logps)], -1).sum(-1)
        return x, lp

    def clamp(self):
        with torch.no_grad():
            for p, (_, typ, values) in zip(self.params, FIELDS):
                p.clamp_(0, len(values)-1 if typ == 'ord' else 1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--deadline', type=float, default=60)
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source_bytes=Path(__file__).read_bytes()
    (out/'simulation_source.py').write_bytes(source_bytes)
    torch.manual_seed(args.seed)
    start_time=time.monotonic()
    init=[]
    for order in ['before','after','both']:
        for g in ['BFS','beam']:
            init.append(canonical(dict(DEFAULT,filter_order=order,graph_impl=g)))
    init += sobol(4,args.seed,avoid=[key(c) for c in init])
    measured={key(c) for c in init}
    X=torch.stack([encode(c) for c in init])
    raw=torch.tensor([synthetic_metrics(c) for c in init])
    y=torch.stack([raw[:,0],-raw[:,1].log1p(),-raw[:,2].log1p()],-1)
    lo=y.amin(0); span=(y.amax(0)-lo).clamp_min(torch.tensor([1,1e-3,1e-3]))
    z=(y-lo)/span
    models=[MixedSingleTaskGP(X,z[:,i:i+1],cat_dims=CAT,
                train_Yvar=torch.full((10,1),1e-4),outcome_transform=Standardize(m=1),
                cont_kernel_factory=partial(get_covar_module_with_dim_scaled_prior,use_rbf_kernel=False))
            for i in range(3)]
    model=ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood,model),
                     optimizer_kwargs={'options':{'maxiter':60}})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    acq=qLogNoisyExpectedHypervolumeImprovement(model=model,ref_point=[-0.1]*3,
          X_baseline=X,sampler=SobolQMCNormalSampler(torch.Size([64]),seed=args.seed),
          prune_baseline=False,cache_root=False)
    fit_seconds=time.monotonic()-start_time
    round_start=time.monotonic()
    scores, evaluated_vectors, forwards={}, {}, 0

    def score_x(xx):
        nonlocal forwards
        forwards+=len(xx)
        return torch.cat([acq(xx[k:k+32].unsqueeze(1)).reshape(-1)
                          for k in range(0,len(xx),32)])

    def score_configs(cs):
        unseen={key(c):c for c in cs if key(c) not in scores}
        if unseen:
            cfgs=list(unseen.values())
            with torch.no_grad():
                vs=score_x(torch.stack([encode(c) for c in cfgs]))
            for c,v in zip(cfgs,vs):
                scores[key(c)]=float(v); evaluated_vectors[key(c)]=encode(c).tolist()

    pool={}
    def add(c, source):
        k=key(c)
        if k in measured:
            return
        if k not in pool:
            pool[k]={'config':c,'sources':[]}
        if source not in pool[k]['sources']:
            pool[k]['sources'].append(source)

    breadth=sobol(256,args.seed+100,measured)
    mock_llm=[canonical(dict(DEFAULT,vector_impl='multi',graph_impl='beam',seed_count=s,
                 filter_order=o,edge_filter_enabled=True,edge_threshold=t,rank_impl=r))
              for s,o,t,r in [(32,'after',.55,'ColBERT'),(16,'before',.7,'cross-encoder'),
                              (64,'both',.4,'LLM'),(8,'after',.8,'ColBERT')]]
    front=[init[i] for i in is_non_dominated(z).nonzero().flatten().tolist()]
    parents=sorted(front,key=key)[:5]
    # Bounded block crossover; no new synthetic metric calls during proposals.
    recomb={}
    blocks=[['filter_order','filter_pre_impl','filter_post_impl','sem_threshold_pre','sem_threshold_post'],
            ['graph_impl','depth','fanout','beam_width','edge_filter_enabled','edge_threshold','graph_direction'],
            ['vector_impl','seed_count','failure_slots'],['rank_impl','returned_k'],
            ['retrieval_unit','injection_timing','retrieval_interval']]
    for a in parents:
        for b in parents:
            for block in blocks:
                c=canonical(dict(a,**{n:b[n] for n in block}))
                if key(c) not in measured:
                    recomb.setdefault(key(c),c)
                if len(recomb)>=64:
                    break
            if len(recomb)>=64: break
        if len(recomb)>=64: break
    channels={'sobol':breadth,'llm_mock':mock_llm,'recombination':list(recomb.values())}
    for source,cs in channels.items():
        score_configs(cs)
        for c in cs: add(c,source)

    llm_start=max(mock_llm,key=lambda c:(scores[key(c)],key(c)))
    incumbent=max(front,key=lambda c:(synthetic_metrics(c)[0],key(c)))
    random_start=sobol(1,args.seed+300,measured | {key(llm_start)})[0]
    used={key(llm_start),key(incumbent),key(random_start)}
    pool_start=max((r['config'] for k,r in pool.items() if k not in used),
                   key=lambda c:(scores[key(c)],key(c)))
    starts=[('llm',llm_start),('incumbent',incumbent),('random',random_start),('pool',pool_start)]
    states=[PRState(s,c) for s,c in starts]
    score_configs([c for _,c in starts])
    init_probs={s.source:[p.detach().tolist() for p in s.params] for s in states}
    counts={'pr_draws':0,'pr_steps':0,'internal_duplicate_draws':0}
    stop_reason='step_limit'
    for step in range(args.steps):
        for st in states:
            if st.stopped: continue
            if time.monotonic()-round_start>=args.deadline:
                stop_reason='deadline'; break
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 4*1024*1024:
                stop_reason='rss_limit'; break
            xx,lp=st.sample(128)
            cs=[decode(row) for row in xx.detach()]
            ks=[key(c) for c in cs]
            unique_indices={}
            for i,k in enumerate(ks): unique_indices.setdefault(k,i)
            new=set(ks)-scores.keys()
            if len(scores)+len(new)>10000:
                stop_reason='unique_budget'; break
            indices=list(unique_indices.values())
            back={ks[i]:j for j,i in enumerate(indices)}
            vals_unique=score_x(xx[indices])
            vals=vals_unique[torch.tensor([back[k] for k in ks])]
            assert torch.isfinite(vals).all()
            # First step: leave-one-out. Later steps: bias-corrected preceding EMA.
            detached=vals.detach()
            baseline=((detached.sum()-detached)/(len(detached)-1) if not st.history
                      else st.ema/(1-0.7**len(st.history)))
            loss=-((detached-baseline)*lp).mean()-vals.mean()
            st.opt.zero_grad(); loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in st.params)
            st.opt.step(); st.clamp()
            st.ema=0.7*st.ema+0.3*float(detached.mean())
            for c,k,v,row in zip(cs,ks,detached,xx.detach()):
                scores[k]=float(v); evaluated_vectors[k]=row.tolist()
                if k not in measured:
                    st.candidates[k]=c
            st.candidates=dict(sorted(st.candidates.items(),key=lambda kv:(-scores[kv[0]],kv[0]))[:8])
            counts['pr_draws']+=128; counts['pr_steps']+=1
            counts['internal_duplicate_draws']+=128-len(indices)
            forbidden=sum(k in measured for k in ks)/128
            st.stale=st.stale+1 if forbidden>=.8 or not new else 0
            st.stopped=st.stale>=2
            st.history.append({'step':step+1,'mean_log_acq':float(detached.mean()),
                'best_log_acq':float(detached.max()),'unique_in_batch':len(indices),
                'new_keys':len(new),'forbidden_fraction':forbidden,
                'edge_control_after':float(st.params[13].detach())})
        else:
            if all(s.stopped for s in states):
                stop_reason='stagnation';break
            continue
        break
    for st in states:
        for c in st.candidates.values(): add(c,'pr_'+st.source)
    ranked=sorted(pool.items(),key=lambda kv:(-scores[kv[0]],kv[0]))
    winner_key,winner=ranked[0]
    selected=winner['config']
    encoded=encode(selected)
    assert winner_key not in measured
    assert canonical(decode(encoded))==selected
    assert all(torch.allclose(encode(decode(torch.tensor(row))),torch.tensor(row),atol=1e-12,rtol=0)
               for row in evaluated_vectors.values())
    assert len(set(key(c) for _,c in starts))==4
    assert len(scores)<=10000 and len(pool)<=356
    assert counts['pr_draws']<=4*50*128
    assert abs(scores[winner_key]-max(scores[k] for k in pool))<1e-12
    with torch.no_grad():
        pred_z=model.posterior(encoded.reshape(1,-1)).mean[0]
    pred_y=pred_z*span+lo
    predictions={'quality_score':float(pred_y[0]),'total_tokens_inverse_log_mean':float(torch.expm1(-pred_y[1])),
                 'total_latency_s_inverse_log_mean':float(torch.expm1(-pred_y[2]))}
    result={'synthetic_only':True,'production_backend_integrated':False,'real_agent_calls':0,'llm_api_calls':0,
      'seed':args.seed,'schema':{'fields':[n for n,_,_ in FIELDS],
       'mask_fields':[FIELDS[i][0] for i in MASKS],'config_fields':21,'gp_dimensions':29,
       'categorical_fields':11,'ordinal_fields':9,'continuous_fields':1,'continuous_fixture_bounds':[0,1]},
      'initialization':[{'config':c,'synthetic_metrics':synthetic_metrics(c)} for c in init],
      'initialization_count':len(init),'objective_lo':lo.tolist(),'objective_span':span.tolist(),
      'ref_point':[-.1]*3,'channels':{k:len(v) for k,v in channels.items()},
      'pr_starts':[{'source':s,'config':c,'plan':plan(c),'initial_acq':scores[key(c)]} for s,c in starts],
      'pr_runs':[{'source':st.source,'steps':len(st.history),'history':st.history,
                  'kept':len(st.candidates),'initial_controls':init_probs[st.source],
                  'final_controls':[p.detach().tolist() for p in st.params]} for st in states],
      'simulation_baseline':'LOO first step, then bias-corrected prior EMA with beta=0.7',
      'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
      'versions':{n:version(n) for n in ['torch','botorch','gpytorch']},
      'pr_reward':'expected_log_acquisition','counts':counts,'unique_acquisition_inputs':len(scores),
      'acquisition_forward_rows':forwards,'final_pool_size':len(pool),'stop_reason':stop_reason,
      'selection':{'config_key':winner_key,'config':selected,'plan':plan(selected),
                   'encoded_vector':encoded.tolist(),'sources':winner['sources'],
                   'qlognehvi':scores[winner_key],'synthetic_gp_predictions':predictions},
      'ranked_pool':[{'rank':i+1,'key':k,'sources':v['sources'],'qlognehvi':scores[k],
                     'plan':plan(v['config']),'config':v['config']} for i,(k,v) in enumerate(ranked)],
      'fit_seconds':fit_seconds,'proposal_seconds':time.monotonic()-round_start,
      'total_seconds':time.monotonic()-start_time,
      'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'checks':{'roundtrip_all_scored':True,'four_unique_starts':True,'selected_unmeasured':True,
                'winner_is_pool_argmax':True,'no_real_profiling':True,'finite_acq_and_gradients':True},
      'limitations':['Synthetic labels, mock LLM and simulated executor capabilities.',
       'No live graph, agent or task measurements; no full production reservation/recovery integration.',
       'Inverse transformed log-cost means are display values, not unbiased raw cost posterior means.',
       'This spike validates one selection, not PR superiority or the quality of the fitted predictors.']}
    (out/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['channels','counts','unique_acquisition_inputs','final_pool_size',
        'stop_reason','fit_seconds','proposal_seconds','peak_rss_mib','selection','checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
