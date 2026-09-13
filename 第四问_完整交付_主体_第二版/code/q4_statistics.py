"""Equal-probability empirical CVaR and paired moving-block bootstrap."""
import numpy as np

def empirical_cvar(values, beta=.95):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 <= beta < 1:
        raise ValueError('finite nonempty 1D losses and 0 <= beta < 1 required')
    x = np.sort(x)[::-1]
    mass = (1-beta)*len(x)
    k = min(int(np.floor(mass)), len(x))
    return float((x[:k].sum() + (mass-k)*x[k] if k<len(x) else x.sum())/mass)

def moving_block_indices(n, block_len, rng):
    if not 1 <= block_len <= n:
        raise ValueError('block length must lie in [1,n]')
    starts = rng.integers(0, n-block_len+1, size=int(np.ceil(n/block_len)))
    return (starts[:,None]+np.arange(block_len)).reshape(-1)[:n]

def paired_bootstrap(a,b,block_len,resamples=4000,seed=20260941,beta=.95):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1: raise ValueError('paired daily arrays required')
    rng=np.random.default_rng(seed)
    dc=np.empty(resamples); dm=np.empty(resamples)
    for i in range(resamples):
        ix=moving_block_indices(len(a),block_len,rng)
        dc[i]=empirical_cvar(b[ix],beta)-empirical_cvar(a[ix],beta)
        dm[i]=(b[ix]-a[ix]).mean()
    ca,cb=empirical_cvar(a,beta),empirical_cvar(b,beta)
    lo,hi=np.quantile(dc,[.025,.975]); ml,mh=np.quantile(dm,[.025,.975])
    return dict(block_len=block_len,cvar95_reference=ca,cvar95_variant=cb,diff=cb-ca,
        diff_percent=100*(cb-ca)/ca,ci_lo=float(lo),ci_hi=float(hi),
        p_worse=float((dc>0).mean()),mean_diff=float((b-a).mean()),
        mean_ci_lo=float(ml),mean_ci_hi=float(mh),resamples=resamples,
        sample_size=len(a),bootstrap_sample_size=len(a),seed=seed,
        statistic='exact equal-probability empirical CVaR95',
        method='paired moving blocks; all valid starts; ceil and truncate to n')
