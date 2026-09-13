"""L3 参数寻优层：memetic GA（GA + Nelder-Mead 精修），并与 DE、SA 同预算对照；L4：NSGA-II。

**严格因果**：适应度只在评价期（2025-02-01 起）之前的 1 月滚动验证折上计算。
三个验证块：1/8—1/15、1/16—1/23、1/24—1/31，各自从主模型 1 月预热得到的同一 SOC 起步
（所有候选用同一起点，比较公平）；适应度取三块库存校正成本的平均值，并记录最差块用于稳健性。
"""
from __future__ import annotations
from pathlib import Path
import sys, json, time, os
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'

# ---------------------------------------------------------------- 通用优化器
def ga_optimize(fitness_batch, bounds, pop=20, gens=12, seed=7, pm=0.25, pc=0.9, elite=2, log=None):
    """实数编码 GA：锦标赛选择 + BLX-alpha 交叉 + 高斯变异 + 精英保留。批量评估以便并行。"""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    d = len(bounds)
    X = lo + rng.random((pop, d)) * (hi - lo)
    f = np.array(fitness_batch(X))
    hist = []
    nev = pop
    for g in range(gens):
        o = np.argsort(f); X, f = X[o], f[o]
        hist.append(dict(algorithm='GA', generation=g, evaluations=nev,
                         best=float(f[0]), mean=float(f.mean()), std=float(f.std())))
        if log:
            print(f'  [GA] gen={g} evals={nev} best={f[0]:,.1f} mean={f.mean():,.1f}', flush=True)
        new = [X[i].copy() for i in range(elite)]
        while len(new) < pop:
            i, j = rng.integers(0, pop, 2); p1 = X[i] if f[i] <= f[j] else X[j]
            i, j = rng.integers(0, pop, 2); p2 = X[i] if f[i] <= f[j] else X[j]
            if rng.random() < pc:
                gmin, gmax = np.minimum(p1, p2), np.maximum(p1, p2)
                span = gmax - gmin
                ch = rng.uniform(gmin - 0.4 * span, gmax + 0.4 * span)
            else:
                ch = p1.copy()
            ch = ch + (rng.random(d) < pm) * rng.normal(0, 0.12, d) * (hi - lo)
            new.append(np.clip(ch, lo, hi))
        X = np.clip(np.array(new[:pop]), lo, hi)
        f = np.array(fitness_batch(X)); nev += pop
    o = np.argsort(f)
    hist.append(dict(algorithm='GA', generation=gens, evaluations=nev,
                     best=float(f[o[0]]), mean=float(f.mean()), std=float(f.std())))
    return X[o[0]].copy(), float(f[o[0]]), hist, nev


def de_optimize(fitness_batch, bounds, pop=20, gens=12, seed=11, F=0.7, CR=0.9, log=None):
    """差分进化（rand/1/bin），与 GA 同预算，用于交叉验证「不是某一种启发式的偶然」。"""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float); hi = np.array([b[1] for b in bounds], float)
    d = len(bounds)
    X = lo + rng.random((pop, d)) * (hi - lo)
    f = np.array(fitness_batch(X)); nev = pop
    hist = []
    for g in range(gens):
        hist.append(dict(algorithm='DE', generation=g, evaluations=nev,
                         best=float(f.min()), mean=float(f.mean()), std=float(f.std())))
        if log:
            print(f'  [DE] gen={g} evals={nev} best={f.min():,.1f}', flush=True)
        V = np.empty_like(X)
        for i in range(pop):
            a, b, c = rng.choice([k for k in range(pop) if k != i], 3, replace=False)
            mut = X[a] + F * (X[b] - X[c])
            cross = rng.random(d) < CR
            cross[rng.integers(d)] = True
            V[i] = np.clip(np.where(cross, mut, X[i]), lo, hi)
        fv = np.array(fitness_batch(V)); nev += pop
        better = fv < f
        X[better], f[better] = V[better], fv[better]
    hist.append(dict(algorithm='DE', generation=gens, evaluations=nev,
                     best=float(f.min()), mean=float(f.mean()), std=float(f.std())))
    i = int(np.argmin(f))
    return X[i].copy(), float(f[i]), hist, nev


def sa_optimize(fitness_batch, bounds, steps=12, chains=8, seed=13, T0=1.0, log=None):
    """并行回火模拟退火：chains 条链同步推进，便于批量并行评估。"""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float); hi = np.array([b[1] for b in bounds], float)
    d = len(bounds)
    X = lo + rng.random((chains, d)) * (hi - lo)
    f = np.array(fitness_batch(X)); nev = chains
    bx, bf = X[np.argmin(f)].copy(), float(f.min())
    scale0 = 0.25
    hist = []
    for s in range(steps):
        T = T0 * (0.85 ** s)
        scale = scale0 * (0.85 ** s)
        hist.append(dict(algorithm='SA', generation=s, evaluations=nev,
                         best=float(bf), mean=float(f.mean()), std=float(f.std())))
        if log:
            print(f'  [SA] step={s} evals={nev} best={bf:,.1f} T={T:.3f}', flush=True)
        Y = np.clip(X + rng.normal(0, 1, (chains, d)) * scale * (hi - lo), lo, hi)
        fy = np.array(fitness_batch(Y)); nev += chains
        ref = max(abs(bf), 1.0)
        acc = (fy < f) | (rng.random(chains) < np.exp(-(fy - f) / (T * 1e-3 * ref + 1e-12)))
        X[acc], f[acc] = Y[acc], fy[acc]
        if f.min() < bf:
            bf = float(f.min()); bx = X[np.argmin(f)].copy()
    hist.append(dict(algorithm='SA', generation=steps, evaluations=nev,
                     best=float(bf), mean=float(f.mean()), std=float(f.std())))
    return bx, bf, hist, nev


def nelder_mead(fitness_batch, x0, bounds, maxiter=40, step=0.08, log=None):
    """Nelder-Mead 局部精修（memetic GA 的局部算子）；逐点评估，用批量接口包一层。"""
    lo = np.array([b[0] for b in bounds], float); hi = np.array([b[1] for b in bounds], float)
    d = len(x0)

    def f1(x):
        return float(fitness_batch(np.clip(np.atleast_2d(x), lo, hi))[0])

    simplex = [np.clip(x0, lo, hi)]
    for i in range(d):
        y = simplex[0].copy()
        y[i] = np.clip(y[i] + step * (hi[i] - lo[i]), lo[i], hi[i])
        simplex.append(y)
    simplex = np.array(simplex)
    fv = np.array([f1(x) for x in simplex])
    nev = len(simplex)
    hist = []
    for it in range(maxiter):
        o = np.argsort(fv); simplex, fv = simplex[o], fv[o]
        hist.append(dict(algorithm='NM', generation=it, evaluations=nev, best=float(fv[0]),
                         mean=float(fv.mean()), std=float(fv.std())))
        if log and it % 5 == 0:
            print(f'  [NM] it={it} evals={nev} best={fv[0]:,.1f}', flush=True)
        cen = simplex[:-1].mean(axis=0)
        xr = np.clip(cen + (cen - simplex[-1]), lo, hi); fr = f1(xr); nev += 1
        if fr < fv[0]:
            xe = np.clip(cen + 2 * (cen - simplex[-1]), lo, hi); fe = f1(xe); nev += 1
            simplex[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            simplex[-1], fv[-1] = xr, fr
        else:
            xc = np.clip(cen + 0.5 * (simplex[-1] - cen), lo, hi); fc = f1(xc); nev += 1
            if fc < fv[-1]:
                simplex[-1], fv[-1] = xc, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                fv[1:] = [f1(x) for x in simplex[1:]]; nev += d
        if np.ptp(fv) < 1e-6 * max(abs(fv[0]), 1.0):
            break
    o = np.argsort(fv)
    return simplex[o[0]].copy(), float(fv[o[0]]), hist, nev


# ---------------------------------------------------------------- NSGA-II（多目标）
def nsga2(obj_batch, bounds, pop=16, gens=10, seed=17, pm=0.25, pc=0.9, log=None):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float); hi = np.array([b[1] for b in bounds], float)
    d = len(bounds)

    def dominates(a, b):
        return bool((a <= b).all() and (a < b).any())

    def fronts_of(F):
        n = len(F); S = [[] for _ in range(n)]; nd = np.zeros(n, int); fr = [[]]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if dominates(F[i], F[j]):
                    S[i].append(j)
                elif dominates(F[j], F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                fr[0].append(i)
        k = 0
        while fr[k]:
            nxt = []
            for i in fr[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        nxt.append(j)
            k += 1; fr.append(nxt)
        return fr[:-1]

    def crowd_of(F, idx):
        l = len(idx); dd = np.zeros(l)
        if l <= 2:
            return np.full(l, np.inf)
        for m in range(F.shape[1]):
            o = np.argsort(F[idx, m]); dd[o[0]] = dd[o[-1]] = np.inf
            rg = F[idx, m].max() - F[idx, m].min()
            if rg > 0:
                dd[o[1:-1]] += (F[idx, m][o[2:]] - F[idx, m][o[:-2]]) / rg
        return dd

    X = lo + rng.random((pop, d)) * (hi - lo)
    F = np.array(obj_batch(X)); nev = pop
    for g in range(gens):
        fr = fronts_of(F)
        rank = np.empty(pop, int); crowd = np.empty(pop)
        for r, ff in enumerate(fr):
            rank[ff] = r; crowd[ff] = crowd_of(F, np.array(ff))
        o = np.lexsort((-crowd, rank)); X, F, rank, crowd = X[o], F[o], rank[o], crowd[o]
        if log:
            print(f'  [NSGA2] gen={g} front0={(rank == 0).sum()} best_f1={F[:, 0].min():,.0f}', flush=True)
        new = [X[0].copy(), X[1].copy()]
        while len(new) < pop:
            i, j = rng.integers(0, pop, 2); p1 = X[i] if (rank[i], -crowd[i]) <= (rank[j], -crowd[j]) else X[j]
            i, j = rng.integers(0, pop, 2); p2 = X[i] if (rank[i], -crowd[i]) <= (rank[j], -crowd[j]) else X[j]
            if rng.random() < pc:
                gmin, gmax = np.minimum(p1, p2), np.maximum(p1, p2); span = gmax - gmin
                ch = rng.uniform(gmin - 0.4 * span, gmax + 0.4 * span)
            else:
                ch = p1.copy()
            ch = ch + (rng.random(d) < pm) * rng.normal(0, 0.12, d) * (hi - lo)
            new.append(np.clip(ch, lo, hi))
        Xc = np.clip(np.array(new[:pop]), lo, hi)
        Fc = np.array(obj_batch(Xc)); nev += pop
        Xa = np.vstack([X, Xc]); Fa = np.vstack([F, Fc])
        fr = fronts_of(Fa); keep = []
        for ff in fr:
            if len(keep) + len(ff) <= pop:
                keep.extend(ff)
            else:
                cd = crowd_of(Fa, np.array(ff))
                keep.extend(list(np.array(ff)[np.argsort(-cd)][:pop - len(keep)]))
                break
        X, F = Xa[keep], Fa[keep]
    fr = fronts_of(F)
    return X, F, fr, nev
