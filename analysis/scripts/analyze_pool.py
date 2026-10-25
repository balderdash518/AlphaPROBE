import re, sys, collections
import numpy as np
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
last = text.rsplit("=== Logging at iteration", 1)[1]
alphas = re.findall(r"> Alpha #(\d+): ic=(-?[\d.]+), test_ic=(-?[\d.]+), test_icir=(-?[\d.]+), expr=(.+)", last)
ic = np.array([float(a[1]) for a in alphas]); tic = np.array([float(a[2]) for a in alphas]); tir = np.array([float(a[3]) for a in alphas])
exprs = [a[4].strip() for a in alphas]
print("final pool size", len(alphas))
print(f"train IC: min {ic.min():.4f} median {np.median(ic):.4f} max {ic.max():.4f}  (all positive? {bool((ic > 0).all())})")
print(f"test IC : median {np.median(tic):.4f} mean |test IC| {np.abs(tic).mean():.4f}; |test IC|>0.02: {(np.abs(tic) > 0.02).sum()}")
print(f"test |ICIR|>0.1: {(np.abs(tir) > 0.1).sum()}")
# sign of train IC is not printed (ic_ret is |IC|-like), so compare the stored train sign via 'ics' separately; here: share of near-zero test IC
print(f"test IC within +-0.005: {(np.abs(tic) <= 0.005).sum()}")
skel = collections.Counter(re.sub(r"\d+(\.\d+)?", "N", e) for e in exprs)
print("structural families in pool (windows/consts ignored):", len(skel), "| largest:", [(k[:70], v) for k, v in skel.most_common(4)])
feats = collections.Counter(f for e in exprs for f in set(re.findall(r"\$[a-z]+", e)))
print("pool factors using each feature:", dict(feats))
lens = [len(re.findall(r"[A-Za-z]+\(|\$[a-z]+", e)) for e in exprs]
print("pool expression size (ops+features): median", int(np.median(lens)), "max", max(lens))
order = np.argsort(-np.abs(tic))
print("top-5 by |test IC|:"); [print(f"  train {ic[i]:.4f} test {tic[i]:+.4f}  {exprs[i][:90]}") for i in order[:5]]
print("top-5 by train IC:"); [print(f"  train {ic[i]:.4f} test {tic[i]:+.4f}  {exprs[i][:90]}") for i in np.argsort(-ic)[:5]]
