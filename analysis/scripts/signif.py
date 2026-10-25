import json, sys, numpy as np, re
F = json.load(open(sys.argv[1], encoding='utf-8'))['factors']
n_days = int(sys.argv[2]); n_eff = n_days / 20
t = np.array([abs(f['train_icir']) * np.sqrt(n_eff) for f in F])
keep = np.array([np.sign(f['train_ic']) == np.sign(f['test_ic']) for f in F])
price = np.array([(not f['dimensionless']) or bool(re.search(r'(?<!Get)(Greater|Less)\(', f['expr'])) for f in F])
print(f"train-period |t| (overlap-adjusted, n_eff={n_eff:.0f}): median {np.median(t):.2f}; t<2: {(t < 2).sum()}/{len(F)}; t>=3: {(t >= 3).sum()}")
for lo, hi in ((0, 2), (2, 3), (3, 99)):
    m = (t >= lo) & (t < hi)
    if m.any(): print(f"  t in [{lo},{hi}): {m.sum():>2} factors, direction kept OOS {keep[m].sum()}/{m.sum()}, of which price-level {price[m].sum()}")
m = ~price
print(f"  dimensionless only: t>=2 {((t >= 2) & m).sum()}, kept among those {(keep & (t >= 2) & m).sum()}")
dup = len(F) - len({f['expr'] for f in F}); print('exact duplicate expressions in pool:', dup)
