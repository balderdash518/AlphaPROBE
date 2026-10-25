import re, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from alphagen.data.dimension import DimensionCalculator, _is_close
import numpy as np
text = open(sys.argv[1], encoding="utf-8", errors="replace").read().rsplit("=== Logging at iteration", 1)[1]
alphas = re.findall(r"> Alpha #(\d+): ic=(-?[\d.]+), test_ic=(-?[\d.]+), test_icir=(-?[\d.]+), expr=(.+)", text)
calc = DimensionCalculator()
fn = lambda e: calc.dimension(e).dimension
dimless, other = [], []
for a in alphas:
    e = a[4].strip()
    try:
        d = fn(e)
    except Exception as exc:
        d = f"err:{exc}"
    ok = (not isinstance(d, str)) and _is_close(d, 0)
    (dimless if ok else other).append((float(a[1]), float(a[2]), e, d))
print("dimensionless:", len(dimless), "| not dimensionless:", len(other))
for grp, name in ((dimless, "dimensionless"), (other, "NOT dimensionless")):
    if grp:
        t = np.array([g[1] for g in grp]); tr = np.array([g[0] for g in grp])
        print(f"  {name}: mean train IC {tr.mean():.4f}, mean |test IC| {np.abs(t).mean():.4f}")
print("examples not dimensionless:", [(g[2][:55], g[3]) for g in other[:6]])
