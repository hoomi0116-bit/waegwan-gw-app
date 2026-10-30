"""관측 수위(칠곡왜관 2024)에 맞춰 모델 매개변수를 보정한다 (무작위 탐색)."""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from model import Params, run_model

RANGES = {
    "k": (0.9, 4.6),          # 관측정 시험 1.02 m/d ~ 문헌 최대 4.7 m/d
    "sy": (0.05, 0.20),
    "rech_coef": (0.10, 0.45),
    "drn_elev": (27.6, 28.8),
    "drn_cond": (0.5, 40.0),
    "et_max": (0.3, 1.5),
    "ext_depth": (2.5, 4.5),
    "pump_m3d": (0.0, 400.0),
}


def score(vals):
    p = Params(**vals)
    try:
        m = run_model(p)["metrics"]
        return vals, m
    except Exception as e:  # noqa
        return vals, {"rmse": 9e9, "me": 0, "r": 0, "nse": -9}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = int(os.environ.get("N", 96))
    samples = [{k: float(rng.uniform(*v)) for k, v in RANGES.items()} for _ in range(n)]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(score, samples))
    results.sort(key=lambda x: x[1]["rmse"])
    for v, m in results[:5]:
        print(round(m["rmse"], 3), round(m["r"], 3), round(m["nse"], 3), {k: round(x, 3) for k, x in v.items()})
    json.dump(results[0][0], open("calibrated.json", "w"), indent=2)
