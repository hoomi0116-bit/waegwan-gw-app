"""
Theis 해석해 검증: 같은 MODFLOW 6 엔진으로 피압대수층 단일 양수정 문제를 풀어
해석해와 비교한다. 수리상수는 칠곡왜관 관측정 제원(투수량계수 7.15 m2/day)과
왜관 문헌 저류계수(1.84e-4)를 사용한다.
"""
from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pandas as pd
import flopy
from scipy.special import exp1

from model import ensure_mf6

T = 7.15        # m2/day (GIMS 칠곡왜관 제원)
S = 1.84e-4     # 문헌 (왜관 죽전리)
B = 10.0        # 대수층 두께 (m)
Q = 50.0        # 양수량 (m3/day)
TIME = 2.0      # 양수 시간 (day)
OBS_R = [10.0, 20.0, 40.0, 80.0, 160.0]


def theis(r, t, q=Q, tr=T, s=S):
    u = r ** 2 * s / (4 * tr * t)
    return q / (4 * np.pi * tr) * exp1(u)


def run_theis(q: float = Q, t_end: float = TIME) -> dict:
    exe = ensure_mf6()
    ws = tempfile.mkdtemp(prefix="theis_")
    # 중심이 세밀한 정사각 격자 (5 m -> 바깥 확장), 영역 반폭 약 3 km
    inner = np.full(20, 5.0)
    outer = 5.0 * 1.25 ** np.arange(1, 22)
    half = np.concatenate([inner, outer])
    widths = np.concatenate([half[::-1], [5.0], half])
    n = len(widths)
    c = n // 2
    centers = np.cumsum(widths) - widths / 2
    xc = centers - centers[c]

    sim = flopy.mf6.MFSimulation(sim_name="theis", sim_ws=ws, exe_name=exe)
    nstp = 40
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(t_end, nstp, 1.2)], time_units="days")
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", inner_dvclose=1e-8, outer_dvclose=1e-7)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="theis")
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=n, ncol=n, delr=widths, delc=widths,
                            top=0.0, botm=-B)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=T / B)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=S / B, transient={0: True})
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={0: [((0, c, c), -q)]})
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="theis.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("Theis 검증 모델 실행 실패")
    hobj = gwf.output.head()
    times = np.array(hobj.get_times())
    allh = hobj.get_alldata()[:, 0]
    shutil.rmtree(ws, ignore_errors=True)

    # 동쪽 방향 셀 중심에서 관측 거리와 가장 가까운 셀 사용
    rows = []
    for r in OBS_R:
        j = c + int(np.argmin(np.abs(xc[c:] - r)))
        rr = xc[j]
        for ti, tt in enumerate(times):
            rows.append({"r_m": r, "r_cell_m": rr, "t_day": tt,
                         "modflow": -allh[ti, c, j], "theis": theis(rr, tt, q)})
    ts = pd.DataFrame(rows)
    # 거리-수위강하 곡선 (최종 시간)
    prof = pd.DataFrame({"r_m": xc[c + 1:], "modflow": -allh[-1, c, c + 1:],
                         "theis": theis(xc[c + 1:], times[-1], q)})
    prof = prof[prof["r_m"] <= 1000]
    last = ts[ts["t_day"] == times[-1]]
    err = last["modflow"] - last["theis"]
    metrics = {"rmse_m": float(np.sqrt((err ** 2).mean())),
               "max_abs_m": float(err.abs().max()),
               "rel_err_pct": float((err.abs() / last["theis"]).mean() * 100)}
    return {"timeseries": ts, "profile": prof, "metrics": metrics,
            "inputs": {"T": T, "S": S, "Q": q, "t_end": t_end}}


if __name__ == "__main__":
    r = run_theis()
    print(r["metrics"])
    print(r["timeseries"][r["timeseries"]["t_day"] == r["timeseries"]["t_day"].max()])
