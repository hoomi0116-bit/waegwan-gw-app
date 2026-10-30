"""
왜관 낙동강 충적층 지하수 모델 (MODFLOW 6 + flopy)

- 대상: 국가지하수관측망 칠곡왜관(충적, 관정번호 82045)
- 기간: 2021-01 ~ 2025-12, 월 단위 응력기간 60개 (첫 응력기간은 정상류 초기화)
- 경계: 서측 낙동강(칠곡보 관리수위 EL. 25.5 m, RIV), 배수로망(DRN),
        강수 함양(RCH = 함양률 x 월강수량), 지하수 증발산(EVT), 영농기 양수(WEL, 선택)
- 관측 수위는 2021~2024년만 공개되어 있어 2025년은 예측 구간으로 다룬다.
"""
from __future__ import annotations

import shutil
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import flopy

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MF6_URL = "https://github.com/MODFLOW-ORG/executables/releases/download/29.0/linux.zip"

# ---------------------------------------------------------------- 고정 제원
GROUND_EL = 31.25      # 관측정 표고 (GIMS 제원)
BOTTOM_EL = 21.25      # 충적층+풍화대 하부 (심도 10 m, GIMS 지층구성)
RIVER_STAGE = 25.5     # 칠곡보 관리수위 (EL. m)
NCOL, NROW = 60, 10
DELR, DELC = 50.0, 100.0          # 동서 3 km x 남북 1 km
OBS_DIST = 2200.0                 # 관측정 ~ 낙동강 거리 (m)
OBS_COL = int(OBS_DIST // DELR)
OBS_ROW = NROW // 2
NSTP = 3                          # 월당 시간단계 수
FARM_MONTHS = (5, 6)              # 영농기 양수 월


@dataclass
class Params:
    k: float = 2.0            # 수평 수리전도도 (m/day)
    sy: float = 0.10          # 비산출률
    ss: float = 1e-5          # 비저류계수 (1/m)
    rech_coef: float = 0.25   # 함양률 (강수 대비)
    rain_factor: float = 1.0  # 가뭄 시나리오: 강수 배율 (1.0 = 실측)
    drought_years: tuple = () # 강수 배율을 적용할 연도 (빈 값이면 전체 기간)
    drn_elev: float = 28.4    # 배수로망 바닥 표고 (EL. m)
    drn_cond: float = 10.0    # 배수로 컨덕턴스 (m2/day, 셀당)
    et_max: float = 1.0       # 잠재증발산 배율 (기온 기반 PET에 곱함)
    ext_depth: float = 3.5    # 증발산 소멸심도 (m)
    river_stage: float = RIVER_STAGE
    pump_m3d: float = 0.0     # 영농기(5~6월) 양수량 (m3/day, 관측정 위치)


def ensure_mf6() -> str:
    """mf6 실행 파일 경로 반환. 없으면 공식 릴리스에서 내려받는다."""
    local = BASE / "bin" / "mf6"
    if local.exists():
        return str(local)
    found = shutil.which("mf6")
    if found:
        return found
    local.parent.mkdir(parents=True, exist_ok=True)
    zpath = local.parent / "linux.zip"
    urllib.request.urlretrieve(MF6_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        with z.open("mf6") as src, open(local, "wb") as dst:
            dst.write(src.read())
    local.chmod(local.stat().st_mode | stat.S_IEXEC)
    zpath.unlink(missing_ok=True)
    return str(local)


def load_forcing() -> pd.DataFrame:
    rain = pd.read_csv(DATA / "rain_kma_gumi279_monthly_2021_2025.csv")
    obs = pd.read_csv(DATA / "gw_82045_monthly_2021_2024.csv")
    df = rain.merge(obs[["month", "elev_mean_m", "elev_min_m", "elev_max_m"]],
                    on="month", how="left")
    df["date"] = pd.to_datetime(df["month"] + "-01")
    df["year"] = df["date"].dt.year
    df["mon"] = df["date"].dt.month
    # 기온 기반 잠재증발산 (단순식, mm/day): 0.16 x 월평균기온, 영하는 0
    df["pet_mm_d"] = np.clip(0.16 * df["tavg_c"], 0, None)
    return df


def run_model(p: Params, workdir: str | None = None) -> dict:
    df = load_forcing()
    nper = len(df)
    exe = ensure_mf6()
    ws = workdir or tempfile.mkdtemp(prefix="mf6_")
    name = "waegwan"

    # 강수 배율 (가뭄 시나리오)
    factor = np.ones(nper)
    if p.rain_factor != 1.0:
        mask = df["year"].isin(p.drought_years) if p.drought_years else np.ones(nper, bool)
        factor[np.asarray(mask)] = p.rain_factor
    precip = df["precip_mm"].to_numpy() * factor
    ndays = df["ndays"].to_numpy().astype(float)

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name=exe)
    perioddata = [(1.0, 1, 1.0)] + [(d, NSTP, 1.0) for d in ndays]
    flopy.mf6.ModflowTdis(sim, nper=nper + 1, perioddata=perioddata, time_units="days")
    flopy.mf6.ModflowIms(sim, complexity="MODERATE", outer_dvclose=1e-4,
                         inner_dvclose=1e-5, linear_acceleration="BICGSTAB")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, newtonoptions="UNDER_RELAXATION")
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=NROW, ncol=NCOL, delr=DELR, delc=DELC,
                            top=GROUND_EL, botm=BOTTOM_EL)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=p.k)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, sy=p.sy, ss=p.ss,
                            steady_state={0: True}, transient={1: True})
    flopy.mf6.ModflowGwfic(gwf, strt=28.0)

    riv = [((0, r, 0), p.river_stage, 500.0, 22.0) for r in range(NROW)]
    flopy.mf6.ModflowGwfriv(gwf, stress_period_data={0: riv}, save_flows=True)
    drn = [((0, r, c), p.drn_elev, p.drn_cond) for r in range(NROW) for c in range(1, NCOL)]
    flopy.mf6.ModflowGwfdrn(gwf, stress_period_data={0: drn}, save_flows=True)

    rch_rate = precip / 1000.0 / ndays * p.rech_coef        # m/day
    pet = df["pet_mm_d"].to_numpy() / 1000.0 * p.et_max      # m/day
    obs_years = (df["year"] <= 2024).to_numpy()
    rch_spd = {0: float(rch_rate[obs_years].mean())}
    evt_spd = {0: float(pet[obs_years].mean())}
    for i in range(nper):
        rch_spd[i + 1] = float(rch_rate[i])
        evt_spd[i + 1] = float(pet[i])
    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch_spd, save_flows=True)
    flopy.mf6.ModflowGwfevta(gwf, surface=GROUND_EL, rate=evt_spd, depth=p.ext_depth,
                             save_flows=True)

    if p.pump_m3d > 0:
        wel_spd = {0: [((0, OBS_ROW, OBS_COL), 0.0)]}
        for i, mon in enumerate(df["mon"]):
            q = -p.pump_m3d if mon in FARM_MONTHS else 0.0
            wel_spd[i + 1] = [((0, OBS_ROW, OBS_COL), q)]
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd, save_flows=True)

    flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{name}.hds", budget_filerecord=f"{name}.cbc",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "LAST")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError("MODFLOW 6 실행 실패")

    allh = gwf.output.head().get_alldata()[:, 0]          # (1 + nper*NSTP, nrow, ncol)
    steps = allh[1:].reshape(nper, NSTP, NROW, NCOL)
    heads_end = steps[:, -1]                              # 월말 수두
    obs_sim = steps[:, :, OBS_ROW, OBS_COL].mean(axis=1)  # 월평균 (관측 월평균과 비교)

    # 월별 물수지 (m3/day, 월말 시간단계)
    budget = gwf.output.list().get_dataframes(diff=False)[0].iloc[1:].reset_index(drop=True)
    if len(budget) > nper:
        budget = budget.iloc[NSTP - 1::NSTP].reset_index(drop=True)
    budget["date"] = df["date"].values[: len(budget)]

    out = pd.DataFrame({"date": df["date"], "precip_mm": precip,
                        "obs": df["elev_mean_m"], "obs_min": df["elev_min_m"],
                        "obs_max": df["elev_max_m"], "sim": obs_sim})
    valid = out.dropna(subset=["obs"])
    resid = valid["sim"] - valid["obs"]
    metrics = {
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "me": float(resid.mean()),
        "r": float(np.corrcoef(valid["sim"], valid["obs"])[0, 1]),
        "nse": float(1 - (resid ** 2).sum() / ((valid["obs"] - valid["obs"].mean()) ** 2).sum()),
    }
    if workdir is None:
        shutil.rmtree(ws, ignore_errors=True)
    return {"series": out, "heads": heads_end, "budget": budget, "metrics": metrics,
            "params": asdict(p)}


if __name__ == "__main__":
    import time
    t = time.time()
    r = run_model(Params())
    print(r["metrics"], f"{time.time()-t:.1f}s")
    print(r["series"].tail(14).to_string())
    print(r["budget"].columns.tolist(), len(r["budget"]))
