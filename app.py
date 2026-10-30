"""
왜관 충적층 지하수 가뭄 시나리오 웹앱 (Streamlit + MODFLOW 6)
실행: streamlit run app.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from model import (Params, run_model, DELR, DELC, NCOL, NROW, OBS_COL, OBS_ROW,
                   RIVER_STAGE, GROUND_EL)
from verify_theis import run_theis

BASE = Path(__file__).resolve().parent
CAL = Params(**json.load(open(BASE / "calibrated.json")))

# 색 (dataviz 기준 팔레트 슬롯 1·2, 관측은 본문 잉크)
C_BASE = "#2a78d6"
C_SCEN = "#eb6834"
C_OBS = "#3a3936"
C_GRID = "rgba(128,128,128,0.18)"
WELL_LAT, WELL_LON = 35.97972, 128.42381

st.set_page_config(page_title="왜관 지하수 가뭄 시나리오", page_icon="💧", layout="wide")


# ------------------------------------------------------------------ 모델 실행 (캐시)
@st.cache_data(show_spinner=False, max_entries=32)
def simulate(p_items: tuple) -> dict:
    p = Params(**dict(p_items))
    r = run_model(p)
    return {"series": r["series"], "heads": r["heads"], "budget": r["budget"],
            "metrics": r["metrics"]}


@st.cache_data(show_spinner=False)
def theis_result() -> dict:
    return run_theis()


def key(p: Params) -> tuple:
    d = p.__dict__.copy()
    d["drought_years"] = tuple(d["drought_years"])
    return tuple(sorted(d.items()))


def kpis(res: dict, mask) -> dict:
    s = res["series"][mask]
    b = res["budget"][mask.values]
    return {
        "mean": float(s["sim"].mean()),
        "min": float(s["sim"].min()),
        "min_month": s.loc[s["sim"].idxmin(), "date"].strftime("%Y-%m"),
        "river": float((b["RIV_OUT"] - b["RIV_IN"]).mean()),   # 지하수 -> 낙동강 (m3/day)
        "et": float(b["EVTA_OUT"].mean()),
        "rch": float(b["RCHA_IN"].mean()),
    }


def style(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hovermode="x unified", font=dict(size=13))
    fig.update_xaxes(showgrid=False, linecolor=C_GRID)
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False)
    return fig


# ------------------------------------------------------------------ 사이드바: 모의조건
st.sidebar.title("모의조건 설정")
st.sidebar.caption("가뭄 시나리오: 2021~2025년 전체 기간의 월강수량에 배율을 곱합니다.")
rain_pct = st.sidebar.slider("강수 배율 (%)", 30, 150, 70, 5,
                             help="100% = 기상청 구미 관측 강수 그대로")
with st.sidebar.expander("고급 설정 (보정값이 기본)"):
    k = st.number_input("수리전도도 K (m/day)", 0.5, 10.0, round(CAL.k, 2), 0.1)
    sy = st.number_input("비산출률 Sy", 0.02, 0.35, round(CAL.sy, 3), 0.01)
    rc = st.number_input("함양률 (강수 대비)", 0.05, 0.6, round(CAL.rech_coef, 3), 0.01)
    pump = st.number_input("영농기(5~6월) 양수량 (m³/day)", 0.0, 500.0, round(CAL.pump_m3d, 1), 5.0)
    stage = st.number_input("낙동강(칠곡보) 수위 EL. m", 22.5, 28.0, RIVER_STAGE, 0.1)
label = st.sidebar.text_input("시나리오 이름", f"강수 {rain_pct}%")
run = st.sidebar.button("모델 실행", type="primary", width="stretch")

scen_p = replace(CAL, rain_factor=rain_pct / 100, k=k, sy=sy, rech_coef=rc,
                 pump_m3d=pump, river_stage=stage)

if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}
if run or not st.session_state.scenarios:
    with st.spinner("MODFLOW 6 실행 중..."):
        st.session_state.scenarios[label] = {"params": scen_p, "res": simulate(key(scen_p))}
    st.session_state.current = label

base = simulate(key(CAL))
cur_label = st.session_state.get("current", label)
cur = st.session_state.scenarios[cur_label]["res"]

if st.sidebar.button("비교 목록 비우기", width="stretch"):
    st.session_state.scenarios = {cur_label: st.session_state.scenarios[cur_label]}

# ------------------------------------------------------------------ 헤더 + 필터
st.title("왜관 낙동강 충적층 지하수 — 가뭄 시나리오")
st.caption("MODFLOW 6 (flopy) · 국가지하수관측망 칠곡왜관(충적, 82045) · 기상청 구미(279) · "
           "칠곡보 관리수위 EL. 25.5 m · 2021~2025 월 단위")

dates = base["series"]["date"]
months = [d.strftime("%Y-%m") for d in dates]
f1, f2 = st.columns([3, 1])
with f1:
    rng = st.select_slider("분석 기간", options=months, value=(months[0], months[-1]))
with f2:
    var = st.selectbox("표시 변수", ["지하수위", "기준 대비 수위 변화", "낙동강 유출량", "증발산량"])
mask = (dates >= pd.Timestamp(rng[0] + "-01")) & (dates <= pd.Timestamp(rng[1] + "-01"))

tab1, tab2, tab3, tab4 = st.tabs(["대시보드", "시나리오 비교", "모델 검증", "모델 정보"])

# ================================================================== 대시보드
with tab1:
    kb, kc = kpis(base, mask), kpis(cur, mask)
    st.subheader(f"{cur_label} vs 기준(실측 강수)")
    m = st.columns(5)
    m[0].metric("평균 지하수위", f"EL. {kc['mean']:.2f} m", f"{kc['mean']-kb['mean']:+.2f} m")
    m[1].metric("최저 월평균 수위", f"EL. {kc['min']:.2f} m", f"{kc['min']-kb['min']:+.2f} m")
    m[2].metric("최저 수위 시기", kc["min_month"])
    m[3].metric("낙동강 유출량", f"{kc['river']:,.0f} m³/일",
                f"{(kc['river']/kb['river']-1)*100:+.0f}%" if kb["river"] else None)
    m[4].metric("지표 아래 수심 (최저)", f"{GROUND_EL-kc['min']:.2f} m",
                f"{-(kc['min']-kb['min']):+.2f} m", delta_color="inverse")

    sb, sc = base["series"][mask], cur["series"][mask]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.25, 0.75],
                        vertical_spacing=0.05)
    fig.update_yaxes(title_text="강수 (mm)", row=1, col=1)
    fig.add_bar(x=sb["date"], y=sb["precip_mm"], name="실측 강수", marker_color="#9cc3ee",
                row=1, col=1)
    fig.add_bar(x=sc["date"], y=sc["precip_mm"], name=f"{cur_label} 강수",
                marker_color="#f4a582", row=1, col=1)
    if var == "지하수위":
        fig.add_scatter(x=sb["date"], y=sb["obs"], mode="markers", name="관측 (월평균)",
                        marker=dict(color=C_OBS, size=7), row=2, col=1)
        fig.add_scatter(x=sb["date"], y=sb["sim"], name="모의: 기준", line=dict(color=C_BASE, width=2),
                        row=2, col=1)
        fig.add_scatter(x=sc["date"], y=sc["sim"], name=f"모의: {cur_label}",
                        line=dict(color=C_SCEN, width=2), row=2, col=1)
        ytitle = "지하수위 (EL. m)"
    elif var == "기준 대비 수위 변화":
        fig.add_bar(x=sc["date"], y=sc["sim"].values - sb["sim"].values, name="수위 변화",
                    marker_color=C_SCEN, row=2, col=1)
        ytitle = "기준 대비 (m)"
    else:
        col = ("RIV_OUT", "RIV_IN") if var == "낙동강 유출량" else ("EVTA_OUT", None)
        bb, bc = base["budget"][mask.values], cur["budget"][mask.values]
        yb = bb[col[0]] - (bb[col[1]] if col[1] else 0)
        yc = bc[col[0]] - (bc[col[1]] if col[1] else 0)
        fig.add_scatter(x=sb["date"], y=yb, name="기준", line=dict(color=C_BASE, width=2), row=2, col=1)
        fig.add_scatter(x=sc["date"], y=yc, name=cur_label, line=dict(color=C_SCEN, width=2), row=2, col=1)
        ytitle = f"{var} (m³/일)"
    fig.update_yaxes(title_text=ytitle, row=2, col=1)
    fig.add_vrect(x0="2024-12-16", x1="2025-12-31", fillcolor="gray", opacity=0.07, line_width=0,
                  annotation_text="2025 예측 (관측 미공개)", annotation_position="top left",
                  row=2, col=1)
    st.plotly_chart(style(fig, 470), width="stretch")

    st.subheader("공간 분포")
    sel = st.select_slider("월 선택", options=months, value="2022-06" if "2022-06" in months else months[0])
    i = months.index(sel)
    x = (np.arange(NCOL) + 0.5) * DELR
    y = (np.arange(NROW) + 0.5) * DELC
    h1, h2 = st.columns([3, 2])
    with h1:
        mode = st.radio("지도 표시", ["기준 대비 수위 변화", "지하수위"], horizontal=True)
        st.markdown(f"**{cur_label} · {sel} 월말 {mode}**")
        if mode == "지하수위":
            z = cur["heads"][i]
            zmin = float(np.percentile(z, 5))
            hm = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale="Blues", zmin=zmin,
                                      zmax=float(z.max()), colorbar=dict(title="EL. m"),
                                      hovertemplate="낙동강에서 %{x:.0f} m<br>수위 EL. %{z:.2f} m<extra></extra>"))
        else:
            z = cur["heads"][i] - base["heads"][i]
            lim = max(float(np.abs(z).max()), 0.01)
            hm = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim,
                                      colorbar=dict(title="m"),
                                      hovertemplate="낙동강에서 %{x:.0f} m<br>변화 %{z:+.2f} m<extra></extra>"))
        hm.add_scatter(x=[x[OBS_COL]], y=[y[OBS_ROW]], mode="markers+text", text=["관측정"],
                       textposition="top center",
                       marker=dict(color="white", size=11, line=dict(color=C_OBS, width=2)),
                       showlegend=False, hoverinfo="skip")
        hm.update_xaxes(title="낙동강으로부터 거리 (m)")
        hm.update_yaxes(title="남북 (m)", scaleanchor=None)
        hm.update_layout(hovermode="closest")
        st.plotly_chart(style(hm, 340), width="stretch")
    with h2:
        st.markdown("&nbsp;")
        st.markdown(f"**관측정 통과 단면 · {sel}**")
        pf = go.Figure()
        pf.add_scatter(x=x, y=base["heads"][i][OBS_ROW], name="기준", line=dict(color=C_BASE, width=2))
        pf.add_scatter(x=x, y=cur["heads"][i][OBS_ROW], name=cur_label, line=dict(color=C_SCEN, width=2))
        pf.add_hline(y=RIVER_STAGE, line_dash="dot", line_color=C_OBS,
                     annotation_text="칠곡보 관리수위", annotation_position="bottom right")
        pf.update_xaxes(title="낙동강으로부터 거리 (m)")
        pf.update_yaxes(title="수위 (EL. m)")
        st.plotly_chart(style(pf, 340), width="stretch")

# ================================================================== 시나리오 비교
with tab2:
    st.subheader("저장된 시나리오")
    st.caption("사이드바에서 조건을 바꾸고 '모델 실행'을 누를 때마다 비교 목록에 추가됩니다.")
    rows = [{"시나리오": "기준 (실측 강수)", **kpis(base, mask)}]
    for name, v in st.session_state.scenarios.items():
        rows.append({"시나리오": name, **kpis(v["res"], mask)})
    tbl = pd.DataFrame(rows)
    tbl["평균 수위 변화 (m)"] = tbl["mean"] - tbl.loc[0, "mean"]
    show = tbl.rename(columns={"mean": "평균 수위 (EL. m)", "min": "최저 수위 (EL. m)",
                               "min_month": "최저 시기", "river": "낙동강 유출 (m³/일)",
                               "et": "증발산 (m³/일)", "rch": "함양 (m³/일)"})
    st.dataframe(show.round(3), width="stretch", hide_index=True)

    cmp = go.Figure()
    cmp.add_scatter(x=base["series"]["date"][mask], y=base["series"]["sim"][mask],
                    name="기준 (실측 강수)", line=dict(color=C_OBS, width=2, dash="dot"))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300", "#eda100", "#e34948"]
    for j, (name, v) in enumerate(st.session_state.scenarios.items()):
        s = v["res"]["series"][mask]
        cmp.add_scatter(x=s["date"], y=s["sim"], name=name,
                        line=dict(color=palette[j % len(palette)], width=2))
    cmp.update_yaxes(title="모의 지하수위 (EL. m)")
    st.plotly_chart(style(cmp, 420), width="stretch")

    st.subheader("강수 배율별 민감도")
    if st.button("배율 50~120% 일괄 실행"):
        sens = []
        prog = st.progress(0.0)
        fs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        for n_, f in enumerate(fs):
            r = simulate(key(replace(CAL, rain_factor=f)))
            kk = kpis(r, mask)
            sens.append({"강수 배율 (%)": int(f * 100), "평균 수위 (EL. m)": kk["mean"],
                         "최저 수위 (EL. m)": kk["min"], "낙동강 유출 (m³/일)": kk["river"]})
            prog.progress((n_ + 1) / len(fs))
        sd = pd.DataFrame(sens)
        sf = go.Figure()
        sf.add_scatter(x=sd["강수 배율 (%)"], y=sd["평균 수위 (EL. m)"], mode="lines+markers",
                       name="평균 수위", line=dict(color=C_BASE, width=2), marker=dict(size=8))
        sf.add_scatter(x=sd["강수 배율 (%)"], y=sd["최저 수위 (EL. m)"], mode="lines+markers",
                       name="최저 수위", line=dict(color=C_SCEN, width=2), marker=dict(size=8))
        sf.update_xaxes(title="강수 배율 (%)")
        sf.update_yaxes(title="지하수위 (EL. m)")
        st.plotly_chart(style(sf, 340), width="stretch")
        st.dataframe(sd.round(3), width="stretch", hide_index=True)

# ================================================================== 검증
with tab3:
    mt = base["metrics"]
    st.subheader("① 현장 재현성: 관측 vs 모의 (2021~2024 월평균, 48개월)")
    c = st.columns(4)
    c[0].metric("RMSE", f"{mt['rmse']:.3f} m")
    c[1].metric("평균오차", f"{mt['me']:+.3f} m")
    c[2].metric("상관계수 r", f"{mt['r']:.2f}")
    c[3].metric("NSE", f"{mt['nse']:.2f}")
    v = base["series"].dropna(subset=["obs"])
    a1, a2 = st.columns([2, 1])
    with a1:
        vf = go.Figure()
        vf.add_scatter(x=v["date"], y=v["obs_max"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        vf.add_scatter(x=v["date"], y=v["obs_min"], fill="tonexty", fillcolor="rgba(58,57,54,0.12)",
                       line=dict(width=0), name="관측 월 최저~최고")
        vf.add_scatter(x=v["date"], y=v["obs"], mode="markers", name="관측 월평균",
                       marker=dict(color=C_OBS, size=7))
        vf.add_scatter(x=v["date"], y=v["sim"], name="모의", line=dict(color=C_BASE, width=2))
        vf.update_yaxes(title="지하수위 (EL. m)")
        st.plotly_chart(style(vf, 340), width="stretch")
    with a2:
        lo, hi = float(v[["obs", "sim"]].min().min()) - 0.05, float(v[["obs", "sim"]].max().max()) + 0.05
        sc_ = go.Figure()
        sc_.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=C_GRID, dash="dot"),
                        name="1:1", hoverinfo="skip")
        sc_.add_scatter(x=v["obs"], y=v["sim"], mode="markers", name="월",
                        marker=dict(color=C_BASE, size=8), text=v["date"].dt.strftime("%Y-%m"),
                        hovertemplate="%{text}<br>관측 %{x:.2f}<br>모의 %{y:.2f}<extra></extra>")
        sc_.update_xaxes(title="관측 (EL. m)", range=[lo, hi])
        sc_.update_yaxes(title="모의 (EL. m)", range=[lo, hi])
        sc_.update_layout(hovermode="closest", showlegend=False)
        st.plotly_chart(style(sc_, 340), width="stretch")

    st.subheader("② 계산 정확성: Theis 해석해 비교")
    th = theis_result()
    st.caption(f"피압대수층 단일 양수정, T = {th['inputs']['T']} m²/일 (칠곡왜관 제원), "
               f"S = {th['inputs']['S']} (왜관 문헌), Q = {th['inputs']['Q']:.0f} m³/일, "
               f"양수 {th['inputs']['t_end']:.0f}일. 같은 MODFLOW 6 엔진으로 계산.")
    c = st.columns(3)
    c[0].metric("RMSE", f"{th['metrics']['rmse_m']*100:.1f} cm")
    c[1].metric("최대 오차", f"{th['metrics']['max_abs_m']*100:.1f} cm")
    c[2].metric("평균 상대오차", f"{th['metrics']['rel_err_pct']:.1f} %")
    pr = th["profile"]
    tf = go.Figure()
    tf.add_scatter(x=pr["r_m"], y=pr["theis"], name="Theis 해석해", line=dict(color=C_OBS, width=2))
    tf.add_scatter(x=pr["r_m"], y=pr["modflow"], mode="markers", name="MODFLOW 6",
                   marker=dict(color=C_BASE, size=8))
    tf.update_xaxes(title="양수정으로부터 거리 (m, 로그)", type="log")
    tf.update_yaxes(title="수위강하 (m)")
    tf.update_layout(hovermode="closest")
    st.plotly_chart(style(tf, 340), width="stretch")

# ================================================================== 모델 정보
with tab4:
    i1, i2 = st.columns([3, 2])
    with i1:
        dlon = DELR * NCOL / (111320 * np.cos(np.radians(WELL_LAT)))
        lon0 = WELL_LON - OBS_COL * DELR / (111320 * np.cos(np.radians(WELL_LAT)))
        dlat = DELC * NROW / 2 / 111320
        mp = go.Figure()
        mp.add_trace(go.Scattermap(lon=[lon0, lon0 + dlon, lon0 + dlon, lon0, lon0],
                                   lat=[WELL_LAT - dlat, WELL_LAT - dlat, WELL_LAT + dlat,
                                        WELL_LAT + dlat, WELL_LAT - dlat],
                                   mode="lines", line=dict(color=C_BASE, width=3), name="모델 영역 (3 km × 1 km)"))
        mp.add_trace(go.Scattermap(lon=[lon0, lon0], lat=[WELL_LAT - dlat, WELL_LAT + dlat], mode="lines",
                                   line=dict(color=C_SCEN, width=4), name="낙동강 경계 (EL. 25.5 m)"))
        mp.add_trace(go.Scattermap(lon=[WELL_LON], lat=[WELL_LAT], mode="markers",
                                   marker=dict(size=13, color=C_OBS), name="관측정 칠곡왜관(82045)"))
        mp.update_layout(map=dict(style="open-street-map", center=dict(lat=WELL_LAT, lon=WELL_LON - 0.006),
                                  zoom=12.3), height=430, margin=dict(l=0, r=0, t=0, b=0),
                         legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(mp, width="stretch")
        st.caption("모델 영역은 개념적 단면 띠로, 낙동강 위치는 관측정 서쪽 약 2.2 km로 근사했다.")
    with i2:
        st.markdown(f"""
**모델 구성**
- 엔진: MODFLOW 6 (flopy로 입력 생성·실행)
- 격자: 1층, {NROW}×{NCOL} 셀 (50 m × 100 m), 자유면 대수층
- 표고: 지표 EL. 31.25 m, 바닥 EL. 21.25 m (관측정 지층구성)
- 기간: 2021-01 ~ 2025-12, 월 응력기간 60개 (월 3 시간단계)
- 경계: 낙동강 RIV (칠곡보 관리수위), 배수로 DRN, 함양 RCH, 증발산 EVT, 영농기 양수 WEL

**보정값**
- K {CAL.k:.2f} m/일 · Sy {CAL.sy:.2f} · 함양률 {CAL.rech_coef:.2f}
- 배수로 EL. {CAL.drn_elev:.1f} m · 소멸심도 {CAL.ext_depth:.1f} m · 양수 {CAL.pump_m3d:.0f} m³/일

**자료 출처**
- 지하수위: [국가지하수정보센터](https://www.gims.go.kr) 칠곡왜관(충적) 2021~2024
- 강수·기온: [기상청 날씨누리](https://www.weather.go.kr) 구미(279) 2021~2025
- 관측정 제원·수리상수: GIMS 국가관리측정망 제원
- 칠곡보 관리수위 EL. 25.5 m
""")
