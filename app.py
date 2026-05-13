import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="서울시 치안 데이터 대시보드", layout="wide")

def set_korean_font():
    font_list = [f.name for f in fm.fontManager.ttflist]
    if   "Malgun Gothic"     in font_list: plt.rcParams["font.family"] = "Malgun Gothic"
    elif "AppleGothic"       in font_list: plt.rcParams["font.family"] = "AppleGothic"
    elif "NanumGothic"       in font_list: plt.rcParams["font.family"] = "NanumGothic"
    elif "NanumBarunGothic"  in font_list: plt.rcParams["font.family"] = "NanumBarunGothic"
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# 데이터 경로 (DATA 폴더 기준)
# CCTV_FILE  = "./DATA/서울시 자치구 (범죄예방 수사용) CCTV 설치현황(25.12.31 기준).xlsx"
# CRIME_CSV  = "./DATA/5대_범죄_발생현황_20260512140024.csv"
# POP_FILE   = "./DATA/등록인구_20260512150716.xlsx"
# LOC_FILE   = "./DATA/5대_범죄_발생장소별_현황.xlsx"
CCTV_FILE  = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\서울시 자치구 (범죄예방 수사용) CCTV 설치현황(25.12.31 기준).xlsx"
CRIME_CSV  = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\5대+범죄+발생현황_20260512140024.csv"
POP_FILE   = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\등록인구_20260512150716.xlsx"
LOC_FILE   = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\5대+범죄+발생장소별+현황.xlsx"


# ════════════════════════════════════════════════════════════════════════════════
# 데이터 로딩 (캐싱)
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_cctv():
    raw = pd.read_excel(CCTV_FILE, header=2)
    # 데이터 전처리
    raw = raw.drop(columns=["Unnamed: 0"], errors="ignore")
    raw = raw[raw["구분"] != "계"].copy()
    raw = raw.iloc[:-2]

    # 년도 칼럼 선택
    year_cols = [c for c in raw.columns if str(c).endswith("년")]
    
    # 숫자 형태 문자열 -> 숫자 변환 / 다른 문자열 -> 결측치 -> 0
    raw[year_cols] = raw[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return raw.set_index("순번")


@st.cache_data
def load_crime_csv():
    return pd.read_csv(CRIME_CSV, encoding="utf-8")


@st.cache_data
def load_population():
    return pd.read_excel(POP_FILE)


@st.cache_data
def load_crime_loc():
    return pd.read_excel(LOC_FILE)


@st.cache_data
def build_df_final():
    """2024년 기준 25개 자치구 통합 데이터프레임."""
    cctv      = load_cctv()
    crime_raw = load_crime_csv()
    pop_raw   = load_population()

    # ── 인구 (2024.1 = 총인구 / 2024 = 세대수로 오류 방지) ───────────────────
    pop_24 = pop_raw[["동별(2)", "2024.1"]].copy()
    pop_24.columns = ["자치구", "인구수"]
    pop_24["자치구"] = pop_24["자치구"].astype(str).str.strip()
    pop_24 = pop_24[~pop_24["자치구"].isin(["소계", "합계", "동별(2)"])]
    pop_24["인구수"] = pd.to_numeric(
        pop_24["인구수"].astype(str).str.replace(",", ""), errors="coerce"
    )
    pop_24 = pop_24.dropna(subset=["인구수"])
    pop_24 = pop_24[pop_24["인구수"] > 50_000]          # 동 단위 행 제거
    pop_24["인구수"] = pop_24["인구수"].astype(int)

    # ── CCTV (2024년) ─────────────────────────────────────────────────────────
    cctv_24 = cctv[["구분", "2024년"]].rename(
        columns={"구분": "자치구", "2024년": "CCTV수"}
    ).copy()
    cctv_24["CCTV수"] = pd.to_numeric(cctv_24["CCTV수"], errors="coerce")

    # ── 범죄 (2024년 소계 발생건수, 구별) ─────────────────────────────────────
    crime_filt = crime_raw[[c for c in crime_raw.columns if "." not in str(c)]].copy()
    crime_filt = crime_filt.drop(0).rename(columns={"자치구별(2)": "자치구"})
    if "자치구별(1)" in crime_filt.columns:
        crime_filt = crime_filt.drop(columns=["자치구별(1)"])
    crime_24 = crime_filt[["자치구", "2024"]].rename(columns={"2024": "발생건수"}).copy()
    crime_24["발생건수"] = pd.to_numeric(crime_24["발생건수"], errors="coerce")

    # ── 병합 ─────────────────────────────────────────────────────────────────
    df = pd.merge(pop_24, cctv_24, on="자치구")
    df = pd.merge(df, crime_24, on="자치구")
    df = df.dropna()

    # ── 지표 계산 ─────────────────────────────────────────────────────────────
    df["범죄율"]        = (df["발생건수"] / df["인구수"]) * 100_000   # 인구 10만명당
    df["CCTV_밀도"]     = (df["CCTV수"]   / df["인구수"]) * 10_000    # 인구 1만명당
    df["CCTV당_범죄수"] = df["발생건수"]  / df["CCTV수"]
    df.index = range(1, len(df) + 1)
    return df


@st.cache_data
def build_crime_trend():
    """2015–2024 서울 전체 연도별 CCTV·인구·5대 범죄 트렌드."""
    cctv      = load_cctv()
    crime_raw = load_crime_csv()
    pop_raw   = load_population()

    CRIME_TYPES   = ["살인", "강도", "강간·강제추행", "절도", "폭력"]
    CRIME_OFFSETS = [2, 4, 6, 8, 10]   # 발생 컬럼 오프셋

    rows = []
    for year in range(2015, 2025):
        yr = str(year)

        # 인구 (서울 전체 = 동별(2)=='소계', 총인구={yr}.1)
        try:
            pop_total = pd.to_numeric(
                str(pop_raw.loc[pop_raw["동별(2)"] == "소계", f"{yr}.1"].values[0])
                .replace(",", ""), errors="coerce"
            )
        except Exception:
            continue

        # CCTV 총합
        cctv_col = f"{yr}년"
        cctv_sum = cctv[cctv_col].sum() if cctv_col in cctv.columns else np.nan

        # 합계·소계 행
        row_sum = crime_raw[
            (crime_raw["자치구별(1)"] == "합계") &
            (crime_raw["자치구별(2)"] == "소계")
        ]
        if row_sum.empty:
            row_sum = crime_raw[crime_raw["자치구별(1)"] == "합계"].iloc[[0]]

        total_crime = pd.to_numeric(
            str(row_sum[yr].values[0]).replace(",", ""), errors="coerce"
        )
        crime_rate_total = (total_crime / pop_total) * 100_000

        type_rates = []
        for offset in CRIME_OFFSETS:
            col = f"{yr}.{offset}"
            val = pd.to_numeric(
                str(row_sum[col].values[0]).replace(",", ""), errors="coerce"
            )
            type_rates.append((val / pop_total) * 100_000)

        rows.append([yr, cctv_sum, pop_total, total_crime, crime_rate_total] + type_rates)

    df = pd.DataFrame(
        rows,
        columns=["연도", "CCTV수량", "인구수", "총범죄수", "총범죄율"] + CRIME_TYPES
    )
    df["연도"] = df["연도"].astype(str)
    return df


# ── 데이터 불러오기 ───────────────────────────────────────────────────────────
df_final  = build_df_final()
df_trend  = build_crime_trend()
cctv_raw  = load_cctv()

# ════════════════════════════════════════════════════════════════════════════════
# 제목
# ════════════════════════════════════════════════════════════════════════════════
st.title("🛡️ 서울시 치안 인프라 및 범죄 현황 분석 대시보드")
st.markdown(
    "서울시 25개 자치구의 **CCTV 설치 현황**, **5대 범죄 발생 통계**, **인구 데이터**를 "
    "종합 분석합니다.  *(데이터 기준: 2015 ~ 2024)*"
)
st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# 섹션 A: CCTV 연도별 설치 현황
# ════════════════════════════════════════════════════════════════════════════════
st.header("📷 서울시 CCTV 연도별 설치 현황 (2015–2025)")

year_cols   = [c for c in cctv_raw.columns if str(c).endswith("년")]
cctv_years  = cctv_raw[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

tab_a1, tab_a2, tab_a3 = st.tabs(
    ["📊 연도별 누적 설치 총량", "📈 연도별 신규 설치 추이", "🗂️ 자치구별 2024년 CCTV"]
)

with tab_a1:
    total_cum = cctv_years.sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(total_cum.index, total_cum.values,
                  color=sns.color_palette("Blues_d", len(total_cum)))
    ax.set_title("서울시 연도별 CCTV 누적 설치 대수 (전체 합산)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("누적 설치 대수 (대)")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, total_cum.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{int(val):,}", ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

with tab_a2:
    cctv_diff = cctv_years.diff(axis=1).iloc[:, 1:]
    total_inc = cctv_diff.sum()
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.fill_between(range(len(total_inc)), total_inc.values, alpha=0.2, color="forestgreen")
    ax2.plot(total_inc.index, total_inc.values, marker="o", linewidth=3,
             color="forestgreen", markersize=8)
    ax2.set_title("서울시 연도별 신규 CCTV 설치 추이", fontsize=14, fontweight="bold")
    ax2.set_ylabel("신규 설치 대수 (대)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.tick_params(axis="x", rotation=30)
    for i, val in enumerate(total_inc.values):
        ax2.text(i, val + max(total_inc)*0.03, f"{int(val):,}", ha="center",
                 fontsize=8, fontweight="bold", color="darkgreen")
    plt.tight_layout()
    st.pyplot(fig2)

with tab_a3:
    df_cctv_s = df_final.sort_values("CCTV수", ascending=True)
    mean_cctv = df_final["CCTV수"].mean()
    colors_c  = ["#d73027" if v > mean_cctv else "#4575b4" for v in df_cctv_s["CCTV수"]]
    fig3, ax3 = plt.subplots(figsize=(9, 9))
    b3 = ax3.barh(df_cctv_s["자치구"], df_cctv_s["CCTV수"], color=colors_c)
    ax3.axvline(mean_cctv, color="orange", linestyle="--", linewidth=2,
                label=f"평균 {mean_cctv:.0f}대")
    ax3.set_title("자치구별 CCTV 누적 설치 대수 (2024년)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("CCTV 설치 대수 (대)")
    ax3.legend()
    for bar, val in zip(b3, df_cctv_s["CCTV수"]):
        ax3.text(val + 50, bar.get_y() + bar.get_height()/2,
                 f"{int(val):,}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    st.caption("🔴 평균 초과 / 🔵 평균 이하")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 B: 인구 현황
# ════════════════════════════════════════════════════════════════════════════════
st.header("👥 서울시 자치구별 등록인구 현황 (2024년)")

pop_s = df_final.sort_values("인구수", ascending=False)
fig_pop, ax_pop = plt.subplots(figsize=(14, 5))
sns.barplot(x="자치구", y="인구수", data=pop_s, palette="coolwarm", ax=ax_pop)
ax_pop.set_title("2024년 서울시 자치구별 등록인구 (내림차순)",
                 fontsize=14, fontweight="bold")
ax_pop.set_ylabel("인구수 (명)")
ax_pop.tick_params(axis="x", rotation=45)
for i, val in enumerate(pop_s["인구수"]):
    ax_pop.text(i, val + 3_000, f"{val/10000:.1f}만", ha="center",
                fontsize=8, fontweight="bold")
plt.tight_layout()
st.pyplot(fig_pop)
st.divider()

# ── 데이터 요약표 ─────────────────────────────────────────────────────────────
with st.expander("🔍 2024년 자치구별 종합 데이터 요약 보기", expanded=False):
    disp = df_final[
        ["자치구", "인구수", "CCTV수", "발생건수", "범죄율", "CCTV_밀도", "CCTV당_범죄수"]
    ].rename(columns={
        "인구수": "인구수",
        "CCTV수": "CCTV수(대)",
        "발생건수": "범죄발생건수",
        "범죄율": "범죄율(10만당)",
        "CCTV_밀도": "CCTV밀도(1만당)",
        "CCTV당_범죄수": "CCTV당범죄건수"
    })
    st.dataframe(
        disp.style.format({
            "인구수": "{:,.0f}",
            "CCTV수(대)": "{:,.0f}",
            "범죄발생건수": "{:,.0f}",
            "범죄율(10만당)": "{:.1f}",
            "CCTV밀도(1만당)": "{:.1f}",
            "CCTV당범죄건수": "{:.2f}"
        }),
        use_container_width=True
    )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 1: 범죄 발생 건수(절대량) 분석
# ════════════════════════════════════════════════════════════════════════════════
st.header("1️⃣ 범죄 발생 건수(절대량) 분석")

corr_abs = df_final["CCTV수"].corr(df_final["발생건수"])
st.caption(f"CCTV수 ↔ 범죄발생건수 상관계수: **{corr_abs:.4f}**")

view1 = st.segmented_control(
    "그래프 형태 선택",
    options=["📊 추이 분석", "📈 상관관계"],
    default="📊 추이 분석", key="view_1"
)

if view1 == "📊 추이 분석":
    st.subheader("CCTV 설치 규모에 따른 범죄 발생 건수 변화")
    df_s = df_final.sort_values("CCTV수")
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.bar(df_s["자치구"], df_s["CCTV수"], color="lightgray", alpha=0.6)
    ax1.set_ylabel("CCTV 설치 수 (대)", color="gray")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(df_s["자치구"], df_s["발생건수"], color="royalblue",
             marker="s", linewidth=2)
    ax2.set_ylabel("범죄 발생 건수 (건)", color="royalblue")
    ax1.set_title("CCTV 설치 수 오름차순 기준 · 범죄 발생 건수",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    st.info("💡 절대 건수는 유동인구·상권 밀집도에 따라 함께 높아집니다. 인구 보정 지표(섹션 2)와 함께 해석하세요.")

else:
    st.subheader("CCTV 수 vs 범죄 발생 건수 산점도")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df_final, x="CCTV수", y="발생건수",
                scatter_kws={"s": 60}, line_kws={"color": "red"}, ax=ax)
    for _, row in df_final.iterrows():
        ax.text(row["CCTV수"] + 60, row["발생건수"] + 30, row["자치구"], fontsize=8)
    ax.set_title(f"CCTV 수 vs 범죄 발생 건수  (상관계수: {corr_abs:.4f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("CCTV 설치 수 (대)")
    ax.set_ylabel("범죄 발생 건수 (건)")
    plt.tight_layout()
    st.pyplot(fig)
    if corr_abs > 0:
        st.warning("⚠️ 양의 상관관계: CCTV가 범죄를 늘린다는 의미가 아닙니다. **인구·상권 밀도** 교란변수가 작용합니다.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 2: 인구 대비 범죄율 분석
# ════════════════════════════════════════════════════════════════════════════════
st.header("2️⃣ 인구 대비 범죄율(비율) 분석")

corr_rate = df_final["CCTV수"].corr(df_final["범죄율"])
st.caption(f"CCTV수 ↔ 범죄율(10만당) 상관계수: **{corr_rate:.4f}**")

view2 = st.segmented_control(
    "그래프 형태 선택",
    options=["📊 추이 분석", "📈 상관관계"],
    default="📊 추이 분석", key="view_2"
)

if view2 == "📊 추이 분석":
    st.subheader("CCTV 설치 규모에 따른 인구 10만명당 범죄율 변화")
    df_s = df_final.sort_values("CCTV수")
    fig2, ax3 = plt.subplots(figsize=(13, 5))
    ax3.bar(df_s["자치구"], df_s["CCTV수"], color="skyblue", alpha=0.7)
    ax3.set_ylabel("CCTV 설치 수 (대)", color="steelblue")
    ax3.tick_params(axis="x", rotation=45)
    ax4 = ax3.twinx()
    ax4.plot(df_s["자치구"], df_s["범죄율"], color="crimson",
             marker="o", linewidth=2.5)
    ax4.set_ylabel("인구 10만명당 범죄율 (건)", color="crimson")
    ax3.set_title("CCTV 설치 수 오름차순 기준 · 인구 대비 범죄율",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.subheader("CCTV 수 vs 인구 대비 범죄율 산점도")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df_final, x="CCTV수", y="범죄율",
                scatter_kws={"s": 60, "alpha": 0.8}, line_kws={"color": "red"}, ax=ax)
    for _, row in df_final.iterrows():
        ax.text(row["CCTV수"] + 60, row["범죄율"] + 0.5, row["자치구"], fontsize=8)
    ax.set_title(f"CCTV 수 vs 인구 10만명당 범죄율  (상관계수: {corr_rate:.4f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("CCTV 설치 수 (대)")
    ax.set_ylabel("범죄율 (인구 10만명당)")
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 4: CCTV 인프라 밀도 분석 (NEW)
# ════════════════════════════════════════════════════════════════════════════════
st.header("🔍 CCTV 인프라 밀도 분석 (인구 1만명당)")
st.markdown("단순 설치 대수가 아닌 **인구 대비 CCTV 밀도**로 자치구 간 형평성을 비교합니다.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("인구 1만명당 CCTV 수 순위")
    df_dens = df_final.sort_values("CCTV_밀도", ascending=True)
    mean_d  = df_final["CCTV_밀도"].mean()
    colors_d = ["#d73027" if v > mean_d else "#4575b4" for v in df_dens["CCTV_밀도"]]
    fig4, ax4 = plt.subplots(figsize=(8, 9))
    b4 = ax4.barh(df_dens["자치구"], df_dens["CCTV_밀도"], color=colors_d)
    ax4.axvline(mean_d, color="orange", linestyle="--", linewidth=2,
                label=f"평균 {mean_d:.1f}대/만명")
    ax4.set_title("인구 1만명당 CCTV 밀도", fontsize=12, fontweight="bold")
    ax4.set_xlabel("대 / 인구 1만명")
    ax4.legend()
    for bar, val in zip(b4, df_dens["CCTV_밀도"]):
        ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig4)
    st.caption("🔴 평균 초과 / 🔵 평균 이하")

with col2:
    st.subheader("CCTV 밀도 vs 범죄율 산점도")
    corr_dens = df_final["CCTV_밀도"].corr(df_final["범죄율"])
    fig5, ax5 = plt.subplots(figsize=(8, 7))
    sns.regplot(data=df_final, x="CCTV_밀도", y="범죄율",
                scatter_kws={"s": 70, "alpha": 0.8}, line_kws={"color": "red"}, ax=ax5)
    for _, row in df_final.iterrows():
        ax5.text(row["CCTV_밀도"] + 0.15, row["범죄율"] + 0.5, row["자치구"], fontsize=8)
    ax5.set_title(f"CCTV 밀도 vs 범죄율  (상관계수: {corr_dens:.4f})",
                  fontsize=12, fontweight="bold")
    ax5.set_xlabel("인구 1만명당 CCTV 수 (대)")
    ax5.set_ylabel("인구 10만명당 범죄율 (건)")
    plt.tight_layout()
    st.pyplot(fig5)
    if corr_dens < 0:
        st.success(f"✅ CCTV 밀도 ↑ → 범죄율 ↓ 경향 확인 (상관계수: {corr_dens:.4f})")
    else:
        st.info(f"ℹ️ 상관계수: {corr_dens:.4f}")

# CCTV 1대당 범죄 건수
st.subheader("자치구별 CCTV 1대당 범죄 건수 (부담 지표)")
st.caption("높을수록 CCTV가 상대적으로 부족함을 의미합니다.")
df_eff  = df_final.sort_values("CCTV당_범죄수", ascending=False)
mean_eff = df_final["CCTV당_범죄수"].mean()
colors_e = ["#d73027" if v > mean_eff else "#74add1" for v in df_eff["CCTV당_범죄수"]]
fig6, ax6 = plt.subplots(figsize=(14, 4))
ax6.bar(df_eff["자치구"], df_eff["CCTV당_범죄수"], color=colors_e)
ax6.axhline(mean_eff, color="black", linestyle="--", linewidth=1.5,
            label=f"평균 {mean_eff:.2f}건/대")
ax6.set_title("자치구별 CCTV 1대당 범죄 발생 건수 (2024년)",
              fontsize=13, fontweight="bold")
ax6.set_ylabel("범죄 건수 / CCTV 1대")
ax6.tick_params(axis="x", rotation=45)
ax6.legend()
plt.tight_layout()
st.pyplot(fig6)
st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 3: 범죄 유형별 발생율 및 전년 대비 증감 (FIXED)
# ════════════════════════════════════════════════════════════════════════════════
st.header("📈 범죄 유형별 발생율 및 전년 대비 증감 분석")
st.markdown(
    "2015 ~ 2024년 **인구 10만명당 발생율** 추이와 CCTV 증가의 관계를 범죄 유형별로 분석합니다."
)

CRIME_TYPES = ["살인", "강도", "강간·강제추행", "절도", "폭력"]
tabs3       = st.tabs([f"🚨 {c}" for c in CRIME_TYPES])

for i, crime in enumerate(CRIME_TYPES):
    with tabs3[i]:
        base_val = df_trend.loc[df_trend["연도"] == "2015", crime].values[0]
        if base_val == 0: base_val = 1
        df_trend[f"{crime}_비율"] = (df_trend[crime] / base_val) * 100
        df_trend[f"{crime}_증감"] = df_trend[crime].diff()

        col_a, col_b = st.columns([3, 2])

        with col_a:
            st.subheader(f"{crime} 발생율 추이 (2015=100%)")
            fig, ax1 = plt.subplots(figsize=(11, 5))
            ax1.bar(df_trend["연도"], df_trend["CCTV수량"],
                    color="lightgray", alpha=0.5, label="CCTV 수량")
            ax1.set_ylabel("CCTV 설치 수량 (대)", color="gray")
            ax1.tick_params(axis="y", labelcolor="gray")
            ax1.tick_params(axis="x", rotation=30)

            ax2 = ax1.twinx()
            ax2.plot(df_trend["연도"], df_trend[f"{crime}_비율"],
                     marker="o", color="crimson", linewidth=2.5)
            ax2.axhline(100, color="crimson", linestyle="--", alpha=0.4, linewidth=1)
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax2.set_ylabel("2015년 대비 발생율 (%)", color="crimson")
            ax2.tick_params(axis="y", labelcolor="crimson")
            for _, row in df_trend.iterrows():
                ax2.text(row["연도"], row[f"{crime}_비율"] + 2.5,
                         f"{row[f'{crime}_비율']:.0f}%", ha="center",
                         fontsize=8, color="darkred", fontweight="bold")

            ax1.set_title(f"{crime} 발생율 추이 (CCTV 배경)", fontsize=13, fontweight="bold")
            lines1, lbl1 = ax1.get_legend_handles_labels()
            lines2, lbl2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

        with col_b:
            st.subheader("연도별 상세 수치")
            df_disp = df_trend[["연도", "CCTV수량", crime, f"{crime}_증감"]].copy()
            df_disp.columns = ["연도", "CCTV수", f"{crime}발생율", "전년대비"]
            df_disp["전년대비"] = df_disp["전년대비"].fillna(0)
            st.dataframe(
                df_disp.style.format({
                    "CCTV수": "{:,.0f}",
                    f"{crime}발생율": "{:.2f}",
                    "전년대비": "{:+.2f}"
                }).background_gradient(subset=["전년대비"], cmap="RdYlGn_r"),
                use_container_width=True, height=370
            )

        # 요약 지표
        c_chg = (df_trend[crime].iloc[-1] - df_trend[crime].iloc[0]) / df_trend[crime].iloc[0] * 100
        cctv_chg = (df_trend["CCTV수량"].iloc[-1] - df_trend["CCTV수량"].iloc[0]) \
                   / df_trend["CCTV수량"].iloc[0] * 100
        st.metric(
            label=f"CCTV 증가율 (2015→2024): +{cctv_chg:.1f}%",
            value=f"{crime} 발생율 변화",
            delta=f"{'감소' if c_chg < 0 else '증가'} {abs(c_chg):.1f}%",
            delta_color="inverse"
        )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 5: 10년간 CCTV 증가 vs 서울 전체 범죄율 추이 (NEW)
# ════════════════════════════════════════════════════════════════════════════════
st.header("📊 10년간 CCTV 증가 vs 서울 전체 범죄율 추이 (2015–2024)")

fig7, ax7a = plt.subplots(figsize=(13, 5))
ax7a.fill_between(df_trend["연도"], df_trend["CCTV수량"], alpha=0.25, color="steelblue")
ax7a.plot(df_trend["연도"], df_trend["CCTV수량"],
          color="steelblue", linewidth=2, marker="o", label="CCTV 설치 수량")
ax7a.set_ylabel("CCTV 누적 설치 수량 (대)", color="steelblue")
ax7a.tick_params(axis="y", labelcolor="steelblue")
ax7a.tick_params(axis="x", rotation=30)

ax7b = ax7a.twinx()
ax7b.plot(df_trend["연도"], df_trend["총범죄율"],
          color="crimson", linewidth=2.5, marker="^", linestyle="--",
          label="5대 범죄율 (10만당)")
ax7b.set_ylabel("인구 10만명당 범죄율 (건)", color="crimson")
ax7b.tick_params(axis="y", labelcolor="crimson")

ax7a.set_title("서울시 전체 CCTV 누적 수량 vs 5대 범죄율 추이 (2015–2024)",
               fontsize=14, fontweight="bold")
lines_a, lbl_a = ax7a.get_legend_handles_labels()
lines_b, lbl_b = ax7b.get_legend_handles_labels()
ax7a.legend(lines_a + lines_b, lbl_a + lbl_b, loc="upper right")
plt.tight_layout()
st.pyplot(fig7)

# 변화율 요약 카드
chg_cctv  = (df_trend["CCTV수량"].iloc[-1] - df_trend["CCTV수량"].iloc[0]) \
            / df_trend["CCTV수량"].iloc[0] * 100
chg_crime = (df_trend["총범죄율"].iloc[-1] - df_trend["총범죄율"].iloc[0]) \
            / df_trend["총범죄율"].iloc[0] * 100
corr_t    = df_trend["CCTV수량"].corr(df_trend["총범죄율"])

c1, c2, c3 = st.columns(3)
c1.metric("CCTV 증가율 (2015→2024)", f"+{chg_cctv:.1f}%")
c2.metric("5대 범죄율 변화 (2015→2024)",
          f"{chg_crime:.1f}%",
          delta="감소" if chg_crime < 0 else "증가",
          delta_color="inverse")
c3.metric("연도별 CCTV–범죄율 상관계수", f"{corr_t:.4f}",
          help="음수: CCTV 증가 시 범죄율 감소 경향")
st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 6: 범죄 발생 장소 분포 (NEW)
# ════════════════════════════════════════════════════════════════════════════════
st.header("📍 서울시 5대 범죄 발생 장소 분포 (2024년)")
st.markdown("**어디서 범죄가 주로 발생하는지** 파악해 CCTV 설치 전략에 활용합니다.")

try:
    crime_loc_raw = load_crime_loc()

    # 2024년 장소 컬럼: 2024.1 ~ 2024.12 (소계 제외, 12개 장소)
    loc_cols   = [f"2024.{i}" for i in range(1, 13)]
    loc_labels = [
        "아파트/연립다세대", "단독주택",   "기타거주시설",
        "도로/골목길",       "상점/창고",  "공중위생업소",
        "음식점/유흥업소",   "역/대합실",  "교통수단",
        "문화/체육시설",     "학교/도서관", "기타"
    ]

    crime_loc_filter = ["살인", "강도", "강간강제추행", "절도", "폭력"]
    crime_loc_kor    = ["살인", "강도", "강간·강제추행", "절도", "폭력"]

    rows_loc = []
    for c_type, c_kor in zip(crime_loc_filter, crime_loc_kor):
        row = crime_loc_raw[crime_loc_raw["범죄별(2)"] == c_type]
        if not row.empty:
            vals = []
            for col in loc_cols:
                v = row[col].values[0]
                v = pd.to_numeric(str(v).replace(",", "").replace("-", "0"),
                                  errors="coerce")
                vals.append(float(v) if pd.notna(v) else 0.0)
            rows_loc.append([c_kor] + vals)

    df_loc = pd.DataFrame(rows_loc, columns=["범죄유형"] + loc_labels).set_index("범죄유형")

    # 히트맵 (비율)
    st.subheader("범죄 유형 × 발생 장소 비율 히트맵")
    df_loc_pct = df_loc.div(df_loc.sum(axis=1), axis=0) * 100
    fig8, ax8 = plt.subplots(figsize=(14, 4))
    sns.heatmap(df_loc_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax8, cbar_kws={"label": "비율 (%)"})
    ax8.set_title("2024년 범죄 유형별 발생 장소 비율 히트맵 (%)",
                  fontsize=13, fontweight="bold")
    ax8.set_xticklabels(ax8.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig8)

    # 범죄 유형별 상세 탭
    st.subheader("범죄 유형별 발생 장소 상세")
    tabs_loc = st.tabs([f"🔍 {c}" for c in crime_loc_kor])
    for ti, crime in enumerate(crime_loc_kor):
        with tabs_loc[ti]:
            row_data = df_loc_pct.loc[crime].sort_values(ascending=False)
            row_data = row_data[row_data > 0]
            fig9, ax9 = plt.subplots(figsize=(9, 4))
            palette9 = sns.color_palette("RdYlBu_r", len(row_data))
            b9 = ax9.barh(row_data.index[::-1], row_data.values[::-1],
                          color=palette9[::-1])
            ax9.set_title(f"{crime} 발생 장소 비율 (2024년)",
                          fontsize=12, fontweight="bold")
            ax9.set_xlabel("비율 (%)")
            for bar, val in zip(b9, row_data.values[::-1]):
                ax9.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                         f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig9)

except Exception as e:
    st.warning(f"범죄 발생 장소 데이터를 불러오는 중 오류: {e}")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 7: CCTV 투자 우선순위 종합 분석 (NEW)
# ════════════════════════════════════════════════════════════════════════════════
st.header("🏆 자치구별 CCTV 투자 우선순위 종합 분석")
st.markdown(
    "**범죄율 순위** + **CCTV 밀도 역순위** + **CCTV 1대당 범죄수 순위**를 합산해 "
    "추가 설치가 가장 시급한 자치구를 도출합니다."
)

df_pri = df_final[["자치구", "범죄율", "CCTV_밀도", "CCTV당_범죄수"]].copy()
df_pri["범죄율_순위"]     = df_pri["범죄율"].rank(ascending=False)
df_pri["CCTV밀도_역순위"] = df_pri["CCTV_밀도"].rank(ascending=True)
df_pri["효율성_순위"]     = df_pri["CCTV당_범죄수"].rank(ascending=False)
df_pri["종합점수"]        = df_pri[["범죄율_순위", "CCTV밀도_역순위", "효율성_순위"]].sum(axis=1)
df_pri = df_pri.sort_values("종합점수", ascending=False).reset_index(drop=True)
df_pri.index += 1

cp1, cp2 = st.columns([1, 2])

with cp1:
    st.dataframe(
        df_pri[["자치구", "범죄율", "CCTV_밀도", "CCTV당_범죄수", "종합점수"]]
        .rename(columns={
            "범죄율": "범죄율\n(10만당)",
            "CCTV_밀도": "CCTV밀도\n(1만당)",
            "CCTV당_범죄수": "CCTV당\n범죄건수",
            "종합점수": "종합점수"
        })
        .style.format({
            "범죄율\n(10만당)": "{:.1f}",
            "CCTV밀도\n(1만당)": "{:.1f}",
            "CCTV당\n범죄건수": "{:.2f}",
            "종합점수": "{:.0f}"
        }).background_gradient(subset=["종합점수"], cmap="RdYlGn_r"),
        use_container_width=True, height=700
    )

with cp2:
    top10 = df_pri.head(10)
    fig10, ax10 = plt.subplots(figsize=(9, 7))
    palette10 = sns.color_palette("Reds_r", 10)
    b10 = ax10.barh(top10["자치구"].iloc[::-1],
                    top10["종합점수"].iloc[::-1],
                    color=palette10[::-1])
    ax10.set_title("CCTV 추가 설치 우선순위 TOP 10 자치구",
                   fontsize=13, fontweight="bold")
    ax10.set_xlabel("종합 우선순위 점수 (높을수록 시급)")
    for bar, val in zip(b10, top10["종합점수"].iloc[::-1]):
        ax10.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                  f"{val:.0f}점", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig10)

st.success(
    "✅ **분석 요약**: 우선순위가 높은 자치구는 ① 범죄율이 높고, "
    "② 인구 대비 CCTV 밀도가 낮으며, ③ CCTV 1대당 커버해야 할 범죄 건수가 많은 곳입니다."
)
