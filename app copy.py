import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.font_manager as fm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

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
# DATA = "./DATA/"
# CCTV_FILE      = DATA + "서울시_자치구__범죄예방_수사용__CCTV_설치현황_25_12_31_기준_.xlsx"
# CRIME_CSV      = DATA + "5대_범죄_발생현황_20260512140024.csv"
# POP_FILE       = DATA + "등록인구_2024.xlsx"            # 2024년 구별 인구 (df_final용)
# POP_TREND_FILE = DATA + "등록인구_20260512150716.xlsx"  # 2015–2024 다년도 인구 (trend용)
# LOC_FILE       = DATA + "5대_범죄_발생장소별_현황.xlsx"
# CCTV_NEW_FILE  = DATA + "서울시_자치구_CCTV_설치현황.xlsx"    # 전체 합계 연도별 CCTV
# CRIME_NEW_FILE = DATA + "crime_2015-2024_.xlsx"              # 유형별 연도별 범죄
# UTIL_FILE      = DATA + "util_clean.xlsx"                    # CCTV 실시간 탐지 실적

CCTV_FILE      = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\서울시 자치구 (범죄예방 수사용) CCTV 설치현황(25.12.31 기준).xlsx"
CRIME_CSV      = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\5대+범죄+발생현황_20260512140024.csv"
POP_FILE       = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\등록인구_2024.xlsx"            # 2024년 구별 인구 (df_final용)
POP_TREND_FILE = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\등록인구_20260512150716.xlsx"  # 2015–2024 다년도 인구 (trend용)
LOC_FILE       = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\5대+범죄+발생장소별+현황.xlsx"
CCTV_NEW_FILE  = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\서울시 자치구 CCTV 설치현황.xlsx"    # 전체 합계 연도별 CCTV
CRIME_NEW_FILE = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\crime(2015-2024).xlsx"              # 유형별 연도별 범죄
UTIL_FILE      = r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\util_clean.xlsx"                    # CCTV 실시간 탐지 실적

# ════════════════════════════════════════════════════════════════════════════════
# 데이터 로딩 (캐싱)
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_cctv():
    raw = pd.read_excel(CCTV_FILE, header=2)
    raw = raw.drop(columns=["Unnamed: 0"], errors="ignore")
    raw = raw.iloc[:-2]
    raw = raw[raw["구분"] != "계"].copy()
    year_cols = [c for c in raw.columns if str(c).endswith("년")]
    raw[year_cols] = raw[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return raw.set_index("순번")


@st.cache_data
def load_crime_csv():
    return pd.read_csv(CRIME_CSV, encoding="utf-8")


@st.cache_data
def load_population():
    """2024년 구별 등록인구 (등록인구_2024.xlsx)."""
    return pd.read_excel(POP_FILE)


@st.cache_data
def load_population_trend():
    """2015–2024 연도별 다년도 인구 (등록인구_20260512150716.xlsx)."""
    return pd.read_excel(POP_TREND_FILE)


@st.cache_data
def load_crime_loc():
    return pd.read_excel(LOC_FILE)


@st.cache_data
def load_cctv_new():
    """서울시_자치구_CCTV_설치현황.xlsx — 전체 합계(계) 행 연도별 누적 대수."""
    raw = pd.read_excel(CCTV_NEW_FILE, header=2)
    total = raw[raw["구분"] == "계"]
    years = [f"{y}년" for y in range(2015, 2025)]
    vals  = total[years].iloc[0].values
    return pd.DataFrame({"연도": [str(y) for y in range(2015, 2025)],
                         "CCTV누적": vals.astype(float)})


@st.cache_data
def load_util():
    """util_clean.xlsx — 연도별 CCTV 실시간 탐지 실적."""
    return pd.read_excel(UTIL_FILE)


@st.cache_data
def build_crime_yearly():
    """crime_2015-2024_.xlsx → 연도·유형별 서울 전체 발생건수 DataFrame."""
    headers = pd.read_excel(CRIME_NEW_FILE, header=None, nrows=4)
    data    = pd.read_excel(CRIME_NEW_FILE, skiprows=4, header=None)
    crime_list = ["절도", "폭력", "살인", "강도", "강간·강제추행"]

    rows = []
    for col in range(2, len(headers.columns)):
        try:
            yr = str(int(float(headers.iloc[0, col])))
        except Exception:
            yr = str(headers.iloc[0, col]).strip()
        c_type = str(headers.iloc[2, col]).strip()
        s_type = str(headers.iloc[3, col]).strip()
        if (yr.isdigit() and 2015 <= int(yr) <= 2024
                and s_type == "발생" and c_type in crime_list):
            val = pd.to_numeric(
                data.loc[data[1] == "소계", col], errors="coerce"
            ).fillna(0).sum()
            rows.append({"연도": yr, "유형": c_type, "건수": int(val)})

    return pd.DataFrame(rows)


@st.cache_data
def build_df_final():
    """2024년 기준 25개 자치구 통합 데이터프레임."""
    cctv      = load_cctv()
    crime_raw = load_crime_csv()
    pop_raw   = load_population()

    # ── 인구 (등록인구_2024.xlsx: '2024 4/4.1' = 총인구 계 소계) ─────────────
    # 동별(1)이 NaN인 행 = 구(자치구) 레벨, 총인구 컬럼 = '2024 4/4.1'
    pop_24 = pop_raw[pop_raw["동별(1)"].isna() & pop_raw["동별(2)"].notna()][
        ["동별(2)", "2024 4/4.1"]
    ].copy()
    pop_24.columns = ["자치구", "인구수"]
    pop_24["자치구"] = pop_24["자치구"].astype(str).str.strip()
    pop_24["인구수"] = pd.to_numeric(
        pop_24["인구수"].astype(str).str.replace(",", ""), errors="coerce"
    )
    pop_24 = pop_24.dropna(subset=["인구수"])
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
    pop_raw   = load_population_trend()   # 다년도 인구 파일 (등록인구_20260512150716.xlsx)

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
df_final      = build_df_final()
df_trend      = build_crime_trend()
cctv_raw      = load_cctv()
df_all_crime  = build_crime_yearly()          # 유형별 연도별 발생건수
df_10yr_base  = load_cctv_new()               # 연도별 CCTV 누적 합계
df_util       = load_util()                   # 실시간 탐지 실적

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
# 섹션 3: 범죄 유형별 발생 추이 및 상관성 분석 (인터랙티브)
# ════════════════════════════════════════════════════════════════════════════════
st.header("📈 범죄 유형별 발생 추이 및 CCTV 상관성 분석 (2015–2024)")
st.markdown("2015 ~ 2024년 서울시 **5대 범죄 유형별 발생 건수**와 CCTV 누적 설치량의 관계를 분석합니다.")
st.write("📌 아래 버튼을 눌러 분석할 범죄 유형을 선택하세요. **(중복 선택 가능)**")

CRIME_TYPES_BTN = ["절도", "폭력", "살인", "강도", "강간·강제추행"]

if "active_crimes" not in st.session_state:
    st.session_state.active_crimes = set()

# ── 토글 버튼 ─────────────────────────────────────────────────────────────────
btn_cols = st.columns(len(CRIME_TYPES_BTN))
for i, crime in enumerate(CRIME_TYPES_BTN):
    is_active = crime in st.session_state.active_crimes
    label = f"🔵 {crime}" if is_active else f"⚪ {crime}"
    if btn_cols[i].button(label, key=f"btn_{crime}", use_container_width=True):
        if is_active:
            st.session_state.active_crimes.remove(crime)
        else:
            st.session_state.active_crimes.add(crime)
        st.rerun()

if not st.session_state.active_crimes:
    st.info("범죄 유형 버튼을 클릭하면 분석 차트가 활성화됩니다.")
else:
    selected = sorted(st.session_state.active_crimes,
                      key=lambda x: CRIME_TYPES_BTN.index(x))
    df_sub   = df_all_crime[df_all_crime["유형"].isin(selected)]

    # ── 왼쪽 차트 / 오른쪽 수치 테이블 ─────────────────────────────────────
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        color_map = dict(zip(CRIME_TYPES_BTN, COLORS))

        fig_s3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_s3.add_trace(
            go.Bar(x=df_10yr_base["연도"], y=df_10yr_base["CCTV누적"],
                   name="CCTV 누적 (배경)", marker_color="rgba(149,165,166,0.4)"),
            secondary_y=False
        )
        for crime in selected:
            crime_df = df_sub[df_sub["유형"] == crime].sort_values("연도")
            fig_s3.add_trace(
                go.Scatter(x=crime_df["연도"], y=crime_df["건수"],
                           name=crime, mode="lines+markers",
                           line=dict(width=3, color=color_map[crime]),
                           marker=dict(size=8)),
                secondary_y=True
            )
        fig_s3.update_layout(
            template="plotly_white", hovermode="x unified",
            height=420, margin=dict(b=0),
            yaxis_title="CCTV 누적 설치 수 (대)",
            yaxis2_title="범죄 발생 건수 (건)"
        )
        st.plotly_chart(fig_s3, use_container_width=True)

        # 상관계수 카드
        df_sub_agg = df_sub.groupby("연도")["건수"].sum().reset_index().sort_values("연도")
        merged_corr = df_10yr_base.merge(df_sub_agg, on="연도")
        r_val, _ = stats.pearsonr(merged_corr["CCTV누적"], merged_corr["건수"])
        label_str = " + ".join(selected)
        color_card = "#e7f5ff" if r_val < 0 else "#fff5f5"
        border_color = "#a5d8ff" if r_val < 0 else "#ffa8a8"
        text_color  = "#1864ab" if r_val < 0 else "#c92a2a"
        st.markdown(
            f'<div style="padding:12px 20px; border-radius:0 0 12px 12px; '
            f'background:{color_card}; color:{text_color}; border-top:2px solid {border_color}; '
            f'text-align:center; margin-top:-14px; font-size:1.05em;">'
            f'📊 <b>{label_str}</b> 합계 ↔ CCTV 상관계수(r): <b>{r_val:.4f}</b></div>',
            unsafe_allow_html=True
        )

    with col_table:
        st.subheader("연도별 상세 수치")
        # 기본 베이스: 연도 + CCTV 누적
        tbl = df_10yr_base.copy()
        for crime in selected:
            crime_yearly = (df_sub[df_sub["유형"] == crime]
                            .sort_values("연도")
                            .set_index("연도")["건수"])
            tbl[crime]           = tbl["연도"].map(crime_yearly)
            tbl[f"{crime}_증감"] = tbl[crime].diff().fillna(0)

        fmt_dict = {"CCTV누적": "{:,.0f}"}
        col_order = ["연도", "CCTV누적"]
        for crime in selected:
            fmt_dict[crime]           = "{:,.0f}"
            fmt_dict[f"{crime}_증감"] = "{:+,.0f}"
            col_order += [crime, f"{crime}_증감"]

        st.dataframe(
            tbl[col_order].style
            .format(fmt_dict)
            .background_gradient(
                subset=[f"{c}_증감" for c in selected], cmap="RdYlGn_r"
            ),
            use_container_width=True, height=390
        )

    # ── 변화율 지표 (crime 유형별 + CCTV) ──────────────────────────────────
    st.markdown("##### 📉 2015 → 2024 변화율 요약")
    cctv_chg = (df_10yr_base["CCTV누적"].iloc[-1] - df_10yr_base["CCTV누적"].iloc[0]) \
               / df_10yr_base["CCTV누적"].iloc[0] * 100

    metric_cols = st.columns(len(selected) + 1)
    metric_cols[0].metric("CCTV 증가율", f"+{cctv_chg:.1f}%")
    for j, crime in enumerate(selected):
        crime_df = df_sub[df_sub["유형"] == crime].sort_values("연도")
        v_first  = crime_df["건수"].iloc[0]
        v_last   = crime_df["건수"].iloc[-1]
        chg      = (v_last - v_first) / v_first * 100 if v_first != 0 else 0
        metric_cols[j + 1].metric(
            f"{crime} 발생 변화",
            f"{v_last:,.0f}건",
            delta=f"{'▼' if chg < 0 else '▲'} {abs(chg):.1f}%",
            delta_color="inverse"
        )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# 섹션 3-B: 연도별 CCTV 인프라 및 실시간 탐지 총량 분석
# ════════════════════════════════════════════════════════════════════════════════
st.header("🚀 연도별 CCTV 인프라 및 실시간 탐지 총량 분석")
st.caption("※ 탐지 실적 데이터: 2021년 ~ 2024년 CCTV 통합관제센터 실시간 대응 통계")

short_years = ["2021", "2022", "2023", "2024"]
detect_vals = [
    df_util[df_util["연도"].astype(str).str.contains(yr)]["합계"].sum()
    for yr in short_years
]
df_util_base = df_10yr_base[df_10yr_base["연도"].isin(short_years)].copy()
df_util_base["탐지실적"] = detect_vals

fig_util = make_subplots(specs=[[{"secondary_y": True}]])
fig_util.add_trace(
    go.Bar(x=df_util_base["연도"], y=df_util_base["CCTV누적"],
           name="CCTV 설치 누적", marker_color="#aed6f1"),
    secondary_y=False
)
fig_util.add_trace(
    go.Scatter(x=df_util_base["연도"], y=df_util_base["탐지실적"],
               name="실시간 탐지 실적", mode="lines+markers",
               line=dict(color="#e74c3c", width=4, dash="dot"),
               marker=dict(size=12)),
    secondary_y=True
)
fig_util.update_layout(
    template="plotly_white", height=420, margin=dict(b=0),
    hovermode="x unified",
    yaxis_title="CCTV 누적 설치 수 (대)",
    yaxis2_title="탐지 실적 (건)"
)
st.plotly_chart(fig_util, use_container_width=True)

r_util, _ = stats.pearsonr(df_util_base["CCTV누적"], df_util_base["탐지실적"])
st.markdown(
    f'<div style="padding:12px 20px; border-radius:0 0 12px 12px; '
    f'background:#fff5f5; color:#c92a2a; border-top:2px solid #ffa8a8; '
    f'text-align:center; margin-top:-14px; font-size:1.05em;">'
    f'🚀 인프라 확충 대비 실시간 탐지 효율 상관계수(r): <b>{r_util:.4f}</b></div>',
    unsafe_allow_html=True
)

# 탐지 유형별 Top10 Bar
st.subheader("이벤트 유형별 탐지 실적 (2021–2024 누적 Top 10)")
df_event_top = (df_util.groupby("이벤트상세")["합계"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index())
fig_top, ax_top = plt.subplots(figsize=(12, 4))
sns.barplot(x="합계", y="이벤트상세", data=df_event_top,
            palette="Blues_d", ax=ax_top)
ax_top.set_title("CCTV 탐지 이벤트 유형 TOP 10 (2021–2024 누적)",
                 fontsize=13, fontweight="bold")
ax_top.set_xlabel("누적 탐지 건수")
for bar, val in zip(ax_top.patches, df_event_top["합계"]):
    ax_top.text(val + 5, bar.get_y() + bar.get_height()/2,
                f"{int(val):,}", va="center", fontsize=9, fontweight="bold")
plt.tight_layout()
st.pyplot(fig_top)

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