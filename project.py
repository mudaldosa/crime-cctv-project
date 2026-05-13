import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import numpy as np

# 1. 페이지 및 테마 설정
st.set_page_config(page_title="서울시 치안 데이터 고도화 분석", layout="wide")

# [고급 UI 및 레이아웃 최적화 CSS]
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    
    /* 버튼 공통 스타일 (둥근 모서리) */
    div.stButton > button {
        border-radius: 30px !important;
        padding: 8px 20px !important;
        font-weight: bold !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid #d1d8e0 !important;
    }

    /* 상관계수 결과 박스 (그래프에 밀착시키기 위해 margin-top 최소화) */
    .result-card {
        padding: 12px 20px;
        border-radius: 0px 0px 15px 15px; /* 아래쪽만 둥글게 해서 그래프와 연결감 부여 */
        text-align: center;
        margin-top: -15px; /* 그래프와의 간격을 좁힘 */
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-size: 1.05em;
    }
    
    /* 섹션 구분선 스타일 */
    hr { margin: 2rem 0; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------
# 2. 데이터 로드 (캐싱)
# ----------------------------------------------------------------
@st.cache_data
def load_data():
    df_cctv_raw = pd.read_excel("서울시 자치구 CCTV 설치현황.xlsx", header=2)
    cctv_total = df_cctv_raw[df_cctv_raw['구분'] == '계']
    years_cctv = [f"{y}년" for y in range(2015, 2025)]
    cctv_vals = cctv_total[years_cctv].iloc[0].values

    crime_path = "crime(2015-2024).xlsx"
    headers = pd.read_excel(crime_path, header=None, nrows=4)
    data = pd.read_excel(crime_path, skiprows=4, header=None)
    df_util = pd.read_excel("util_clean.xlsx")
    
    return cctv_vals, headers, data, df_util

try:
    cctv_vals, crime_headers, crime_data, df_util = load_data()
except Exception as e:
    st.error(f"❌ 데이터 로드 실패: {e}")
    st.stop()

# ----------------------------------------------------------------
# 3. 데이터 가공
# ----------------------------------------------------------------
crime_list = ['절도', '폭력', '살인', '강도', '강간·강제추행']
yearly_results = []

for col in range(2, len(crime_headers.columns)):
    try:
        yr = str(int(float(crime_headers.iloc[0, col])))
    except:
        yr = str(crime_headers.iloc[0, col]).strip()
    c_type = str(crime_headers.iloc[2, col]).strip()
    s_type = str(crime_headers.iloc[3, col]).strip()
    
    if yr.isdigit() and 2015 <= int(yr) <= 2024 and s_type == '발생' and c_type in crime_list:
        val = pd.to_numeric(crime_data.loc[crime_data[1] == '소계', col], errors='coerce').fillna(0).sum()
        yearly_results.append({'연도': yr, '유형': c_type, '건수': val})

df_all_crime = pd.DataFrame(yearly_results)
df_10yr_base = pd.DataFrame({
    '연도': [str(y) for y in range(2015, 2025)],
    'CCTV누적': cctv_vals
})

# ----------------------------------------------------------------
# 4. 메인 대시보드 구성
# ----------------------------------------------------------------

st.title("🏙️ 서울시 치안 인프라 & 범죄 대응 정밀 분석 솔루션")
st.markdown("---")

# --- 1번 섹션 ---
st.subheader("1️⃣ CCTV 설치량 대비 전체 범죄(5대 범죄 합계) 추이")
df_total_agg = df_all_crime.groupby('연도')['건수'].sum().reset_index()
df_1 = df_10yr_base.merge(df_total_agg, on='연도')

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Bar(x=df_1['연도'], y=df_1['CCTV누적'], name="CCTV 누적", 
                      marker_color='rgba(52, 152, 219, 0.4)'), secondary_y=False)
fig1.add_trace(go.Scatter(x=df_1['연도'], y=df_1['건수'], name="5대 범죄 전체 합계", 
                          line=dict(color='#2c3e50', width=4), mode='lines+markers'), secondary_y=True)
fig1.update_layout(template="plotly_white", hovermode="x unified", height=400, margin=dict(b=0))
st.plotly_chart(fig1, use_container_width=True)

r_1, _ = stats.pearsonr(df_1['CCTV누적'], df_1['건수'])
st.markdown(f"""<div class="result-card" style="background-color:#f1f3f5; border-top: 2px solid #ced4da;">
    📉 전체 범죄 발생량과 CCTV 설치량 상관계수(r): <b>{r_1:.4f}</b>
    </div>""", unsafe_allow_html=True)

st.divider()

# --- 2번 섹션: 인터랙티브 버튼 UI ---
st.subheader("2️⃣ 범죄 유형별 상세 발생 추이 및 상관성 분석")
st.write("📌 아래 버튼을 눌러 분석할 범죄 유형을 선택하세요.")

if 'active_crimes' not in st.session_state:
    st.session_state.active_crimes = set()

cols = st.columns(len(crime_list))
for i, crime in enumerate(crime_list):
    is_active = crime in st.session_state.active_crimes
    label = f"🔵 {crime}" if is_active else f"⚪ {crime}"
    if cols[i].button(label, key=f"btn_{crime}", use_container_width=True):
        if is_active: st.session_state.active_crimes.remove(crime)
        else: st.session_state.active_crimes.add(crime)
        st.rerun()

if not st.session_state.active_crimes:
    st.info("범죄 유형 버튼을 클릭하면 분석 차트가 활성화됩니다.")
else:
    selected_list = list(st.session_state.active_crimes)
    df_sub = df_all_crime[df_all_crime['유형'].isin(selected_list)]
    df_sub_agg = df_sub.groupby('연도')['건수'].sum().reset_index().sort_values('연도')
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df_10yr_base['연도'], y=df_10yr_base['CCTV누적'], name="CCTV 인프라(배경)", 
                          marker_color='rgba(149, 165, 166, 0.4)'), secondary_y=False)
    for crime in selected_list:
        crime_data_single = df_sub[df_sub['유형'] == crime].sort_values('연도')
        fig2.add_trace(go.Scatter(x=crime_data_single['연도'], y=crime_data_single['건수'], 
                                  name=crime, mode='lines+markers', line=dict(width=3)), secondary_y=True)
    
    fig2.update_layout(template="plotly_white", hovermode="x unified", height=450, margin=dict(b=0))
    st.plotly_chart(fig2, use_container_width=True)

    r_2, _ = stats.pearsonr(df_10yr_base['CCTV누적'], df_sub_agg['건수'])
    st.markdown(f"""<div class="result-card" style="background-color:#e7f5ff; color:#1864ab; border-top: 2px solid #a5d8ff;">
        📊 <b>{', '.join(selected_list)}</b> 합계와 CCTV 설치량 상관계수(r): <b>{r_2:.4f}</b>
        </div>""", unsafe_allow_html=True)

st.divider()

# --- 3번 섹션 ---
st.subheader("3️⃣ 연도별 CCTV 인프라 및 실시간 탐지(신고 활용) 총량 분석")
st.caption("※ 2021년~2024년 시스템 운용 데이터 기준")

short_years = ['2021', '2022', '2023', '2024']
total_detect_vals = [df_util[df_util['연도'].astype(str).str.contains(yr)]['합계'].sum() for yr in short_years]
df_3 = df_10yr_base[df_10yr_base['연도'].isin(short_years)].copy()
df_3['전체탐지'] = total_detect_vals

fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Bar(x=df_3['연도'], y=df_3['CCTV누적'], name="CCTV 설치량", marker_color='#aed6f1'), secondary_y=False)
fig3.add_trace(go.Scatter(x=df_3['연도'], y=df_3['전체탐지'], name="실시간 탐지 실적", 
                          line=dict(color='#e74c3c', width=4, dash='dot'), marker=dict(size=12)), secondary_y=True)
fig3.update_layout(template="plotly_white", height=400, margin=dict(b=0))
st.plotly_chart(fig3, use_container_width=True)

r_3, _ = stats.pearsonr(df_3['CCTV누적'], df_3['전체탐지'])
st.markdown(f"""<div class="result-card" style="background-color:#fff5f5; color:#c92a2a; border-top: 2px solid #ffa8a8;">
    🚀 인프라 확충 대비 실시간 탐지 효율 상관계수(r): <b>{r_3:.4f}</b>
    </div>""", unsafe_allow_html=True)

# --- 📋 상세 분석 데이터 수치 (접이식 Expander 적용) ---
st.divider()
with st.expander("📋 상세 분석 데이터 수치 확인하기 (2015-2024)", expanded=False):
    st.write("분석에 사용된 연도별 CCTV 누적 대수와 5대 범죄 발생 합계 데이터입니다.")
    st.table(df_1.style.format({"CCTV누적": "{:,.0f}", "건수": "{:,.0f}"}))

# 데이터 출처 정보
st.info("""
**데이터 출처 및 분석 근거:**
- **CCTV 설치 현황**: `서울시 자치구 CCTV 설치현황.xlsx`
- **범죄 통계**: `crime(2015-2024).xlsx` (살인, 강도, 강간·추행, 절도, 폭력 합산)
- **탐지 효율**: `util_clean.xlsx` (CCTV 통합관제센터 실시간 대응 통계)
""")