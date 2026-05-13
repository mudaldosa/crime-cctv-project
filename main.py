import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 설치된 폰트 중 한글 지원 폰트 확인 및 설정
# Windows: 'Malgun Gothic', Mac: 'AppleGothic'
# Streamlit Cloud/Linux: 'NanumGothic' 또는 'DejaVu Sans'
def set_korean_font():
    try:
        # 시스템에 설치된 폰트 리스트에서 한글 폰트 찾기
        font_list = [f.name for f in fm.fontManager.ttflist]
        if 'Malgun Gothic' in font_list:
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif 'AppleGothic' in font_list:
            plt.rcParams['font.family'] = 'AppleGothic'
        elif 'NanumGothic' in font_list:
            plt.rcParams['font.family'] = 'NanumGothic'
        else:
            # 리눅스 서버 환경 등을 대비해 폰트 경로 직접 지정 필요할 수 있음
            st.warning("한글 폰트를 찾을 수 없어 텍스트가 깨질 수 있습니다.")
        
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        st.error(f"폰트 설정 중 오류 발생: {e}")
set_korean_font()

# 서울시 구역별/년도별 cctv 개수 데이터 로딩
cctv_raw = pd.read_excel('./DATA/서울시 자치구 (범죄예방 수사용) CCTV 설치현황(25.12.31 기준).xlsx', header=2)

# 쓸모 없는 칼럼 삭제
cctv = cctv_raw.drop(columns=['Unnamed: 0'])
cctv = cctv.iloc[:-2]
cctv = cctv[cctv['구분'] != '계']

# '순번' 컬럼을 인덱스로 설정
cctv = cctv.set_index('순번')

# 서울시 구역별/년도별 범죄횟수 데이터 로딩
crime_raw = pd.read_csv('./DATA/5대+범죄+발생현황_20260512140024.csv')


# ----------------------------------------------------------------------
# 2. 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_process_data():
    # [A] 범죄 데이터 정제
    crime_origin = pd.read_excel(r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\경찰청_범죄 발생 지역별 통계_20241231.xlsx")
    seoul_col = [col for col in crime_origin.columns if '서울' in col]
    crime_seoul = crime_origin[['범죄대분류', '범죄중분류'] + seoul_col]
    crime_seoul.columns = [col.replace('서울 ', '').strip() for col in crime_seoul.columns]
    
    gu_columns = [col for col in crime_seoul.columns if col not in ['범죄대분류', '범죄중분류']]
    crime_final = crime_seoul[gu_columns].sum().reset_index()
    crime_final.columns = ['지역구', '범죄수']

    # [B] CCTV 데이터 정제
    cctv_origin = pd.read_excel(r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\서울시 자치구 (범죄예방 수사용) CCTV 설치현황(25.12.31 기준).xlsx", skiprows=2)
    cctv_seoul = cctv_origin[cctv_origin['구분'] != '계'].iloc[:25].copy()
    cctv_seoul['구분'] = cctv_seoul['구분'].str.strip()
    cctv_2024 = cctv_seoul[['구분', '2024년']].copy()
    cctv_2024['2024년'] = pd.to_numeric(cctv_2024['2024년'].astype(str).str.replace(',', ''), errors='coerce')
    cctv_2024.columns = ['지역구', 'CCTV수']

    # [C] 인구 데이터 정제
    people_origin = pd.read_excel(r"C:\Users\Win11Pro\Documents\과제\cctv\DATA\등록인구_2024.xlsx")
    people = people_origin.iloc[:, [1, 3]].copy()
    people.columns = ['지역구', '인구수']
    people['지역구'] = people['지역구'].str.strip()
    people = people[people['지역구'].notna()]
    people = people[~people['지역구'].str.contains('합계|소계|동별')]
    people = people.iloc[:25]
    people['인구수'] = pd.to_numeric(people['인구수'].astype(str).str.replace(',', ''), errors='coerce')

    # [D] 데이터 병합 및 비율 계산
    df_merge = pd.merge(crime_final, cctv_2024, on='지역구')
    df_final = pd.merge(df_merge, people, on='지역구')
    df_final['범죄율'] = (df_final['범죄수'] / df_final['인구수']) * 10000
    
    # 인덱스를 1~25로 재설정
    df_final.index = range(1, len(df_final) + 1)
    
    return df_final

# 데이터 불러오기
df_final = load_and_process_data()
corr_total = df_final['CCTV수'].corr(df_final['범죄수'])

# ----------------------------------------------------------------------

# --- Streamlit UI 시작 ---
st.title("서울시 치안 인프라 및 범죄 현황 분석")

# --- 연도별 CCTV 증가량 분석 (전체 추이 집중) ---
# '년'으로 끝나는 모든 컬럼 추출 (연도 데이터 자동 필터링)
all_year_cols = [col for col in cctv.columns if col.endswith('년')]

# 데이터 숫자 변환 (누적 대수 데이터)
cctv_years = cctv[all_year_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# --- 연도별 CCTV 분석 (Tabs 분할) ---
tab1, tab2 = st.tabs(["📊 연도별 누적 설치 현황", "⏳ 연도별 신규 설치 추이"])

# 데이터 계산 부분 (중복 방지)
all_year_cols = [col for col in cctv.columns if col.endswith('년')]
cctv_years = cctv[all_year_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# --- Tab 1: 누적 합계 (막대 그래프) ---
with tab1:
    st.subheader("서울시 CCTV 누적 설치 총량")
    total_cumulative = cctv_years.sum()
    
    fig_cum, ax_cum = plt.subplots(figsize=(12, 6))
    sns.barplot(x=total_cumulative.index, y=total_cumulative.values, palette='Blues_d', ax=ax_cum)
    
    ax_cum.set_title('연도별 누적 대수 현황', fontsize=15)
    ax_cum.set_ylabel('총 설치 대수 (대)')
    
    for i, val in enumerate(total_cumulative.values):
        ax_cum.text(i, val + (total_cumulative.max() * 0.01), f'{int(val):,}대', 
                    ha='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_cum)

# --- Tab 2: 신규 설치 (꺾은선 그래프) ---
with tab2:
    st.subheader("서울시 연도별 신규 설치 추이")
    cctv_increase = cctv_years.diff(axis=1).iloc[:, 1:] 
    total_increase = cctv_increase.sum()
    
    fig_inc, ax_inc = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=total_increase.index, y=total_increase.values, 
                 marker='o', linewidth=3, color='forestgreen', ax=ax_inc)
    
    ax_inc.set_title('연도별 신규 설치량 추이', fontsize=15)
    ax_inc.set_ylabel('신규 설치 대수 (대)')
    ax_inc.grid(True, linestyle='--', alpha=0.5)
    
    for i, val in enumerate(total_increase.values):
        ax_inc.text(i, val + (total_increase.max() * 0.02), f'{int(val):,}대', 
                    ha='center', fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    st.pyplot(fig_inc)


st.divider()

# 섹션 0: 데이터 요약표 (사용자 요청 유지)
with st.expander("🔍 데이터 요약 표 보기 (종로구 ~ 강동구)", expanded=True):
    st.dataframe(df_final, use_container_width=True)

st.markdown("---")

# --- 섹션 1: 범죄 발생 건수 분석 ---
st.header("1️⃣ 범죄 발생 건수(절대량) 분석")

# 그래프 바로 위에 토글 버튼 배치
view_mode_1 = st.segmented_control(
    "그래프 형태 선택",
    options=["📊 추이 분석", "📈 상관관계"],
    default="📊 추이 분석",
    key="view_1"
)

if view_mode_1 == "📊 추이 분석":
    st.subheader("CCTV 설치 규모에 따른 범죄수 변화")
    df_sorted = df_final.sort_values(by='CCTV수')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(df_sorted['지역구'], df_sorted['CCTV수'], color='lightgray', alpha=0.6)
    ax1.set_ylabel('CCTV 설치 수 (대)', color='gray')
    ax2 = ax1.twinx()
    ax2.plot(df_sorted['지역구'], df_sorted['범죄수'], color='blue', marker='s', linewidth=2)
    ax2.set_ylabel('범죄 발생 건수 (건)', color='blue')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
else:
    st.subheader("CCTV 수와 범죄 발생 건수 상관관계")
    fig_corr = plt.figure(figsize=(10, 6))
    sns.regplot(data=df_final, x='CCTV수', y='범죄수', scatter_kws={'s': 60}, line_kws={'color': 'red'})
    for i in range(len(df_final)):
        plt.text(df_final['CCTV수'].iloc[i]+50, df_final['범죄수'].iloc[i]+50, df_final['지역구'].iloc[i], fontsize=9)
    plt.title(f'상관계수: {corr_total:.4f}')
    st.pyplot(fig_corr)

st.markdown("---")

corr_total = df_final['CCTV수'].corr(df_final['범죄율'])

# 년도별 총합 정렬
# 컬럼명에 '.'이 포함되지 않은 컬럼만 선택하여 슬라이싱
crime_filtered = crime_raw[[col for col in crime_raw.columns if '.' not in col]]

# 불필요한 첫 번째 행(단위 등) 제거
crime_filtered = crime_filtered.drop(0)

# 컬럼명 정리 (자치구별(2) -> 자치구)
crime_filtered.rename(columns={'자치구별(2)': '자치구'}, inplace=True)

# 4. '자치구별(1)' 컬럼이 남아있다면 삭제
if '자치구별(1)' in crime_filtered.columns: crime_filtered.drop(columns=['자치구별(1)'], inplace=True)

# 전체 인구수
pop_raw = pd.read_excel('./DATA/등록인구_20260512150716.xlsx')

# 데이터 정제 (위의 로직 그대로 사용하되 변수명 등은 기존 흐름 유지)
pop_24 = pop_raw[['동별(2)', '2024']].copy()
pop_24.columns = ['자치구', '인구수']
pop_24 = pop_24[~pop_24['자치구'].isin(['소계', '합계', '동별(2)'])]
pop_24['인구수'] = pd.to_numeric(pop_24['인구수'].astype(str).str.replace(',', ''), errors='coerce')
pop_24 = pop_24.dropna(subset=['인구수'])
pop_24['인구수'] = pop_24['인구수'].astype(int)

pop_24_sorted = pop_24.sort_values(by='인구수', ascending=False)

# --- 데이터 병합 및 추가 지표 계산 (2024년 기준) ---
cctv_24 = cctv[['구분', '2024년']].rename(columns={'구분': '자치구', '2024년': 'CCTV수'})
crime_24 = crime_filtered[['자치구', '2024']].rename(columns={'2024': '발생건수'})

# 숫자 변환
cctv_24['CCTV수'] = pd.to_numeric(cctv_24['CCTV수'], errors='coerce')
crime_24['발생건수'] = pd.to_numeric(crime_24['발생건수'], errors='coerce')

# 병합: 자치구 기준 (인구 + CCTV + 범죄)
df_final = pd.merge(pop_24, cctv_24, on='자치구')
df_final = pd.merge(df_final, crime_24, on='자치구')

# 범죄율 계산 (인구 10만명당)
df_final['범죄율'] = (df_final['발생건수'] / df_final['인구수']) * 100000

# 자치구 순서 고정 (발생건수 내림차순)
df_sorted = df_final.sort_values(by='발생건수', ascending=False)
order = df_sorted['자치구'].tolist()


# 1. 인구 현황 그래프
st.header("서울시 자치구별 인구 현황")
fig_pop, ax_p = plt.subplots(figsize=(15, 6))
sns.barplot(x='자치구', y='인구수', data=pop_24_sorted, palette='coolwarm', ax=ax_p)
ax_p.set_title('2024년 서울시 자치구별 등록인구 현황 (내림차순)', fontsize=15)
ax_p.tick_params(axis='x', rotation=45)
for i, val in enumerate(pop_24_sorted['인구수']):
    ax_p.text(i, val + 5000, f'{val/10000:.1f}만', ha='center', fontsize=9, fontweight='bold')
st.pyplot(fig_pop)

# --------------------------------------------------------
df_final.columns = ['지역구','범죄수','CCTV수','인구수','범죄율']

# --- 섹션 2: 인구 대비 범죄율 분석 ---
st.header("2️⃣ 인구 대비 범죄율(비율) 분석")

# 그래프 바로 위에 토글 버튼 배치
view_mode_2 = st.segmented_control(
    "그래프 형태 선택",
    options=["📊 추이 분석", "📈 상관관계"],
    default="📊 추이 분석",
    key="view_2"
)

if view_mode_2 == "📊 추이 분석":
    st.subheader("CCTV 설치 규모에 따른 인구당 범죄율 변화")
    # 컬럼명 앞뒤 공백을 완전히 제거한 후 정렬
    df_final.columns = df_final.columns.str.strip() 
    df_sorted = df_final.sort_values(by='CCTV수')
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    ax3.bar(df_sorted['지역구'], df_sorted['CCTV수'], color='skyblue', alpha=0.7)
    ax3.set_ylabel('CCTV 설치 수 (대)', color='skyblue')
    ax4 = ax3.twinx()
    ax4.plot(df_sorted['지역구'], df_sorted['범죄율'], color='red', marker='o', linewidth=2)
    ax4.set_ylabel('인구 만 명당 범죄율 (건)', color='red')
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)
else:
    st.subheader("CCTV 수와 인구 대비 범죄율 상관관계")
    fig_corr2 = plt.figure(figsize=(10, 6))
    sns.regplot(data=df_final, x='CCTV수', y='범죄율', scatter_kws={'s': 60, 'alpha': 0.6}, line_kws={'color': 'red'})
    for i in range(len(df_final)):
        plt.text(df_final['CCTV수'].iloc[i]+50, df_final['범죄율'].iloc[i]+0.1, df_final['지역구'].iloc[i], fontsize=9)
    plt.title(f'상관계수: {corr_total:.4f}')
    st.pyplot(fig_corr2)
    
st.divider()

years = [str(y) for y in range(2015, 2025)]
crime_types = ['살인', '강도', '강간', '절도', '폭력']
data_list = []

for year in years:
    # 1. 인구 데이터 (소계)
    pop_total = pd.to_numeric(str(pop_raw.loc[pop_raw['동별(2)'] == '소계', year].values[0]).replace(',', ''))
    
    # 2. CCTV 총량
    cctv_sum = cctv[f'{year}년'].sum()
    
    # 3. 5대 범죄별 발생 건수 (데이터프레임 내 컬럼 위치를 기준으로 순서대로 추출)
    # 데이터 구조에 따라 '발생' 컬럼들을 순회합니다.
    # 예시: 합계 행에서 각 범죄별 발생 수치 추출 (실제 CSV 컬럼 인덱스 확인 필요)
    # 여기서는 각 범죄별 '발생' 데이터 위치를 변수로 지정하여 가져옵니다.
    row_sum = crime_raw[crime_raw['자치구별(1)'] == '합계']
    
    # 각 범죄별 발생 건수 (데이터의 고정된 오프셋 사용 또는 컬럼명 매칭)
    # 데이터 구조상 '합계' 행의 연도별 첫 번째는 총합, 그 뒤로 5대 범죄가 이어짐
    occ_list = []
    # 2015, 2015.1, 2015.2 ... 순서에서 발생(짝수 인덱스)만 추출
    # 실제 데이터 구조에 맞춰 조정이 필요할 수 있습니다.
    for i in range(1, 6): # 살인~폭력 5개 항목
        val = pd.to_numeric(str(row_sum[f'{year}.{i*2}'].values[0]).replace(',', ''))
        occ_list.append((val / pop_total) * 100000) # 인구 10만명당 발생률

    data_list.append([year, cctv_sum] + occ_list)

df_crime_trend = pd.DataFrame(data_list, columns=['연도', 'CCTV수량'] + crime_types)
                
st.divider()








