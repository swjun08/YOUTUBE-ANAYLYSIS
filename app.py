import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import platform
warnings.filterwarnings('ignore')

# 한글 폰트 설정
def set_korean_font():
import matplotlib.font_manager as fm

system = platform.system()

if system == 'Windows':
font_name = 'Malgun Gothic'
elif system == 'Darwin':  # macOS
font_name = 'AppleGothic'
    else:  # Linux
    else:  # Linux (Streamlit Cloud)
        # Streamlit Cloud용 폰트 설정
        font_dirs = ['/usr/share/fonts/truetype/nanum']
        font_files = fm.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            fm.fontManager.addfont(font_file)
font_name = 'NanumGothic'

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 폰트 설정 실행
set_korean_font()

# 페이지 설정
st.set_page_config(
page_title="YouTube 영상 상관관계 분석",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
   <style>
   .main {
       padding: 0rem 1rem;
   }
   .stButton>button {
       width: 100%;
       background-color: #FF0000;
       color: white;
       font-weight: bold;
       border-radius: 10px;
       padding: 0.5rem 1rem;
       border: none;
       transition: all 0.3s;
   }
   .stButton>button:hover {
       background-color: #CC0000;
       border: none;
       transform: scale(1.02);
   }
   .metric-card {
       background-color: #f0f2f6;
       padding: 1.5rem;
       border-radius: 10px;
       text-align: center;
   }
   h1 {
       color: #FF0000;
   }
   h2, h3 {
       color: #282828;
   }
   .stTabs [data-baseweb="tab-list"] {
       gap: 8px;
   }
   .stTabs [data-baseweb="tab"] {
       height: 50px;
       background-color: #f0f2f6;
       border-radius: 10px 10px 0 0;
       padding: 10px 20px;
       font-weight: bold;
   }
   .stTabs [aria-selected="true"] {
       background-color: #FF0000;
       color: white;
   }
   </style>
   """, unsafe_allow_html=True)

# 데이터 전처리 함수
@st.cache_data
def load_and_process_data(uploaded_file):
df = pd.read_csv(uploaded_file)

# 원본 데이터 크기 저장
original_size = len(df)

# 컬럼명 매핑 (원본 데이터셋의 컬럼명에 맞춤)
column_mapping = {
'Video Title': 'title',
'Video Views': 'views',
'Like_count': 'likes',
'comment_count': 'comment_count'
}

# 필요한 컬럼 확인
available_cols = []
for old_name, new_name in column_mapping.items():
if old_name in df.columns:
available_cols.append(old_name)

if len(available_cols) < 4:
st.error(f"⚠️ 필요한 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")
return None, None

# 컬럼명 변경
df = df.rename(columns=column_mapping)

# 필요한 컬럼만 선택
df = df[['title', 'views', 'likes', 'comment_count']].copy()

# 전처리 통계 저장
preprocessing_stats = {
'original_size': original_size,
'missing_values': {},
'outliers_removed': {},
'final_size': 0
}

# 1. 결측치 처리
st.info("🔄 1단계: 결측치 처리 중...")

# 결측치 확인
missing_before = df.isnull().sum()
preprocessing_stats['missing_values']['before'] = missing_before.to_dict()

# title이 없는 행 제거
df = df.dropna(subset=['title'])

# 숫자형 컬럼의 결측치를 중앙값으로 대체
numeric_cols = ['views', 'likes', 'comment_count']
for col in numeric_cols:
df[col] = pd.to_numeric(df[col], errors='coerce')
if df[col].isnull().sum() > 0:
median_val = df[col].median()
df[col].fillna(median_val, inplace=True)

missing_after = df.isnull().sum()
preprocessing_stats['missing_values']['after'] = missing_after.to_dict()
preprocessing_stats['missing_values']['removed'] = original_size - len(df)

# 2. title_length 파생 변수 생성
df['title_length'] = df['title'].str.len()

# 3. 이상치 처리 (IQR 방법)
st.info("🔄 2단계: 이상치 처리 중...")

size_before_outlier = len(df)

for col in ['views', 'likes', 'comment_count', 'title_length']:
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 제거 전 개수
outliers_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
preprocessing_stats['outliers_removed'][col] = outliers_count

# 이상치 제거
df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

preprocessing_stats['outliers_removed']['total'] = size_before_outlier - len(df)

# 4. 데이터 정규화 (Min-Max Scaling)
st.info("🔄 3단계: 데이터 정규화 중...")

df_normalized = df.copy()

for col in ['views', 'likes', 'comment_count', 'title_length']:
min_val = df[col].min()
max_val = df[col].max()
df_normalized[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)

preprocessing_stats['final_size'] = len(df)
preprocessing_stats['removed_percentage'] = ((original_size - len(df)) / original_size) * 100

return df, preprocessing_stats

# 상관관계 계산 함수
def get_top_correlations(df, n=1):
corr_cols = ['title_length', 'views', 'likes', 'comment_count']
corr_matrix = df[corr_cols].corr()

# 상관계수를 리스트로 변환
corr_pairs = []
for i in range(len(corr_matrix.columns)):
for j in range(i+1, len(corr_matrix.columns)):
corr_pairs.append({
'var1': corr_matrix.columns[i],
'var2': corr_matrix.columns[j],
'correlation': corr_matrix.iloc[i, j]
})

corr_df = pd.DataFrame(corr_pairs)

# 양의 상관관계 TOP
positive = corr_df.nlargest(n, 'correlation')

# 음의 상관관계 TOP
negative = corr_df.nsmallest(n, 'correlation')

return positive, negative, corr_matrix

# 산점도 그리기 함수
def plot_scatter(df, var1, var2, corr_value):
fig, ax = plt.subplots(figsize=(10, 6))

# 샘플링
if len(df) > 5000:
df_plot = df.sample(5000, random_state=42)
else:
df_plot = df

ax.scatter(df_plot[var1], df_plot[var2], alpha=0.5, s=30, color='#FF0000')

# 회귀선 추가
z = np.polyfit(df_plot[var1], df_plot[var2], 1)
p = np.poly1d(z)
ax.plot(df_plot[var1], p(df_plot[var1]), "b--", linewidth=2, label='추세선')

# 한글 변수명 매핑
var_names = {
'title_length': '제목 길이',
'views': '조회수',
'likes': '좋아요 수',
'comment_count': '댓글 수'
}

ax.set_xlabel(var_names.get(var1, var1), fontsize=12, fontweight='bold')
ax.set_ylabel(var_names.get(var2, var2), fontsize=12, fontweight='bold')
ax.set_title(f'{var_names.get(var1, var1)} vs {var_names.get(var2, var2)}\n상관계수: {corr_value:.4f}', 
fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

return fig

# 메인 앱
def main():
# 헤더
st.title("📊 YouTube 영상 상관관계 분석 대시보드")
st.markdown("### 제목 길이와 조회수, 좋아요, 댓글 수의 관계를 분석합니다")
st.markdown("---")

# 사이드바
with st.sidebar:
st.header("📁 데이터 업로드")
uploaded_file = st.file_uploader(
"CSV 파일을 업로드하세요",
type=['csv'],
help="Video Title, Video Views, Like_count, comment_count 컬럼이 필요합니다"
)

st.markdown("---")

# Kaggle 데이터셋 링크
st.markdown("### 📦 데이터셋 다운로드")
st.markdown("""
       데이터가 없으신가요?  
       아래 링크에서 다운로드하세요!
       """)
st.link_button(
"🔗 Kaggle 데이터셋 다운로드",
"https://www.kaggle.com/datasets/hamza3692/youtube-video-statistics-and-subtitles-dataset?resource=download",
use_container_width=True
)
st.caption("💡 Kaggle 계정이 필요합니다")

st.markdown("---")
st.markdown("### 📌 필수 컬럼")
st.markdown("""
       - `Video Title` (영상 제목)
       - `Video Views` (조회수)
       - `Like_count` (좋아요 수)
       - `comment_count` (댓글 수)
       """)

st.markdown("---")
st.markdown("### 📖 사용 가이드")
st.markdown("""
       1. **개요**: 전체 데이터 요약
       2. **상관관계 분석**: 히트맵 & 상세 분석
       3. **제목 길이 분석**: 구간별 성과
       4. **Top 영상**: 인기 영상 순위
       """)

# 데이터 로드
if uploaded_file is not None:
with st.spinner('데이터를 로딩 및 전처리 중입니다...'):
df, preprocessing_stats = load_and_process_data(uploaded_file)

if df is not None:
st.success(f"✅ 데이터 전처리 완료! (최종 {len(df):,}개 영상)")

# 전처리 요약 표시
with st.expander("🔧 데이터 전처리 요약 보기", expanded=False):
col1, col2, col3 = st.columns(3)

with col1:
st.metric("원본 데이터", f"{preprocessing_stats['original_size']:,}개")
st.metric("결측치 제거", f"{preprocessing_stats['missing_values']['removed']:,}개")

with col2:
st.metric("이상치 제거", f"{preprocessing_stats['outliers_removed']['total']:,}개")
st.metric("최종 데이터", f"{preprocessing_stats['final_size']:,}개")

with col3:
st.metric("제거 비율", f"{preprocessing_stats['removed_percentage']:.1f}%")
st.metric("정규화", "Min-Max Scaling ✓")

st.markdown("---")
st.markdown("#### 📊 전처리 세부 사항")

# 결측치 처리 상세
st.markdown("**1️⃣ 결측치 처리**")
missing_df = pd.DataFrame({
'컬럼': preprocessing_stats['missing_values']['before'].keys(),
'결측치 (전)': preprocessing_stats['missing_values']['before'].values(),
'결측치 (후)': preprocessing_stats['missing_values']['after'].values()
})
st.dataframe(missing_df, use_container_width=True, hide_index=True)
st.caption("💡 title 결측치는 제거, 숫자형 결측치는 중앙값으로 대체")

# 이상치 제거 상세
st.markdown("**2️⃣ 이상치 제거 (IQR 방법)**")
outlier_df = pd.DataFrame({
'변수': [k for k in preprocessing_stats['outliers_removed'].keys() if k != 'total'],
'제거된 이상치': [v for k, v in preprocessing_stats['outliers_removed'].items() if k != 'total']
})
st.dataframe(outlier_df, use_container_width=True, hide_index=True)
st.caption("💡 IQR 기준: Q1 - 1.5×IQR ~ Q3 + 1.5×IQR 범위 밖 데이터 제거")

# 정규화 설명
st.markdown("**3️⃣ 데이터 정규화 (Min-Max Scaling)**")
st.code("normalized_value = (value - min) / (max - min)")
st.caption("💡 모든 수치형 변수를 0~1 범위로 정규화하여 비교 가능하게 함")

# 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
"📈 개요", 
"🔍 상관관계 분석", 
"📏 제목 길이 분석", 
"🏆 Top 영상",
"🔧 전처리 상세"
])

# ==================== TAB 1: 개요 ====================
with tab1:
st.markdown("## 📊 주요 통계 요약")

col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric("총 영상 수", f"{len(df):,}")
with col2:
st.metric("평균 조회수", f"{df['views'].mean():,.0f}")
with col3:
st.metric("평균 좋아요", f"{df['likes'].mean():,.0f}")
with col4:
st.metric("평균 제목 길이", f"{df['title_length'].mean():.1f}자")

st.markdown("---")

# 분포 그래프
st.markdown("### 📊 데이터 분포")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 조회수 분포
axes[0, 0].hist(df['views'], bins=50, color='#FF0000', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('조회수 분포', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('조회수')
axes[0, 0].set_ylabel('빈도')
axes[0, 0].grid(True, alpha=0.3)

# 좋아요 분포
axes[0, 1].hist(df['likes'], bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('좋아요 수 분포', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('좋아요 수')
axes[0, 1].set_ylabel('빈도')
axes[0, 1].grid(True, alpha=0.3)

# 댓글 수 분포
axes[1, 0].hist(df['comment_count'], bins=50, color='#FFA07A', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('댓글 수 분포', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('댓글 수')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].grid(True, alpha=0.3)

# 제목 길이 분포
axes[1, 1].hist(df['title_length'], bins=50, color='#4169E1', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('제목 길이 분포', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('제목 길이 (글자 수)')
axes[1, 1].set_ylabel('빈도')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# 데이터 미리보기
with st.expander("📋 데이터 미리보기 (10개)"):
st.dataframe(df.head(10), use_container_width=True)

# ==================== TAB 2: 상관관계 분석 ====================
with tab2:
st.markdown("## 🔥 상관관계 히트맵")

positive, negative, corr_matrix = get_top_correlations(df)

fig, ax = plt.subplots(figsize=(10, 8))

# 한글 라벨
corr_matrix_kr = corr_matrix.copy()
corr_matrix_kr.columns = ['제목 길이', '조회수', '좋아요 수', '댓글 수']
corr_matrix_kr.index = ['제목 길이', '조회수', '좋아요 수', '댓글 수']

sns.heatmap(corr_matrix_kr, annot=True, fmt='.3f', cmap='RdYlBu_r', 
center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
ax=ax, vmin=-1, vmax=1)
ax.set_title('변수 간 상관관계 히트맵', fontsize=16, fontweight='bold', pad=20)
st.pyplot(fig)

st.markdown("---")

# 상관관계 버튼
st.markdown("## 🔍 상관관계 상세 분석")
st.markdown("버튼을 클릭하여 가장 강한 양의/음의 상관관계를 확인하세요!")

col1, col2 = st.columns(2)

with col1:
if st.button("📈 양의 상관관계 TOP 1 보기", use_container_width=True):
st.session_state['show_positive'] = True
st.session_state['show_negative'] = False

with col2:
if st.button("📉 음의 상관관계 TOP 1 보기", use_container_width=True):
st.session_state['show_negative'] = True
st.session_state['show_positive'] = False

# 양의 상관관계 표시
if st.session_state.get('show_positive', False):
st.markdown("### 📈 가장 강한 양의 상관관계")
top_pos = positive.iloc[0]

var_names = {
'title_length': '제목 길이',
'views': '조회수',
'likes': '좋아요 수',
'comment_count': '댓글 수'
}

st.info(f"**{var_names[top_pos['var1']]}** ↔ **{var_names[top_pos['var2']]}** : 상관계수 = **{top_pos['correlation']:.4f}**")

fig = plot_scatter(df, top_pos['var1'], top_pos['var2'], top_pos['correlation'])
st.pyplot(fig)

# 해석
if top_pos['correlation'] > 0.7:
st.success("💡 **해석**: 매우 강한 양의 상관관계입니다. 한 변수가 증가하면 다른 변수도 크게 증가하는 경향이 있습니다.")
elif top_pos['correlation'] > 0.4:
st.success("💡 **해석**: 중간 정도의 양의 상관관계입니다. 한 변수가 증가하면 다른 변수도 증가하는 경향이 있습니다.")
else:
st.success("💡 **해석**: 약한 양의 상관관계입니다. 두 변수 간 관계가 크지 않습니다.")

# 음의 상관관계 표시
if st.session_state.get('show_negative', False):
st.markdown("### 📉 가장 강한 음의 상관관계")
top_neg = negative.iloc[0]

var_names = {
'title_length': '제목 길이',
'views': '조회수',
'likes': '좋아요 수',
'comment_count': '댓글 수'
}

st.warning(f"**{var_names[top_neg['var1']]}** ↔ **{var_names[top_neg['var2']]}** : 상관계수 = **{top_neg['correlation']:.4f}**")

fig = plot_scatter(df, top_neg['var1'], top_neg['var2'], top_neg['correlation'])
st.pyplot(fig)

# 해석
if top_neg['correlation'] < -0.7:
st.warning("💡 **해석**: 매우 강한 음의 상관관계입니다. 한 변수가 증가하면 다른 변수는 크게 감소하는 경향이 있습니다.")
elif top_neg['correlation'] < -0.4:
st.warning("💡 **해석**: 중간 정도의 음의 상관관계입니다. 한 변수가 증가하면 다른 변수는 감소하는 경향이 있습니다.")
else:
st.warning("💡 **해석**: 약한 음의 상관관계입니다. 두 변수 간 역관계가 크지 않습니다.")

# ==================== TAB 3: 제목 길이 분석 ====================
with tab3:
st.markdown("## 📏 제목 길이와 조회수의 관계")
st.markdown("**프로젝트의 핵심 질문**: 제목 길이가 조회수에 영향을 미칠까?")

# 제목 길이 vs 조회수 산점도
st.markdown("### 📊 제목 길이 vs 조회수 산점도")

# 상관계수 계산
corr_title_views = df['title_length'].corr(df['views'])

fig, ax = plt.subplots(figsize=(12, 7))

# 샘플링 (너무 많으면)
if len(df) > 5000:
df_plot = df.sample(5000, random_state=42)
else:
df_plot = df

# 산점도
scatter = ax.scatter(df_plot['title_length'], df_plot['views'], 
alpha=0.5, s=50, c=df_plot['views'], 
cmap='YlOrRd', edgecolors='black', linewidth=0.5)

# 추세선
z = np.polyfit(df_plot['title_length'], df_plot['views'], 1)
p = np.poly1d(z)
ax.plot(df_plot['title_length'].sort_values(), 
p(df_plot['title_length'].sort_values()), 
"b--", linewidth=3, label=f'추세선 (기울기: {z[0]:.2f})')

ax.set_xlabel('제목 길이 (글자 수)', fontsize=14, fontweight='bold')
ax.set_ylabel('조회수', fontsize=14, fontweight='bold')
ax.set_title(f'제목 길이와 조회수의 관계\n상관계수: {corr_title_views:.4f}', 
fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# 컬러바
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('조회수', fontsize=12, fontweight='bold')

plt.tight_layout()
st.pyplot(fig)

# 해석
col1, col2 = st.columns([2, 1])

with col1:
if corr_title_views > 0.3:
st.success(f"💡 **분석 결과**: 제목 길이와 조회수 사이에 **양의 상관관계** (상관계수: {corr_title_views:.4f})가 있습니다. "
f"제목이 길수록 조회수가 증가하는 경향이 있습니다.")
elif corr_title_views < -0.3:
st.warning(f"💡 **분석 결과**: 제목 길이와 조회수 사이에 **음의 상관관계** (상관계수: {corr_title_views:.4f})가 있습니다. "
f"제목이 짧을수록 조회수가 증가하는 경향이 있습니다.")
else:
st.info(f"💡 **분석 결과**: 제목 길이와 조회수 사이에 **약한 상관관계** (상관계수: {corr_title_views:.4f})가 있습니다. "
f"제목 길이만으로는 조회수를 예측하기 어렵습니다.")

with col2:
st.metric("상관계수", f"{corr_title_views:.4f}")
st.metric("최적 제목 길이", f"{df.loc[df['views'].idxmax(), 'title_length']:.0f}자")

st.markdown("---")

# 구간별 분석
st.markdown("### 📊 제목 길이 구간별 평균 성과")
st.markdown("제목 길이를 구간별로 나누어 평균 성과를 비교합니다.")

# 제목 길이 구간 생성
bins = [0, 20, 40, 60, 80, 100, 200]
labels = ['0-20자', '21-40자', '41-60자', '61-80자', '81-100자', '100자+']
df['length_group'] = pd.cut(df['title_length'], bins=bins, labels=labels, include_lowest=True)

# 구간별 평균 계산
group_stats = df.groupby('length_group', observed=True)[['views', 'likes', 'comment_count']].mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = ['views', 'likes', 'comment_count']
titles = ['평균 조회수', '평균 좋아요', '평균 댓글 수']
colors = ['#FF0000', '#FF6B6B', '#FFA07A']

for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
axes[i].bar(range(len(group_stats)), group_stats[metric], color=color, alpha=0.8)
axes[i].set_xticks(range(len(group_stats)))
axes[i].set_xticklabels(group_stats.index, rotation=45)
axes[i].set_title(title, fontsize=14, fontweight='bold')
axes[i].set_xlabel('제목 길이 구간', fontsize=11)
axes[i].set_ylabel(title, fontsize=11)
axes[i].grid(True, alpha=0.3, axis='y')

# 값 표시
for j, v in enumerate(group_stats[metric]):
axes[i].text(j, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# 통계 테이블
st.markdown("### 📊 구간별 상세 통계")

group_stats_display = group_stats.copy()
group_stats_display['영상 수'] = df.groupby('length_group', observed=True).size()
group_stats_display = group_stats_display[['영상 수', 'views', 'likes', 'comment_count']]
group_stats_display.columns = ['영상 수', '평균 조회수', '평균 좋아요', '평균 댓글 수']

# 숫자 포맷팅
group_stats_display['평균 조회수'] = group_stats_display['평균 조회수'].apply(lambda x: f"{x:,.0f}")
group_stats_display['평균 좋아요'] = group_stats_display['평균 좋아요'].apply(lambda x: f"{x:,.0f}")
group_stats_display['평균 댓글 수'] = group_stats_display['평균 댓글 수'].apply(lambda x: f"{x:,.0f}")

st.dataframe(group_stats_display, use_container_width=True)

st.markdown("---")

# 최적 제목 길이 제안
st.markdown("### 💡 최적 제목 길이 제안")

# 조회수가 높은 영상들의 평균 제목 길이
top_20_percent = df.nlargest(int(len(df) * 0.2), 'views')
optimal_length = top_20_percent['title_length'].mean()

col1, col2, col3 = st.columns(3)

with col1:
st.metric("상위 20% 영상의 평균 제목 길이", f"{optimal_length:.0f}자")
with col2:
st.metric("전체 영상 평균 제목 길이", f"{df['title_length'].mean():.0f}자")
with col3:
diff = optimal_length - df['title_length'].mean()
st.metric("차이", f"{diff:+.0f}자", delta=f"{(diff/df['title_length'].mean()*100):.1f}%")

if diff > 5:
st.success(f"✅ **제안**: 인기 영상들은 평균보다 **{diff:.0f}자 더 긴** 제목을 사용합니다. "
f"제목 길이를 **{optimal_length:.0f}자** 정도로 작성하는 것을 고려해보세요!")
elif diff < -5:
st.success(f"✅ **제안**: 인기 영상들은 평균보다 **{abs(diff):.0f}자 더 짧은** 제목을 사용합니다. "
f"제목 길이를 **{optimal_length:.0f}자** 정도로 작성하는 것을 고려해보세요!")
else:
st.info(f"ℹ️ **제안**: 인기 영상과 전체 평균의 제목 길이 차이가 크지 않습니다. "
f"제목 길이보다는 다른 요소들이 더 중요할 수 있습니다.")

# ==================== TAB 4: Top 영상 ====================
with tab4:
st.markdown("## 🏆 인기 영상 랭킹")

# 정렬 기준 선택
sort_option = st.selectbox(
"정렬 기준을 선택하세요:",
["조회수", "좋아요 수", "댓글 수"]
)

sort_mapping = {
"조회수": "views",
"좋아요 수": "likes",
"댓글 수": "comment_count"
}

sort_col = sort_mapping[sort_option]

st.markdown(f"### 🥇 {sort_option} 기준 Top 10")

top10 = df.nlargest(10, sort_col)[['title', 'title_length', 'views', 'likes', 'comment_count']].copy()
top10.columns = ['제목', '제목 길이', '조회수', '좋아요 수', '댓글 수']

# 숫자 포맷팅
top10['조회수'] = top10['조회수'].apply(lambda x: f"{x:,.0f}")
top10['좋아요 수'] = top10['좋아요 수'].apply(lambda x: f"{x:,.0f}")
top10['댓글 수'] = top10['댓글 수'].apply(lambda x: f"{x:,.0f}")

top10.index = range(1, 11)

st.dataframe(top10, use_container_width=True)

st.markdown("---")

# 제목 길이 비교
st.markdown("### 📏 제목 길이 극단 비교")

col1, col2 = st.columns(2)

with col1:
st.markdown("#### 📉 제목이 가장 짧은 영상 Top 5")
shortest = df.nsmallest(5, 'title_length')[['title', 'title_length', 'views', 'likes', 'comment_count']].copy()
shortest.columns = ['제목', '제목 길이', '조회수', '좋아요', '댓글']
shortest['조회수'] = shortest['조회수'].apply(lambda x: f"{x:,.0f}")
shortest['좋아요'] = shortest['좋아요'].apply(lambda x: f"{x:,.0f}")
shortest['댓글'] = shortest['댓글'].apply(lambda x: f"{x:,.0f}")
st.dataframe(shortest, use_container_width=True)

with col2:
st.markdown("#### 📈 제목이 가장 긴 영상 Top 5")
longest = df.nlargest(5, 'title_length')[['title', 'title_length', 'views', 'likes', 'comment_count']].copy()
longest.columns = ['제목', '제목 길이', '조회수', '좋아요', '댓글']
longest['조회수'] = longest['조회수'].apply(lambda x: f"{x:,.0f}")
longest['좋아요'] = longest['좋아요'].apply(lambda x: f"{x:,.0f}")
longest['댓글'] = longest['댓글'].apply(lambda x: f"{x:,.0f}")
st.dataframe(longest, use_container_width=True)

# ==================== TAB 5: 전처리 상세 ====================
with tab5:
st.markdown("## 🔧 데이터 전처리 상세 분석")
st.markdown("데이터 품질 향상을 위한 전처리 과정을 상세히 확인합니다.")

# 전체 요약
st.markdown("### 📊 전처리 전후 비교")

col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric(
"원본 데이터", 
f"{preprocessing_stats['original_size']:,}개",
help="업로드한 CSV 파일의 전체 행 수"
)

with col2:
st.metric(
"결측치 제거", 
f"-{preprocessing_stats['missing_values']['removed']:,}개",
delta=f"-{(preprocessing_stats['missing_values']['removed']/preprocessing_stats['original_size']*100):.1f}%",
delta_color="inverse",
help="결측치가 있는 행 제거"
)

with col3:
st.metric(
"이상치 제거", 
f"-{preprocessing_stats['outliers_removed']['total']:,}개",
delta=f"-{(preprocessing_stats['outliers_removed']['total']/preprocessing_stats['original_size']*100):.1f}%",
delta_color="inverse",
help="IQR 기준 이상치 제거"
)

with col4:
st.metric(
"최종 데이터", 
f"{preprocessing_stats['final_size']:,}개",
delta=f"{(preprocessing_stats['final_size']/preprocessing_stats['original_size']*100):.1f}%",
help="전처리 완료 후 최종 데이터"
)

st.markdown("---")

# 1. 결측치 분석
st.markdown("### 1️⃣ 결측치 처리")

col1, col2 = st.columns([3, 2])

with col1:
st.markdown("#### 📋 컬럼별 결측치 현황")
missing_df = pd.DataFrame({
'컬럼': preprocessing_stats['missing_values']['before'].keys(),
'결측치 (전처리 전)': preprocessing_stats['missing_values']['before'].values(),
'결측치 (전처리 후)': preprocessing_stats['missing_values']['after'].values()
})
st.dataframe(missing_df, use_container_width=True, hide_index=True)

with col2:
st.markdown("#### 🔧 처리 방법")
st.info("""
                   **title (제목)**
                   - 결측치 행 제거
                   - 이유: 제목은 필수 분석 대상
                   
                   **숫자형 컬럼**
                   - 중앙값(median)으로 대체
                   - 이유: 평균보다 이상치에 강건함
                   """)

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

x = list(missing_df['컬럼'])
before = list(missing_df['결측치 (전처리 전)'])
after = list(missing_df['결측치 (전처리 후)'])

x_pos = np.arange(len(x))
width = 0.35

ax.bar(x_pos - width/2, before, width, label='전처리 전', color='#FF6B6B', alpha=0.8)
ax.bar(x_pos + width/2, after, width, label='전처리 후', color='#4ECDC4', alpha=0.8)

ax.set_xlabel('컬럼', fontsize=12, fontweight='bold')
ax.set_ylabel('결측치 개수', fontsize=12, fontweight='bold')
ax.set_title('결측치 처리 전후 비교', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# 2. 이상치 분석
st.markdown("### 2️⃣ 이상치 제거 (IQR 방법)")

col1, col2 = st.columns([2, 3])

with col1:
st.markdown("#### 🔧 IQR 방법이란?")
st.info("""
                   **IQR (Interquartile Range)**
                   - IQR = Q3 - Q1
                   - 하한: Q1 - 1.5 × IQR
                   - 상한: Q3 + 1.5 × IQR
                   
                   이 범위를 벗어난 값을 이상치로 판단하여 제거
                   """)

st.markdown("#### 📊 제거된 이상치")
outlier_df = pd.DataFrame({
'변수': [k for k in preprocessing_stats['outliers_removed'].keys() if k != 'total'],
'제거된 개수': [v for k, v in preprocessing_stats['outliers_removed'].items() if k != 'total']
})
st.dataframe(outlier_df, use_container_width=True, hide_index=True)

with col2:
# Box plot으로 이상치 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

variables = ['views', 'likes', 'comment_count', 'title_length']
var_names = ['조회수', '좋아요 수', '댓글 수', '제목 길이']

for i, (var, name) in enumerate(zip(variables, var_names)):
ax = axes[i//2, i%2]
ax.boxplot(df[var], vert=True)
ax.set_title(f'{name} 분포', fontsize=12, fontweight='bold')
ax.set_ylabel(name, fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# 3. 데이터 정규화
st.markdown("### 3️⃣ 데이터 정규화 (Min-Max Scaling)")

col1, col2 = st.columns([2, 3])

with col1:
st.markdown("#### 🔧 정규화란?")
st.info("""
                   **Min-Max Scaling**
                   
                   모든 값을 0~1 범위로 변환
                   
                   ```
                   normalized = (X - X_min) / (X_max - X_min)
                   ```
                   
                   **장점:**
                   - 서로 다른 단위의 변수 비교 가능
                   - 머신러닝 모델 학습 시 성능 향상
                   """)

st.markdown("#### 📊 정규화 통계")
norm_stats = []
for col in ['views', 'likes', 'comment_count', 'title_length']:
norm_stats.append({
'변수': col,
'최소값': f"{df[col].min():,.0f}",
'최대값': f"{df[col].max():,.0f}",
'정규화 범위': "0.0 ~ 1.0"
})

norm_df = pd.DataFrame(norm_stats)
st.dataframe(norm_df, use_container_width=True, hide_index=True)

with col2:
# 정규화 전후 비교 그래프
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

variables = ['views', 'likes', 'comment_count', 'title_length']
var_names = ['조회수', '좋아요 수', '댓글 수', '제목 길이']

for i, (var, name) in enumerate(zip(variables, var_names)):
ax = axes[i//2, i%2]

# 정규화된 값 계산
normalized = (df[var] - df[var].min()) / (df[var].max() - df[var].min())

ax.hist(normalized, bins=30, color='#4ECDC4', alpha=0.7, edgecolor='black')
ax.set_title(f'{name} (정규화 후)', fontsize=12, fontweight='bold')
ax.set_xlabel('정규화된 값', fontsize=10)
ax.set_ylabel('빈도', fontsize=10)
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# 4. 전처리 효과
st.markdown("### 📈 전처리 효과 분석")

st.success(f"""
               ✅ **데이터 품질이 향상되었습니다!**
               
               - 원본 데이터: {preprocessing_stats['original_size']:,}개
               - 최종 데이터: {preprocessing_stats['final_size']:,}개 ({(preprocessing_stats['final_size']/preprocessing_stats['original_size']*100):.1f}% 유지)
               - 제거된 데이터: {preprocessing_stats['original_size'] - preprocessing_stats['final_size']:,}개 ({preprocessing_stats['removed_percentage']:.1f}%)
               
               **전처리를 통해:**
               - ✓ 결측치 문제 해결
               - ✓ 이상치로 인한 분석 왜곡 방지
               - ✓ 변수 간 공정한 비교 가능
               """)

# 전처리 프로세스 다이어그램
st.markdown("#### 🔄 전처리 프로세스")
st.code("""
               원본 데이터
                   ↓
               [1단계] 결측치 처리 (제목 제거, 숫자 중앙값 대체)
                   ↓
               [2단계] 이상치 제거 (IQR 방법)
                   ↓
               [3단계] 데이터 정규화 (Min-Max Scaling)
                   ↓
               최종 클린 데이터
               """, language="text")

else:
# 업로드 대기 화면
st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드해주세요!")

col1, col2 = st.columns([1, 1])

with col1:
st.markdown("### 📝 사용 방법")
st.markdown("""
           1. Kaggle에서 YouTube 데이터셋 다운로드
           2. CSV 파일을 사이드바에서 업로드
           3. 탭을 클릭하여 다양한 분석 확인
           
           **4개의 분석 페이지:**
           - 📈 **개요**: 전체 데이터 요약 & 분포
           - 🔍 **상관관계 분석**: 변수 간 관계 파악
           - 📏 **제목 길이 분석**: 구간별 성과 비교
           - 🏆 **Top 영상**: 인기 영상 랭킹
           """)

with col2:
st.markdown("### 🎯 분석 목표")
st.markdown("""
           이 대시보드는 YouTube 영상의 **제목 길이**와 
           **조회수, 좋아요, 댓글 수** 사이의 상관관계를 
           분석합니다.
           
           **주요 질문:**
           - 제목 길이가 조회수에 영향을 줄까?
           - 어떤 변수들이 강한 상관관계를 가질까?
           - 최적의 제목 길이는?
           """)

if __name__ == "__main__":
    main()
    main()
