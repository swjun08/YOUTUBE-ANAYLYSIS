import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
import warnings
import platform

warnings.filterwarnings("ignore")

# ===================== 한글 폰트 설정 =====================
def set_korean_font():
    import matplotlib.font_manager as fm

    system = platform.system()

    if system == "Windows":
        font_name = "Malgun Gothic"
    elif system == "Darwin":
        font_name = "AppleGothic"
    else:  # Linux (Streamlit Cloud)
        font_dirs = ["/usr/share/fonts/truetype/nanum"]
        font_files = fm.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            fm.fontManager.addfont(font_file)
        font_name = "NanumGothic"

    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()

# ===================== 페이지 설정 =====================
st.set_page_config(
    page_title="YouTube 영상 상관관계 분석",
    page_icon="📊",
    layout="wide",
)

# ===================== YouTube API 데이터 수집 =====================
@st.cache_data
def fetch_youtube_data(api_key, query, max_results):
    youtube = build("youtube", "v3", developerKey=api_key)

    search_response = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=max_results
    ).execute()

    video_ids = [item["id"]["videoId"] for item in search_response["items"]]
    if not video_ids:
        return None

    video_response = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    ).execute()

    data = []
    for item in video_response["items"]:
        snippet = item["snippet"]
        stats = item["statistics"]

        data.append({
            "Video Title": snippet.get("title"),
            "Video Views": int(stats.get("viewCount", 0)),
            "Like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0))
        })

    return pd.DataFrame(data)


# ===================== 데이터 전처리 =====================
@st.cache_data
def load_and_process_data(data_source):
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        df = pd.read_csv(data_source)

    original_size = len(df)

    df = df.rename(columns={
        "Video Title": "title",
        "Video Views": "views",
        "Like_count": "likes",
        "comment_count": "comment_count"
    })

    df = df[["title", "views", "likes", "comment_count"]]

    df = df.dropna(subset=["title"])
    for col in ["views", "likes", "comment_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

    df["title_length"] = df["title"].str.len()

    # 이상치 제거 (IQR)
    for col in ["views", "likes", "comment_count", "title_length"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    stats = {
        "original_size": original_size,
        "final_size": len(df)
    }

    return df, stats


# ===================== 상관관계 =====================
def get_correlations(df):
    cols = ["title_length", "views", "likes", "comment_count"]
    return df[cols].corr()


# ===================== 메인 앱 =====================
def main():
    st.title("📊 YouTube 영상 상관관계 분석 대시보드")
    st.markdown("**YouTube API 또는 CSV 파일을 이용해 제목 길이와 성과의 관계를 분석합니다**")

    # ===================== 사이드바 =====================
    with st.sidebar:
        st.header("🔑 YouTube API 연동")

        api_key = st.text_input("YouTube API Key", type="password")
        query = st.text_input("검색어", value="BTS")
        max_results = st.slider("영상 개수", 10, 100, 50, step=10)

        use_api = st.button("🔍 YouTube에서 데이터 가져오기")

        st.markdown("---")
        uploaded_file = st.file_uploader("CSV 업로드", type=["csv"])

    df = None

    # ===================== 데이터 로딩 =====================
    if use_api and api_key:
        with st.spinner("YouTube API 데이터 수집 중..."):
            api_df = fetch_youtube_data(api_key, query, max_results)

        if api_df is not None:
            df, stats = load_and_process_data(api_df)
            st.success("✅ YouTube API 데이터 로드 완료")

    elif uploaded_file is not None:
        with st.spinner("CSV 데이터 로딩 중..."):
            df, stats = load_and_process_data(uploaded_file)
        st.success("✅ CSV 데이터 로드 완료")

    else:
        st.info("👈 사이드바에서 API 사용 또는 CSV 업로드를 선택하세요")
        return

    # ===================== 개요 =====================
    st.markdown("## 📈 데이터 개요")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("영상 수", len(df))
    col2.metric("평균 조회수", f"{df['views'].mean():,.0f}")
    col3.metric("평균 좋아요", f"{df['likes'].mean():,.0f}")
    col4.metric("평균 제목 길이", f"{df['title_length'].mean():.1f}자")

    # ===================== 히트맵 =====================
    st.markdown("## 🔥 상관관계 히트맵")

    corr = get_correlations(df)
    corr.columns = ["제목 길이", "조회수", "좋아요 수", "댓글 수"]
    corr.index = ["제목 길이", "조회수", "좋아요 수", "댓글 수"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="RdYlBu_r",
        center=0,
        fmt=".3f",
        square=True,
        linewidths=1
    )
    st.pyplot(fig)

    # ===================== 제목 길이 vs 조회수 =====================
    st.markdown("## 📏 제목 길이 vs 조회수")

    corr_value = df["title_length"].corr(df["views"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["title_length"], df["views"], alpha=0.5)
    z = np.polyfit(df["title_length"], df["views"], 1)
    p = np.poly1d(z)
    ax.plot(df["title_length"], p(df["title_length"]), "r--")

    ax.set_xlabel("제목 길이")
    ax.set_ylabel("조회수")
    ax.set_title(f"상관계수: {corr_value:.4f}")
    ax.grid(True)

    st.pyplot(fig)

    # ===================== Top 영상 =====================
    st.markdown("## 🏆 조회수 TOP 10")

    top10 = df.nlargest(10, "views")[["title", "title_length", "views", "likes", "comment_count"]]
    top10.columns = ["제목", "제목 길이", "조회수", "좋아요", "댓글"]
    st.dataframe(top10, use_container_width=True)


if __name__ == "__main__":
    main()
