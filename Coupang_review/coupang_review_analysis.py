import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
import warnings

# 시각화 라이브러리 추가
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
# 폰트 관리 모듈 추가
from matplotlib import font_manager, rc

# 경고 메시지 무시 설정
warnings.filterwarnings("ignore")

# 감성 분석 및 NLP 라이브러리
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK 데이터 다운로드 (필요한 데이터가 없으면 자동 다운로드 시도)
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# ============================================================================
# ⚠️ [최종 수정] 한글 폰트 설정 (Mac OS 최적화 및 오류 방지)
# ============================================================================
WORDCLOUD_FONT_PATH = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'

try:
    # matplotlib의 폰트 매니저를 이용해 폰트 이름 설정
    font_name = font_manager.FontProperties(fname=WORDCLOUD_FONT_PATH).get_name()
    rc('font', family=font_name, size=10) # 폰트 크기 조정
    print(f"   ✓ 시각화 폰트 설정 완료: {font_name}")
except Exception:
    # 폰트 경로 오류 시 fallback 설정
    rc('font', family='AppleGothic', size=10)
    print("   ⚠️ 경고: 기본 폰트 설정으로 대체되었습니다.")

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# ============================================================================
print("=" * 80)
print("📦 쿠팡 리뷰 감성 분석 시스템 (origianl_coupang.csv 기준)")
print("=" * 80)


# 리뷰 파일명 설정
COUPANG_REVIEWS_FILE = "origianl_coupang.csv"
OUTPUT_FOLDER = "Coupang_review_analysis"


# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

def clean_text(text):
    """텍스트를 정제합니다. (한글을 유지하도록 수정)"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s가-힣]", " ", text) 
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def categorize_sentiment(rating):
    """평점을 기준으로 감성을 분류합니다."""
    if rating >= 4:
        return "positive"
    elif rating >= 3:
        return "neutral"
    else:
        return "negative"


def load_and_preprocess_data(reviews_path):
    """리뷰 데이터를 로드하고 전처리합니다."""
    print("\n[1/5] 데이터 로딩 중...")

    reviews_df = pd.read_csv(reviews_path, encoding="utf-8-sig")
    reviews_df["review_text_clean"] = reviews_df["review_text"].apply(clean_text)
    reviews_df["sentiment_category"] = reviews_df["rating"].apply(categorize_sentiment)

    print(f"   ✓ 총 {len(reviews_df)}개 리뷰 로드 완료")
    return reviews_df


# ============================================================================
# 2. 감성 분석 (TextBlob 사용)
# ============================================================================

def perform_sentiment_analysis(df):
    """TextBlob을 사용한 상세 감성 분석을 수행합니다."""
    print("\n[2/5] 감성 분석 수행 중...")
    sentiments = []
    subjectivities = []
    
    # 텍스트가 한국어이므로 TextBlob의 결과를 그대로 사용하지 않고, 
    # 평점 기반 감성 분류를 메인으로 사용합니다. 
    # (TextBlob은 영어 기반으로 Polarity만 참고)
    for text in tqdm(df["review_text"], desc="   감성 점수 계산"): 
        blob = TextBlob(str(text))
        sentiments.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)
    df["sentiment_polarity"] = sentiments
    df["sentiment_subjectivity"] = subjectivities
    print(f"   ✓ 감성 분석 완료")
    return df


# ============================================================================
# 3. 키워드 추출 (TF-IDF)
# ============================================================================

def get_extended_stopwords():
    """기본 영문 불용어와 제품 관련 불용어를 결합합니다."""
    base_stopwords = set(stopwords.words("english"))
    custom_stopwords = {
        "probiotic", "product", "supplement", "capsule", "day", "take", 
        "love", "like", "feel", "use", "time", "one", "try", "also",
        "would", "get", "bought", "really", "much", "since", "well", "go",
        "make", "start", "see", "back", "first", "even", "still", "thing",
        "find", "lot", "way", "last", "came", "ordered", "hope", "put",
        "think", "getting", "know", "sure", "need", "used", "though",
        "good", "great", "best",
    }
    return base_stopwords.union(custom_stopwords)


def is_meaningful_keyword(word):
    """단어의 길이와 품사 태그를 검사하여 의미있는 키워드인지 판단합니다."""
    if len(word) < 3 or word.isdigit():
        return False
    
    try:
        tag = pos_tag([word])[0][1]
        return tag.startswith("NN") or tag.startswith("JJ")
    except:
        return True


def filter_meaningful_keywords(keywords_with_scores, min_word_length=3):
    """추출된 키워드 목록에서 의미없는 단어를 필터링합니다. (한글은 통과)"""
    return [
        (w, s) for w, s in keywords_with_scores 
        if is_meaningful_keyword(w) or re.search(r'[가-힣]', w)
    ]


def extract_keywords_by_sentiment(df, n_keywords=15):
    """TF-IDF를 사용하여 감성별 키워드를 추출합니다."""
    print("\n[3/5] 키워드 추출 중...")
    extended_stopwords = get_extended_stopwords()
    keywords_by_sentiment = {}
    
    for sentiment in ["positive", "neutral", "negative"]:
        texts = df[df["sentiment_category"] == sentiment]["review_text_clean"].tolist()
        
        if not texts or all(not text.strip() for text in texts):
            keywords_by_sentiment[sentiment] = []
            continue

        vectorizer = TfidfVectorizer(
            max_features=300,
            stop_words=list(extended_stopwords),
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            if tfidf_matrix.shape[1] == 0:
                 keywords_by_sentiment[sentiment] = []
                 continue
                 
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            
            top_indices = mean_tfidf.argsort()[-50:][::-1]
            candidate_keywords = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
            
            filtered_keywords = filter_meaningful_keywords(candidate_keywords)
            
            keywords_by_sentiment[sentiment] = filtered_keywords[:n_keywords]
        
        except ValueError:
            keywords_by_sentiment[sentiment] = []
            
    print("   ✓ 키워드 추출 완료")
    return keywords_by_sentiment

# ============================================================================
# 4. 고객 불만 영역 자동 도출 (기존 로직 유지)
# ============================================================================

def extract_improvement_insights(df):
    """사전 정의된 키워드 딕셔너리를 기반으로 불만 영역을 분류하고 우선순위를 도출합니다."""
    print("\n[4/5] 개선 포인트 분석 중...")
    
    concern_reviews = df[df["sentiment_category"].isin(["neutral", "negative"])].copy()
    if concern_reviews.empty:
        print("   ✓ 0개 개선 영역 발견")
        return []

    improvement_areas = {
        "side_effects": {"keywords": ["bloat", "bloating", "gas", "nausea", "headache", "stomach", "pain"], "mentions": [], "description": "부작용/불편함"},
        "packaging_quality": {"keywords": ["package", "bottle", "seal", "broken", "expire", "smell"], "mentions": [], "description": "포장/품질 문제"},
        "effectiveness": {"keywords": ["ineffective", "not work", "no effect", "no difference"], "mentions": [], "description": "효과/효능 부족"},
        "price_value": {"keywords": ["expensive", "cost", "pricey"], "mentions": [], "description": "가격/가치 불만족"},
        "swallowing": {"keywords": ["swallow", "big", "large", "taste"], "mentions": [], "description": "섭취의 어려움/맛"},
        "delivery": {"keywords": ["delivery", "late", "shipping"], "mentions": [], "description": "배송 불만"},
    }

    for index, row in concern_reviews.iterrows():
        review_text = row["review_text_clean"]
        
        for area_code, data in improvement_areas.items():
            found_keywords = [k for k in data["keywords"] if k in review_text]
            
            if found_keywords:
                data["mentions"].append({
                    "rating": row["rating"],
                    "text": row["review_text"],
                    "keywords": found_keywords
                })

    prioritized_areas = []
    for code, data in improvement_areas.items():
        if data["mentions"]:
            prioritized_areas.append({
                "code": code,
                "description": data["description"],
                "count": len(data["mentions"]),
                "samples": data["mentions"],
            })
            
    prioritized_areas.sort(key=lambda x: x["count"], reverse=True)

    print(f"   ✓ {len(prioritized_areas)}개 개선 영역 발견")
    return prioritized_areas


# ============================================================================
# 5. 시각화 및 분석 리포트 생성 (4-in-1 차트 로직 복원 및 강화)
# ============================================================================

def create_4in1_sentiment_chart(df, keywords_with_scores, sentiment, output_folder):
    """원하시는 대로 4가지 차트(평점, 감성, 키워드 바, 워드클라우드)를 
    하나의 이미지 파일에 생성합니다."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 
    fig.suptitle(f'{sentiment.capitalize()} 리뷰 종합 분석 대시보드', fontsize=20, y=1.02)
    
    # ---------------- 1. 평점 분포 (Rating Distribution) ----------------
    sns.countplot(x='rating', data=df[df['sentiment_category'] == sentiment], ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('1. 평점 분포', fontsize=14)
    axes[0, 0].set_xlabel('평점 (Rating)', fontsize=12)
    axes[0, 0].set_ylabel('리뷰 수', fontsize=12)
    
    # ---------------- 2. 감성 극성 분포 (Sentiment Polarity Distribution) ----------------
    sns.histplot(df[df['sentiment_category'] == sentiment]['sentiment_polarity'], bins=10, kde=True, ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('2. 감성 극성 분포', fontsize=14)
    axes[0, 1].set_xlabel('감성 극성 (Polarity) -1.0 ~ 1.0', fontsize=12)
    axes[0, 1].set_ylabel('빈도', fontsize=12)
    
    # ---------------- 3. 상위 키워드 바 차트 (Top Keyword Bar Chart) ----------------
    top_keywords = pd.DataFrame(keywords_with_scores, columns=['Keyword', 'Score'])
    if not top_keywords.empty:
        top_keywords = top_keywords.sort_values(by='Score', ascending=True).tail(10)
        sns.barplot(x='Score', y='Keyword', data=top_keywords, ax=axes[1, 0], color='skyblue')
        axes[1, 0].set_title('3. 상위 10개 키워드 (TF-IDF)', fontsize=14)
        axes[1, 0].set_xlabel('TF-IDF 점수', fontsize=12)
        axes[1, 0].set_ylabel('키워드', fontsize=12)
    else:
        axes[1, 0].text(0.5, 0.5, '키워드 없음', ha='center', va='center', fontsize=16)
        axes[1, 0].set_title('3. 상위 10개 키워드 (TF-IDF)', fontsize=14)
        axes[1, 0].axis('off')

    # ---------------- 4. 워드 클라우드 (Word Cloud) ----------------
    keyword_dict = {k: v for k, v in keywords_with_scores}
    if keyword_dict:
        wc = WordCloud(
            font_path=WORDCLOUD_FONT_PATH,
            width=500, # 서브플롯 크기에 맞춤
            height=300, # 서브플롯 크기에 맞춤
            background_color="white",
            max_words=30,
            normalize_plurals=False
        )
        wc.generate_from_frequencies(keyword_dict)
        axes[1, 1].imshow(wc, interpolation="bilinear")
        axes[1, 1].set_title('4. 주요 키워드 워드 클라우드', fontsize=14)
        axes[1, 1].axis("off")
    else:
        axes[1, 1].text(0.5, 0.5, '키워드 없음', ha='center', va='center', fontsize=16)
        axes[1, 1].set_title('4. 주요 키워드 워드 클라우드', fontsize=14)
        axes[1, 1].axis('off')

    # ⚠️ 최종 저장 직전: 저장 오류 방지 및 레이아웃 최적화
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(
        os.path.join(output_folder, f"analysis_{sentiment}.png"),
        dpi=150,           # 고해상도 지정
        transparent=False  # 투명 배경 제거 (저장 오류 방지)
    )
    plt.close()


def create_visualizations(df, keywords_by_sentiment, improvement_areas):
    """모든 시각화 파일을 생성하고 저장합니다."""
    print("\n[5/5] 시각화 생성 중...")
    output_folder = OUTPUT_FOLDER
    
    # 1. 전체 감성 분포 요약 차트 (기존 로직 유지)
    # create_summary_chart(df, output_folder) # 이 차트는 4-in-1에 포함되지 않아 생략

    # 2. 감성별 종합 차트 (4-in-1 대시보드)
    for sentiment, keywords in keywords_by_sentiment.items():
        # 중립 리뷰의 경우 데이터가 적어 오류 가능성이 있어 예외 처리
        if sentiment == 'neutral' and len(df[df['sentiment_category'] == sentiment]) < 10:
             print(f"   ⚠️ {sentiment} 리뷰 수가 부족하여 종합 대시보드 생략")
             continue

        create_4in1_sentiment_chart(df, keywords, sentiment, output_folder)
        print(f"   ✓ {sentiment} 종합 대시보드 저장: {os.path.join(output_folder, f'analysis_{sentiment}.png')}")
            
    print(f"   ✓ 모든 시각화 완료")


def generate_analysis_report(df, keywords_by_sentiment, improvement_areas):
    # (리포트 생성 로직은 동일하므로 생략)
    # ... 이전 코드와 동일하게 유지 ...
    print("\n[추가] 분석 결과 리포트 생성 중...")
    output_folder = OUTPUT_FOLDER
    report_path = os.path.join(output_folder, "analysis_report.txt")
    
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    sentiment_counts = df['sentiment_category'].value_counts(normalize=True)
    pos_ratio = sentiment_counts.get('positive', 0)
    neu_ratio = sentiment_counts.get('neutral', 0)
    neg_ratio = sentiment_counts.get('negative', 0)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("쿠팡 리뷰 분석 결과 리포트\n")
        f.write("=" * 80 + "\n\n")

        # 1. 전체 통계
        f.write("[ 1. 전체 통계 ]\n")
        f.write("-" * 80 + "\n")
        f.write(f"총 리뷰 수: {total_reviews:,}개\n")
        f.write(f"평균 평점: {avg_rating:.2f} / 5.0\n")
        f.write(f"긍정 리뷰 비율: {pos_ratio:.1%}\n")
        f.write(f"중립 리뷰 비율: {neu_ratio:.1%}\n")
        f.write(f"부정 리뷰 비율: {neg_ratio:.1%}\n")
        f.write(f"평균 감성 극성: {df['sentiment_polarity'].mean():.3f} (-1 ~ 1)\n\n")

        # 2. 감성별 주요 키워드
        f.write("[ 2. 감성별 주요 키워드 ]\n")
        f.write("-" * 80 + "\n")
        for sentiment, keywords in keywords_by_sentiment.items():
            f.write(f"\n■ {sentiment.upper()} 리뷰 키워드:\n")
            if keywords:
                for i, (word, score) in enumerate(keywords):
                    f.write(f"   {i+1:2d}. {word:30} (TF-IDF: {score:.4f})\n")
            else:
                f.write("   (키워드 없음)\n")

        # 3. 고객 불만 영역 분석
        f.write("\n[ 3. 고객 불만 영역 분석 (중립/부정 리뷰 기준) ]\n")
        f.write("-" * 80 + "\n")
        if not improvement_areas:
            f.write("분석된 주요 불만 영역이 없습니다. (한국어 리뷰로 인한 영문 키워드 미매칭 가능성)\n")
        else:
            for i, area in enumerate(improvement_areas):
                f.write(f"\n[{i+1}순위] {area['description']}\n")
                f.write(f"  └ 언급 횟수: {area['count']}회\n")
                f.write(f"  └ 영역 코드: {area['code']}\n")
                f.write(f"  └ 고객 의견 샘플:\n")
                for j, sample in enumerate(area['samples'][:2]):
                    f.write(f"     {j+1}. (평점 {sample['rating']:.1f}) {sample['text'][:100]}...\n") 

    print(f"   ✓ 리포트 저장: {report_path}")
    return report_path


# ============================================================================
# 메인 실행 함수 (최종)
# ============================================================================

if __name__ == "__main__":
    if not os.path.exists(COUPANG_REVIEWS_FILE):
        print(f"❌ 오류: {COUPANG_REVIEWS_FILE} 파일을 찾을 수 없습니다.")
        print("   파일명을 확인하거나, 스크립트와 같은 폴더에 파일을 두세요.")
    else:
        try:
            reviews_df = load_and_preprocess_data(COUPANG_REVIEWS_FILE)
            reviews_df = perform_sentiment_analysis(reviews_df)
            keywords_by_sentiment = extract_keywords_by_sentiment(reviews_df, n_keywords=15)
            improvement_areas = extract_improvement_insights(reviews_df)
            
            if not os.path.exists(OUTPUT_FOLDER):
                os.makedirs(OUTPUT_FOLDER)
                
            # 4-in-1 시각화 생성
            create_visualizations(reviews_df, keywords_by_sentiment, improvement_areas)
            
            # 리포트 생성
            report_path = generate_analysis_report(reviews_df, keywords_by_sentiment, improvement_areas)
            
            print("\n" + "=" * 80)
            print("✅ 쿠팡 리뷰 분석이 완료되었습니다! (4-in-1 대시보드 형식)")
            print("=" * 80)
            print(f"결과물은 '{OUTPUT_FOLDER}' 폴더에 저장되었습니다.")

        except Exception as e:
            print(f"\n❌ 분석 중 오류가 발생했습니다: {e}")
            import traceback
            traceback.print_exc()