import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 감성 분석 및 NLP
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK 데이터 다운로드
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
print("=" * 80)
print("다이어트 유산균 리뷰 분석 시스템")
print("=" * 80)


# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================


def load_and_preprocess_data(reviews_path, products_path):
    """리뷰 및 제품 데이터를 로드하고 전처리합니다."""
    print("\n[1/5] 데이터 로딩 중...")

    reviews_df = pd.read_csv(reviews_path, encoding="utf-8-sig")
    products_df = pd.read_csv(products_path, encoding="utf-8-sig")

    # 리뷰 텍스트 정제
    reviews_df["review_text_clean"] = reviews_df["review_text"].apply(clean_text)

    # 감성 카테고리 분류 (평점 기준)
    reviews_df["sentiment_category"] = reviews_df["rating"].apply(categorize_sentiment)

    print(f"   ✓ 총 {len(reviews_df)}개 리뷰 로드 완료")
    print(f"   ✓ 총 {len(products_df)}개 제품 정보 로드 완료")

    return reviews_df, products_df


def clean_text(text):
    """텍스트를 정제합니다."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
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


# ============================================================================
# 2. 감성 분석
# ============================================================================


def perform_sentiment_analysis(df):
    """TextBlob을 사용한 상세 감성 분석을 수행합니다."""
    print("\n[2/5] 감성 분석 수행 중...")

    sentiments = []
    subjectivities = []
    for text in tqdm(df["review_text_clean"], desc="   감성 점수 계산"):
        blob = TextBlob(text)
        sentiments.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)
    df["sentiment_polarity"] = sentiments
    df["sentiment_subjectivity"] = subjectivities
    print(f"   ✓ 감성 분석 완료")
    return df


# ============================================================================
# 3. 강화된 불용어+명사/형용사 필터 기반 키워드 추출
# ============================================================================


def get_extended_stopwords():
    base_stopwords = set(stopwords.words("english"))
    custom_stopwords = {
        # 기존 및 개선 요청 반영
        "probiotic",
        "probiotics",
        "product",
        "products",
        "supplement",
        "supplements",
        "capsule",
        "capsules",
        "pill",
        "pills",
        "tablet",
        "tablets",
        "gummy",
        "gummies",
        "bottle",
        "bottles",
        "box",
        "boxes",
        "package",
        "packages",
        "container",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "time",
        "times",
        "first",
        "second",
        "third",
        "one",
        "two",
        "three",
        "30",
        "60",
        "90",
        "100",
        "120",
        "daily",
        "every",
        "morning",
        "night",
        "evening",
        "buy",
        "bought",
        "purchase",
        "purchased",
        "order",
        "ordered",
        "ordering",
        "amazon",
        "seller",
        "delivery",
        "shipping",
        "arrived",
        "receive",
        "received",
        "take",
        "taking",
        "took",
        "taken",
        "use",
        "using",
        "used",
        "get",
        "getting",
        "got",
        "try",
        "trying",
        "tried",
        "start",
        "started",
        "starting",
        "begin",
        "began",
        "continue",
        "continued",
        "stop",
        "stopped",
        "make",
        "making",
        "made",
        "go",
        "going",
        "went",
        "gone",
        "come",
        "coming",
        "came",
        "good",
        "great",
        "best",
        "better",
        "nice",
        "fine",
        "ok",
        "okay",
        "bad",
        "worse",
        "worst",
        "poor",
        "terrible",
        "horrible",
        "really",
        "very",
        "much",
        "more",
        "most",
        "less",
        "least",
        "just",
        "only",
        "also",
        "even",
        "still",
        "yet",
        "always",
        "never",
        "sometimes",
        "recommend",
        "recommended",
        "recommendation",
        "review",
        "reviews",
        "rating",
        "star",
        "stars",
        "love",
        "like",
        "liked",
        "dislike",
        "work",
        "works",
        "worked",
        "working",
        "help",
        "helps",
        "helped",
        "helping",
        "effect",
        "effects",
        "result",
        "results",
        "notice",
        "noticed",
        "noticing",
        "see",
        "seeing",
        "saw",
        "seen",
        "feel",
        "feeling",
        "felt",
        "seem",
        "seems",
        "seemed",
        "appear",
        "appears",
        "appeared",
        "thing",
        "things",
        "something",
        "anything",
        "everything",
        "nothing",
        "want",
        "wanted",
        "need",
        "needed",
        "may",
        "might",
        "could",
        "would",
        "should",
        "think",
        "thought",
        "know",
        "knew",
        "sure",
        "however",
        "although",
        "though",
        # 추가 강화: 아래는 개선 요청 및 도메인별 의미 없음 → price, swallow 제외!
        "well",
        "easy",
        "difference",
        "back",
        "since",
        "bit",
        "long",
        "life",
        "said",
        "full",
        "brand",
        "dissolve",
        "different",
        "body",
        "looked",
        "definitely",
        "regular",
        "keep",
        "issues",
        "fiber",
        "highly",
        "ones",
        "new",
        "way",
        "little",
        "twice",
        "people",
        "change",
        "read",
        "chew",
        "collagen",
        "benefits",
        "open",
        "hard",
        "size",
        "large",
    }
    # price, swallow는 제거!
    return base_stopwords.union(custom_stopwords - {"price", "swallow"})


def is_meaningful_keyword(word):
    # 최소 3자, 숫자제외, 중복문자제외, 영어 알파벳
    if len(word) < 3:
        return False
    if word.isdigit():
        return False
    if any(char.isdigit() for char in word):
        return False
    if len(set(word)) == 1:
        return False
    # 명사/형용사 필터 적용 (POS tagging)
    tag = pos_tag([word])[0][1]
    if not (tag.startswith("NN") or tag.startswith("JJ")):
        return False
    return True


def filter_meaningful_keywords(keywords_with_scores, min_word_length=3):
    return [(w, s) for w, s in keywords_with_scores if is_meaningful_keyword(w)]


def extract_keywords_by_sentiment(df, n_keywords=15):
    extended_stopwords = get_extended_stopwords()
    keywords_by_sentiment = {}
    for sentiment in ["positive", "neutral", "negative"]:
        texts = df[df["sentiment_category"] == sentiment]["review_text_clean"].tolist()
        if not texts:
            keywords_by_sentiment[sentiment] = []
            continue
        vectorizer = TfidfVectorizer(
            max_features=300,
            stop_words=list(extended_stopwords),
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.5,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-50:][::-1]
        candidate_keywords = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
        filtered_keywords = filter_meaningful_keywords(candidate_keywords)
        keywords_by_sentiment[sentiment] = filtered_keywords[:n_keywords]
    return keywords_by_sentiment


# ============================================================================
# 4. 고객 불만 영역 자동 도출(기존 유지)
# ============================================================================


def extract_improvement_insights(df):
    print("\n[4/5] 개선 포인트 분석 중...")
    concern_reviews = df[df["sentiment_category"].isin(["negative", "neutral"])]
    if len(concern_reviews) == 0:
        return {}
    improvement_areas = {
        "effectiveness": {
            "keywords": [
                "ineffective",
                "not work",
                "no effect",
                "no result",
                "no change",
                "didnt work",
                "doesnt work",
                "useless",
                "waste",
                "disappointed",
                "nothing",
                "zero",
                "fake",
            ],
            "mentions": [],
            "description": "효과/효능 부족",
        },
        "side_effects": {
            "keywords": [
                "bloat",
                "bloating",
                "gas",
                "gassy",
                "upset stomach",
                "nausea",
                "cramping",
                "cramps",
                "diarrhea",
                "constipation",
                "uncomfortable",
                "pain",
                "ache",
                "sick",
                "headache",
                "allergy",
                "allergic",
                "rash",
            ],
            "mentions": [],
            "description": "부작용/불편함",
        },
        "taste_smell": {
            "keywords": [
                "taste",
                "aftertaste",
                "smell",
                "odor",
                "stink",
                "fishy",
                "bitter",
                "sour",
                "disgusting",
                "nasty",
                "unpleasant",
                "flavor",
            ],
            "mentions": [],
            "description": "맛/냄새 문제",
        },
        "size_difficulty": {
            "keywords": [
                "large",
                "huge",
                "big",
                "swallow",
                "choke",
                "hard swallow",
                "difficult swallow",
                "horse pill",
                "size",
                "gigantic",
            ],
            "mentions": [],
            "description": "크기/복용 불편",
        },
        "price_value": {
            "keywords": [
                "expensive",
                "overpriced",
                "costly",
                "pricey",
                "too much money",
                "not worth",
                "poor value",
                "waste money",
                "cheap quality",
            ],
            "mentions": [],
            "description": "가격/가치 불만족",
        },
        "packaging_quality": {
            "keywords": [
                "packaging",
                "broken",
                "damaged",
                "leak",
                "leaking",
                "seal",
                "unsealed",
                "melted",
                "stuck together",
                "quality control",
                "expire",
            ],
            "mentions": [],
            "description": "포장/품질 문제",
        },
        "ingredient_concerns": {
            "keywords": [
                "artificial",
                "sugar",
                "sweetener",
                "additive",
                "filler",
                "chemical",
                "synthetic",
                "dye",
                "color",
                "preservative",
                "gmo",
            ],
            "mentions": [],
            "description": "성분 우려",
        },
        "dosage_inconvenience": {
            "keywords": [
                "multiple times",
                "too many",
                "several times",
                "inconvenient",
                "forget",
                "refrigerate",
                "refrigeration",
                "storage",
            ],
            "mentions": [],
            "description": "복용량/보관 불편",
        },
    }
    for idx, row in concern_reviews.iterrows():
        review_text = row["review_text_clean"]
        for area, info in improvement_areas.items():
            for keyword in info["keywords"]:
                if keyword in review_text:
                    improvement_areas[area]["mentions"].append(
                        {
                            "review_id": row["review_id"],
                            "product_name": row["product_name"],
                            "rating": row["rating"],
                            "keyword": keyword,
                            "review_snippet": review_text[:200],
                        }
                    )
                    break
    prioritized_areas = []
    for area, info in improvement_areas.items():
        mention_count = len(info["mentions"])
        if mention_count > 0:
            prioritized_areas.append(
                {
                    "area": area,
                    "description": info["description"],
                    "mention_count": mention_count,
                    "sample_reviews": info["mentions"][:3],
                }
            )
    prioritized_areas.sort(key=lambda x: x["mention_count"], reverse=True)
    print(f"   ✓ {len(prioritized_areas)}개 개선 영역 발견")
    return prioritized_areas


# ============================================================================
# 5. 시각화 및 분석 리포트 생성(제품별 평점 제거)
# ============================================================================


def create_visualizations(df, keywords_by_sentiment, improvement_areas):
    print("\n[5/5] 시각화 생성 중...")
    output_folder = "Amazon_review_analysis"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 맑은 고딕
    plt.rcParams['axes.unicode_minus'] = False
    for sentiment in ["positive", "neutral", "negative"]:
        sentiment_df = df[df["sentiment_category"] == sentiment]
        if len(sentiment_df) == 0:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Review Analysis - {sentiment.upper()}", fontsize=20, fontweight="bold"
        )
        # 1. 평점 분포
        ax1 = axes[0, 0]
        rating_counts = sentiment_df["rating"].value_counts().sort_index()
        ax1.bar(
            rating_counts.index,
            rating_counts.values,
            color=get_sentiment_color(sentiment),
            alpha=0.7,
            edgecolor="black",
        )
        ax1.set_title(
            f"Rating Distribution ({sentiment})", fontsize=14, fontweight="bold"
        )
        ax1.set_xlabel("Rating", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.grid(axis="y", alpha=0.3)
        # 2. 감성 극성 분포
        ax2 = axes[0, 1]
        ax2.hist(
            sentiment_df["sentiment_polarity"],
            bins=30,
            color=get_sentiment_color(sentiment),
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_title(
            f"Sentiment Polarity Distribution ({sentiment})",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("Polarity Score", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.axvline(
            sentiment_df["sentiment_polarity"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Mean",
        )
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        # 3. 주요 키워드
        ax3 = axes[1, 0]
        if (
            sentiment in keywords_by_sentiment
            and len(keywords_by_sentiment[sentiment]) > 0
        ):
            keywords = keywords_by_sentiment[sentiment][:12]
            words = [kw[0] for kw in keywords]
            scores = [kw[1] for kw in keywords]
            y_pos = np.arange(len(words))
            ax3.barh(y_pos, scores, color=get_sentiment_color(sentiment), alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(words, fontsize=10)
            ax3.invert_yaxis()
            ax3.set_title(f"Top Keywords ({sentiment})", fontsize=14, fontweight="bold")
            ax3.set_xlabel("TF-IDF Score", fontsize=12)
            ax3.grid(axis="x", alpha=0.3)
        # 4. 워드 클라우드
        ax4 = axes[1, 1]
        text_data = " ".join(sentiment_df["review_text_clean"].tolist())
        if len(text_data) > 50:
            extended_stopwords = get_extended_stopwords()
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                stopwords=extended_stopwords,
                colormap=get_wordcloud_colormap(sentiment),
                max_words=80,
                relative_scaling=0.5,
                min_font_size=10,
            ).generate(text_data)
            ax4.imshow(wordcloud, interpolation="bilinear")
            ax4.axis("off")
            ax4.set_title(f"Word Cloud ({sentiment})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        output_path = os.path.join(output_folder, f"analysis_{sentiment}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✓ {sentiment} 시각화 저장: {output_path}")

    # 전체 요약 차트
    create_summary_chart(df, improvement_areas, output_folder)
    print(f"   ✓ 모든 시각화 완료")


def get_sentiment_color(sentiment):
    colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
    return colors.get(sentiment, "#95a5a6")


def get_wordcloud_colormap(sentiment):
    colormaps = {"positive": "Greens", "neutral": "Oranges", "negative": "Reds"}
    return colormaps.get(sentiment, "Blues")


def create_summary_chart(df, improvement_areas, output_folder):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Overall Analysis Summary", fontsize=20, fontweight="bold")
    ax1 = axes[0, 0]
    sentiment_counts = df["sentiment_category"].value_counts()
    colors = [get_sentiment_color(s) for s in sentiment_counts.index]
    wedges, texts, autotexts = ax1.pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
    ax1.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
    ax2 = axes[0, 1]
    rating_counts = df["rating"].value_counts().sort_index()
    bars = ax2.bar(
        rating_counts.index,
        rating_counts.values,
        color="#3498db",
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_title("Rating Distribution", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Rating", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax3 = axes[1, 0]
    if improvement_areas and len(improvement_areas) > 0:
        top_areas = improvement_areas[:8]
        categories = [area["description"] for area in top_areas]
        counts = [area["mention_count"] for area in top_areas]
        y_pos = np.arange(len(categories))
        bars = ax3.barh(y_pos, counts, color="#e74c3c", alpha=0.7, edgecolor="black")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(categories, fontsize=10)
        ax3.invert_yaxis()
        ax3.set_title(
            "Improvement Areas (Negative/Neutral Reviews)",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_xlabel("Mention Count", fontsize=11)
        ax3.grid(axis="x", alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(width)}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "No significant issues found",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax3.transAxes,
        )
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")
    ax4 = axes[1, 1]
    sentiment_data = []
    sentiment_labels = []
    for sentiment in ["positive", "neutral", "negative"]:
        data = df[df["sentiment_category"] == sentiment]["sentiment_polarity"].values
        if len(data) > 0:
            sentiment_data.append(data)
            sentiment_labels.append(sentiment)
    if sentiment_data:
        bp = ax4.boxplot(sentiment_data, labels=sentiment_labels, patch_artist=True)
        for patch, sentiment in zip(bp["boxes"], sentiment_labels):
            patch.set_facecolor(get_sentiment_color(sentiment))
            patch.set_alpha(0.7)
        ax4.set_title("Sentiment Polarity by Category", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Polarity Score", fontsize=12)
        ax4.set_xlabel("Sentiment Category", fontsize=12)
        ax4.grid(axis="y", alpha=0.3)
        ax4.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    plt.tight_layout()
    output_path = os.path.join(output_folder, "analysis_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✓ 요약 차트 저장: {output_path}")


# ============================================================================
# 6. 분석 결과 리포트 생성 (제품별 평균 평점 완전 제거)
# ============================================================================


def generate_analysis_report(
    df, keywords_by_sentiment, improvement_areas, output_folder
):
    print("\n[추가] 분석 결과 리포트 생성 중...")
    report_path = os.path.join(output_folder, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("다이어트 유산균 리뷰 분석 결과 리포트\n")
        f.write("=" * 80 + "\n\n")
        # 1. 전체 통계
        f.write("[ 1. 전체 통계 ]\n")
        f.write("-" * 80 + "\n")
        total_reviews = len(df)
        avg_rating = df["rating"].mean()
        positive_ratio = (df["sentiment_category"] == "positive").sum() / total_reviews
        negative_ratio = (df["sentiment_category"] == "negative").sum() / total_reviews
        neutral_ratio = (df["sentiment_category"] == "neutral").sum() / total_reviews
        avg_polarity = df["sentiment_polarity"].mean()
        f.write(f"총 리뷰 수: {total_reviews:,}개\n")
        f.write(f"평균 평점: {avg_rating:.2f} / 5.0\n")
        f.write(f"긍정 리뷰 비율: {positive_ratio:.1%}\n")
        f.write(f"중립 리뷰 비율: {neutral_ratio:.1%}\n")
        f.write(f"부정 리뷰 비율: {negative_ratio:.1%}\n")
        f.write(f"평균 감성 극성: {avg_polarity:.3f} (-1 ~ 1)\n\n")
        # 2. 감성별 주요 키워드
        f.write("[ 2. 감성별 주요 키워드 ]\n")
        f.write("-" * 80 + "\n")
        for sentiment in ["positive", "neutral", "negative"]:
            if sentiment in keywords_by_sentiment and keywords_by_sentiment[sentiment]:
                f.write(f"\n■ {sentiment.upper()} 리뷰 키워드:\n")
                keywords = keywords_by_sentiment[sentiment][:15]
                for i, (word, score) in enumerate(keywords, 1):
                    f.write(f"   {i:2d}. {word:30s} (TF-IDF: {score:.4f})\n")
        f.write("\n")
        # 3. 고객 불만 영역 분석
        f.write("[ 3. 고객 불만 영역 분석 (부정/중립 리뷰 기반) ]\n")
        f.write("-" * 80 + "\n")
        if improvement_areas:
            f.write(f"총 {len(improvement_areas)}개 불만 영역이 확인되었습니다.\n\n")
            for i, area in enumerate(improvement_areas, 1):
                f.write(f"[{i}순위] {area['description']}\n")
                f.write(f"  └ 언급 횟수: {area['mention_count']}회\n")
                f.write(f"  └ 영역 코드: {area['area']}\n")
                if area["sample_reviews"]:
                    f.write(f"  └ 고객 의견 샘플:\n")
                    for j, sample in enumerate(area["sample_reviews"][:2], 1):
                        snippet = sample["review_snippet"][:100] + "..."
                        f.write(f"     {j}. (평점 {sample['rating']}) {snippet}\n")
                f.write("\n")
        else:
            f.write("특별한 불만 영역이 발견되지 않았습니다.\n\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("분석 완료\n")
        f.write("=" * 80 + "\n")
    print(f"   ✓ 리포트 저장: {report_path}")
    return report_path


# ============================================================================
# 메인 실행 함수
# ============================================================================


def main():
    """메인 분석 파이프라인을 실행합니다."""
    reviews_path = "Amazon_review/amazon_reviews.csv"
    products_path = "Amazon_review/amazon_products.csv"
    if not os.path.exists(reviews_path):
        print(f"❌ 오류: {reviews_path} 파일을 찾을 수 없습니다.")
        print("   크롤링 코드를 먼저 실행하여 데이터를 수집하세요.")
        return
    if not os.path.exists(products_path):
        print(f"❌ 오류: {products_path} 파일을 찾을 수 없습니다.")
        return
    try:
        reviews_df, products_df = load_and_preprocess_data(reviews_path, products_path)
        reviews_df = perform_sentiment_analysis(reviews_df)
        keywords_by_sentiment = extract_keywords_by_sentiment(reviews_df, n_keywords=15)
        improvement_areas = extract_improvement_insights(reviews_df)
        create_visualizations(reviews_df, keywords_by_sentiment, improvement_areas)
        report_path = generate_analysis_report(
            reviews_df,
            keywords_by_sentiment,
            improvement_areas,
            "Amazon_review_analysis",
        )
        print("\n" + "=" * 80)
        print("✅ 리뷰 분석이 완료되었습니다!")
        print("=" * 80)
        print(f"\n📊 생성된 파일:")
        print(f"   - 긍정 리뷰 분석: Amazon_review_analysis/analysis_positive.png")
        print(f"   - 중립 리뷰 분석: Amazon_review_analysis/analysis_neutral.png")
        print(f"   - 부정 리뷰 분석: Amazon_review_analysis/analysis_negative.png")
        print(f"   - 전체 요약: Amazon_review_analysis/analysis_summary.png")
        print(f"   - 분석 결과 리포트: {report_path}")
        print(f"\n📈 분석 결과 요약:")
        print(f"   • 총 리뷰 수: {len(reviews_df):,}개")
        print(f"   • 평균 평점: {reviews_df['rating'].mean():.2f}/5.0")
        print(
            f"   • 긍정 리뷰: {(reviews_df['sentiment_category'] == 'positive').sum() / len(reviews_df):.1%}"
        )
        if improvement_areas:
            print(
                f"   • 주요 불만 영역 (1순위): {improvement_areas[0]['description']} ({improvement_areas[0]['mention_count']}회)"
            )
            print(f"   • 발견된 불만 영역: {len(improvement_areas)}개")
        else:
            print(f"   • 특별한 불만 영역 없음")
    except Exception as e:
        print(f"\n❌ 분석 중 오류가 발생했습니다: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
