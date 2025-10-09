import pandas as pd
import matplotlib.pyplot as plt
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
import re
import os

nltk.download('stopwords')

df = pd.read_csv('Amazon_review/amazon_reviews.csv')
df = df.dropna(subset=['review_text', 'rating'])

def label_sentiment(r):
    if r >= 4: return 1
    elif r == 3: return 2
    else: return 0
df['sentiment'] = df['rating'].apply(label_sentiment)

stop_words = set(stopwords.words('english'))
custom_stopwords = {
    'product', 'use', 'good', 'like', 'one', 'also', 'would', 'really', 'well',
    'great','works','take','ive','im','best','recommend','work','helps','help','probiotic','probiotics'
}
stop_words = stop_words.union(custom_stopwords)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', str(text))
    text = text.lower()
    return ' '.join([w for w in text.split() if w not in stop_words])
df['clean_review'] = df['review_text'].apply(clean_text)

output_dir = 'Sentiment_Analysis_Results'
os.makedirs(output_dir, exist_ok=True)

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def plot_keybert_bar(corpus, sentiment_name, top_n=20):
    print(f"  [KeyBERT 시작] '{sentiment_name}' 리뷰 {len(corpus)}건 임베딩 처리 중...")
    doc = ' '.join(corpus)
    print("  [KeyBERT] 핵심 구 추출 중... (잠시만 기다리세요)")
    keywords = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1,3),
        stop_words=list(stop_words),
        top_n=top_n,
        use_maxsum=True,
        nr_candidates=60
    )
    print(f"  [KeyBERT] 키프레이즈 {len(keywords)}개 추출, 그래프 그리기...")
    phrases, scores = zip(*keywords)
    plt.figure(figsize=(10,6))
    plt.barh(phrases, scores, color='deepskyblue')
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} KeyPhrases - {sentiment_name}")
    plt.xlabel('Relevance score')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{sentiment_name}_KeyBERT_keywords.png")
    plt.close()
    print(f"  [저장 완료] {output_dir}/{sentiment_name}_KeyBERT_keywords.png")

# 콘솔창 실시간 메시지로 현재 분석 단계와 처리가 진행 중임을 명확히 알 수 있습니다.


sentiment_map = {0:'Negative', 1:'Positive', 2:'Neutral'}
min_reviews_threshold = 50

total = len(sentiment_map)
completed = 0

for label, name in sentiment_map.items():
    completed += 1
    subset = df[df['sentiment'] == label]
    if len(subset) < min_reviews_threshold:
        print(f"[Skip] {name} set too small: {len(subset)} reviews")
    else:
        print(f"[Processing] {name} sentiment with {len(subset)} reviews")
        plot_keybert_bar(subset['clean_review'], name)
    percent = int(100 * completed / total)
    print(f"진행률: {percent}% ({completed}/{total})")

print(f"분석 완료! 결과는 '{output_dir}' 폴더에 저장됐습니다.")
