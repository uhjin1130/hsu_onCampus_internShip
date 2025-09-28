import csv
import time
from datetime import datetime, timedelta
import re
import os

from pygooglenews import GoogleNews
import feedparser

# --- 설정 ---
KEYWORD_LIST = [
    "비엔날씬 다이어트 유산균", "비엔날씬 다이어트", "비엔날씬",
    "다이어트 유산균", "다이어트", "유산균",
    "비엔날씬 유산균", "유산균 다이어트", "다이어트 비엔날씬",

    # 2025. 9. 28 추가 키워드
    "체지방 유산균", "체지방유산균", "유산균 체지방",
    "비엔날씬다이어트유산균", "비엔날씬다이어트", "다이어트유산균", "유산균다이어트"
]
CSV_FILENAME = "DietNewsTrendDataCrawling.csv"
# ------------------------------------

# (parse_date, search_major_rss_feeds, search_google_news 함수는 변경 없음)
def parse_date(date_str):
    now = datetime.now()
    if "분 전" in date_str:
        mins = int(re.search(r'\d+', date_str).group())
        return now - timedelta(minutes=mins)
    elif "시간 전" in date_str:
        hours = int(re.search(r'\d+', date_str).group())
        return now - timedelta(hours=hours)
    elif "어제" in date_str:
        return now - timedelta(days=1)
    try:
        return datetime.strptime(date_str.replace('.', ''), '%Y%m%d')
    except ValueError:
        return now

def search_major_rss_feeds(keyword, limit=50):
    """주요 언론사 RSS 피드에서 키워드 관련 뉴스를 검색합니다."""
    print(f"   - 언론사 RSS 피드에서 '{keyword}' 검색 중...")
    rss_feeds = {
        '조선일보': 'https://www.chosun.com/arc/outboundfeeds/rss/?outputType=xml', '중앙일보': 'https://rss.joins.com/joins_news_list.xml',
        '동아일보': 'https://rss.donga.com/total.xml', '한겨레': 'https://www.hani.co.kr/rss/',
        '경향신문': 'https://www.khan.co.kr/rss/', 'YTN': 'https://www.ytn.co.kr/rss/ytn_news_major.xml'
    }
    all_articles = []
    for press, url in rss_feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if keyword.split(' ')[0] in entry.title or (hasattr(entry, 'summary') and keyword.split(' ')[0] in entry.summary):
                    published_time = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    all_articles.append({'title': entry.title, 'link': entry.link, 'source': press, 'published': published_time, 'source_portal': '언론사RSS'})
        except Exception: continue
    all_articles.sort(key=lambda x: x['published'], reverse=True)
    return all_articles[:limit]

def search_google_news(keyword, limit=50):
    """구글 뉴스에서 뉴스를 검색합니다."""
    print(f"   - 구글 뉴스에서 '{keyword}' 검색 중...")
    gn = GoogleNews(lang='ko', country='KR')
    search_result = gn.search(keyword, when='30d')
    articles = []
    for item in search_result['entries'][:limit]:
        articles.append({'title': item.title, 'link': item.link, 'source': item.source['title'],
                         'published': datetime.strptime(item.published, '%a, %d %b %Y %H:%M:%S %Z'),
                         'source_portal': '구글뉴스'})
    return articles


def news_collection_process(keyword, last_sequence, existing_links):
    """
    하나의 키워드에 대해 뉴스 '데이터'를 수집하여 '리스트'로 반환합니다.
    (파일에 직접 쓰지 않도록 변경)
    """
    all_articles = []
    all_articles.extend(search_google_news(keyword))
    all_articles.extend(search_major_rss_feeds(keyword))
    all_articles.sort(key=lambda x: x['published'], reverse=True)
    
    unique_articles = []
    seen_links = set()
    for article in all_articles:
        if article['link'] not in existing_links and article['link'] not in seen_links:
            unique_articles.append(article)
            seen_links.add(article['link'])
            existing_links.add(article['link']) # 전역 중복 링크셋에도 추가
    
    if not unique_articles:
        print(f"   -> '{keyword}'에 대한 새로운 뉴스가 없습니다.")
        return []

    newly_collected_rows = []
    for i, item in enumerate(unique_articles):
        current_sequence = last_sequence + i + 1
        
        news_data = {
            "trend_id": current_sequence,
            "date": item['published'].strftime('%Y-%m-%d %H:%M'),
            "platform": 'googlenews' if item['source_portal'] == '구글뉴스' else '언론사RSS',
            "keyword": keyword,
            # 2025-09-28 18:10: post_text에 제목과 링크를 줄바꿈(\n)으로 합쳐서 저장
            "post_text": f"{item['title']}\n{item['link']}",
            "hashtags": ""
        }
        newly_collected_rows.append(news_data)

    print(f"   -> '{keyword}' 검색 결과, {len(newly_collected_rows)}개의 새로운 뉴스를 수집했습니다.")
    return newly_collected_rows

if __name__ == "__main__":
    file_exists = os.path.isfile(CSV_FILENAME)
    last_sequence = 0
    existing_links = set()
    existing_rows = [] # 기존 데이터를 저장할 리스트
    
    if file_exists:
        print("기존 데이터를 읽어옵니다...")
        
        read_encoding = 'utf-8-sig'
        try:
            with open(CSV_FILENAME, 'r', encoding=read_encoding) as f:
                f.read()
        except UnicodeDecodeError:
            print("   -> UTF-8 읽기 실패. cp949 인코딩으로 다시 시도합니다 (Excel/한셀 저장 문제일 수 있습니다).")
            read_encoding = 'cp949'
        
        with open(CSV_FILENAME, 'r', newline='', encoding=read_encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                if row.get('trend_id'):
                    last_sequence = int(row['trend_id'])
                
                # 2025-09-28 18:10: post_text에서 줄바꿈으로 링크를 분리하여 중복 체크
                post_text_content = row.get('post_text', '')
                if '\n' in post_text_content:
                    link = post_text_content.split('\n')[-1] # 마지막 줄이 링크
                    if link.startswith('http'):
                        existing_links.add(link)

        print(f"총 {len(existing_rows)}개의 기존 데이터가 있습니다. (마지막 순번: {last_sequence})")

    # --- 새로운 데이터 수집 ---
    new_rows = []
    for keyword in KEYWORD_LIST:
        print("\n" + "─" * 70)
        print(f"키워드 '{keyword}' 검색 시작...")
        # 수집 함수가 파일에 쓰는 대신, 수집된 데이터를 반환하도록 변경
        collected_data = news_collection_process(keyword, last_sequence, existing_links)
        new_rows.extend(collected_data)
        last_sequence += len(collected_data) # 다음 키워드를 위해 마지막 순번 업데이트
        time.sleep(2)
    
    # --- 기존 데이터와 새 데이터를 합쳐서 '전체 덮어쓰기' ---
    print("\n" + "─" * 70)
    print(f"✅ 모든 키워드 검색 완료. 총 {len(new_rows)}개의 새로운 데이터를 추가합니다.")
    print("CSV 파일을 'UTF-8' 형식으로 새로 저장합니다...")

    # 데이터를 합침
    total_rows = existing_rows + new_rows

    # 파일을 'w'(쓰기) 모드로 열어 전체 데이터를 한 번에 저장 (인코딩 문제 근본 해결)
    with open(CSV_FILENAME, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["trend_id", "date", "platform", "keyword", "post_text", "hashtags"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(total_rows)

    print(f"✅ 작업 완료. 총 {len(total_rows)}개의 데이터가 {CSV_FILENAME}에 저장되었습니다.")