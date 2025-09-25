import csv
import time
from datetime import datetime, timedelta
import re
import os

# 웹 브라우저 제어 및 HTML 분석
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# 구글 뉴스 검색, AI 분석, 문장 분리
from pygooglenews import GoogleNews
# from transformers import pipeline # 감성 분석 비활성화
# import kss # 감성 분석 비활성화
import feedparser

# --- 설정 ---
# 💡 [수정됨] 9개의 키워드를 리스트로 관리
KEYWORD_LIST = [
    "비엔날씬 다이어트 유산균", "비엔날씬 다이어트", "비엔날씬",
    "다이어트 유산균", "다이어트", "유산균",
    "비엔날씬 유산균", "유산균 다이어트", "다이어트 비엔날씬"
]
CSV_FILENAME = "통합뉴스_분석_대량수집.csv"
# ------------------------------------

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

def search_major_rss_feeds(keyword, limit=50): # 더 많은 결과를 위해 limit 상향
    """주요 언론사 RSS 피드에서 키워드 관련 뉴스를 검색합니다."""
    # ... (이전과 동일) ...
    print(f"  - 언론사 RSS 피드에서 '{keyword}' 검색 중...")
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


def search_google_news(keyword, limit=50): # 더 많은 결과를 위해 limit 상향
    """구글 뉴스에서 뉴스를 검색합니다."""
    print(f"  - 구글 뉴스에서 '{keyword}' 검색 중...")
    gn = GoogleNews(lang='ko', country='KR')
    search_result = gn.search(keyword, when='30d') # 검색 기간을 30일로 확대
    articles = []
    for item in search_result['entries'][:limit]:
        articles.append({'title': item.title, 'link': item.link, 'source': item.source['title'],
                         'published': datetime.strptime(item.published, '%a, %d %b %Y %H:%M:%S %Z'),
                         'source_portal': '구글뉴스'})
    return articles


def news_collection_process(keyword, writer, last_sequence, existing_links):
    """하나의 키워드에 대해 뉴스를 수집하고 파일에 씁니다."""
    
    all_articles = []
    all_articles.extend(search_google_news(keyword))
    all_articles.extend(search_major_rss_feeds(keyword))
    all_articles.sort(key=lambda x: x['published'], reverse=True)
    
    unique_articles = []
    for article in all_articles:
        if article['link'] not in existing_links:
            unique_articles.append(article)
    
    if not unique_articles:
        print(f"  -> '{keyword}'에 대한 새로운 뉴스가 없습니다.")
        return 0 # 새로 추가된 개수 0

    new_items_count = 0
    for i, item in enumerate(unique_articles):
        current_sequence = last_sequence + i + 1
        
        news_data = {
            "순번": current_sequence,
            "기록일시": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "날짜": item['published'].strftime('%Y-%m-%d %H:%M'),
            "검색키워드": keyword,
            "출처포털": item['source_portal'],
            "출처": item['source'],
            "제목": item['title'],
            "기사평가": "분석 제외",
            "기사평가근거요약": "분석 제외",
            "링크": item['link']
        }
        writer.writerow(news_data)
        existing_links.add(item['link']) # 중복 방지를 위해 링크 추가
        new_items_count += 1

    print(f"  -> '{keyword}' 검색 결과, {new_items_count}개의 새로운 뉴스를 저장했습니다.")
    return new_items_count

if __name__ == "__main__":
    
    file_exists = os.path.isfile(CSV_FILENAME)
    last_sequence = 0
    existing_links = set()
    
    if file_exists:
        print("기존 데이터를 읽어옵니다...")
        with open(CSV_FILENAME, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                try:
                    link_index = header.index('링크')
                    for row in reader:
                        if row: 
                            last_sequence = int(row[0])
                            existing_links.add(row[link_index])
                except (ValueError, IndexError): pass
        print(f"총 {last_sequence}개의 기존 데이터가 있습니다. 이어서 저장합니다.")

    # 💡 [수정됨] 파일을 한 번만 열고 모든 키워드에 대해 작업 수행
    with open(CSV_FILENAME, "a", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["순번", "기록일시", "날짜", "검색키워드", "출처포털", "출처", "제목", "기사평가", "기사평가근거요약", "링크"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists or os.path.getsize(CSV_FILENAME) == 0:
            writer.writeheader()
        
        total_new_count = 0
        # 💡 [수정됨] 키워드 리스트를 순회하며 작업 수행
        for keyword in KEYWORD_LIST:
            print("\n" + "─" * 70)
            print(f"키워드 '{keyword}' 검색 시작...")
            added_count = news_collection_process(keyword, writer, last_sequence, existing_links)
            total_new_count += added_count
            last_sequence += added_count # 다음 키워드를 위해 마지막 순번 업데이트
            time.sleep(2) # 다음 키워드 검색 전 잠시 대기

    print("─" * 70); print(f"\n✅ 모든 키워드 검색이 완료되었습니다. 총 {total_new_count}개의 새로운 데이터를 추가했습니다.")