import asyncio, re, pandas as pd
from playwright.async_api import async_playwright, TimeoutError

# ✅ 크롤링할 상품 URL (m.coupang.com 형태도 OK, pc용 그대로도 접속됨)
PRODUCT_URL = "https://www.coupang.com/vp/products/8581669724?itemId=24874498963&vendorItemId=4872156860&q=%EB%B9%84%EC%97%94%EB%82%A0%EC%94%AC&searchId=bf3830861715093&sourceType=search&itemsCount=36&searchRank=0&rank=0&traceId=mfp03r5h"

MAX_REVIEWS = 10      # 10개만 수집
SCROLL_ROUNDS = 4     # 스크롤 반복 횟수
WAIT = 1000           # 대기(ms)

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

async def main():
    async with async_playwright() as p:
        # 1) 브라우저 실행 (모바일 UA 위장 + HTTP/2 끄기)
        browser = await p.chromium.launch(
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled", "--disable-http2"]
        )
        context = await browser.new_context(
            ignore_https_errors=True,
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            user_agent=("Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                        "Version/16.0 Mobile/15E148 Safari/604.1"),
            viewport={"width":390, "height":844, "isMobile":True, "isLandscape":False}
        )
        await context.set_extra_http_headers({"Referer": "https://m.coupang.com/"})
        page = await context.new_page()

        # 2) 홈 → 상품 상세 진입
        await page.goto("https://m.coupang.com/", wait_until="domcontentloaded")
        await page.goto(PRODUCT_URL, wait_until="domcontentloaded")

        # 3) 스크롤로 리뷰 로드
        for _ in range(SCROLL_ROUNDS):
            await page.mouse.wheel(0, 20000)
            await page.wait_for_timeout(WAIT)

        # 4) 리뷰 카드 찾기 (모바일 DOM)
        cards = page.locator("div.sdp-review__article__list__review")
        count = await cards.count()
        print(f"[INFO] 감지된 리뷰 카드: {count}개")

        rows = []
        for i in range(min(MAX_REVIEWS, count)):
            c = cards.nth(i)
            # 본문
            try:
                content = await c.locator("div.sdp-review__article__list__review__content").inner_text()
            except:
                content = ""
            # 별점 (없으면 빈칸)
            try:
                rating = await c.locator("div.sdp-review__article__list__info__product-info__star-orange").inner_text()
            except:
                rating = ""
            rows.append({"rating": clean(rating), "content": clean(content)})

        # 5) CSV 저장
        df = pd.DataFrame(rows)
        df.to_csv("reviews10.csv", index=False, encoding="utf-8-sig")
        print("Saved reviews10.csv")

        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
