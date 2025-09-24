import asyncio, re, pandas as pd
from playwright.async_api import async_playwright

# ✅ 그대로 유지
PRODUCT_URL = ("https://www.coupang.com/vp/products/8581669724"
               "?itemId=24874498963&vendorItemId=4872156860"
               "&q=%EB%B9%84%EC%97%94%EB%82%A0%EC%94%AC&searchId=bf3830861715093"
               "&sourceType=search&itemsCount=36&searchRank=0&rank=0&traceId=mfp03r5h")

MAX_REVIEWS = 10
SCROLL_ROUNDS = 4
WAIT = 1000  # ms

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

CONTENT_CANDS = [
    "div.sdp-review__article__list__review__content",
    "div.sdp-review__article__list__review__content__review-text",
    "div[class*='review__content']",
    "p"
]
RATING_CANDS = [
    "div.sdp-review__article__list__info__product-info__star-orange",
    "span.sdp-review__article__list__info__product-info__star-orange",
    "span[class*='star']"
]
CARD_SEL = "div.sdp-review__article__list__review"

async def main():
    async with async_playwright() as p:
        # 1) 이미 띄워둔 크롬(9222)에 붙기
        cdp = await p.chromium.connect_over_cdp("http://localhost:9222")

        # 2) 컨텍스트/페이지
        contexts = cdp.contexts
        context = contexts[0] if contexts else await cdp.new_context()
        await context.set_extra_http_headers({"Referer": "https://www.coupang.com/"})
        page = await context.new_page()

        # 3) 홈 -> 상세 진입 (직링크보다 안정적)
        await page.goto("https://www.coupang.com/", wait_until="domcontentloaded")
        await page.goto(PRODUCT_URL, wait_until="domcontentloaded")

        # 4) 스크롤로 리뷰 로드 유도
        for _ in range(SCROLL_ROUNDS):
            await page.mouse.wheel(0, 20000)
            await page.wait_for_timeout(WAIT)

        # 5) 리뷰 카드 탐색
        cards = page.locator(CARD_SEL)
        count = await cards.count()
        print(f"[INFO] 감지된 리뷰 카드: {count}개")

        rows = []
        n = min(MAX_REVIEWS, count)

        for i in range(n):
            c = cards.nth(i)

            # --- 디버그: 카드 전체 텍스트 일부 미리 찍기 ---
            try:
                raw = await c.inner_text(timeout=0)
            except:
                raw = ""
            preview = clean(raw)[:120]
            print(f"[DEBUG] 카드 {i} 원문 미리보기: {preview}")

            # --- 본문 추출 (여러 후보 즉시시도, 실패시 즉시 넘어감) ---
            content = ""
            for sel in CONTENT_CANDS:
                try:
                    t = await c.locator(sel).inner_text(timeout=0)
                    if t and t.strip():
                        content = t
                        break
                except:
                    pass
            if not content:
                # 최후: 카드 전체 텍스트 사용
                content = raw or ""

            # --- 별점 추출 (없어도 진행) ---
            rating = ""
            for sel in RATING_CANDS:
                try:
                    t = await c.locator(sel).inner_text(timeout=0)
                    if t and t.strip():
                        rating = t
                        break
                except:
                    pass

            rows.append({
                "rating": clean(rating),
                "content": clean(content)
            })

        # 6) 저장
        df = pd.DataFrame(rows)
        df.to_csv("reviews10.csv", index=False, encoding="utf-8-sig")
        print(f"Saved reviews10.csv ({len(df)} rows)")

if __name__ == "__main__":
    asyncio.run(main())
