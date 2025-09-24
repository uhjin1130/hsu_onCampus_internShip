import asyncio, re, pandas as pd
from playwright.async_api import async_playwright

# 네가 준 상품 상세 URL
URL = ("https://www.coupang.com/vp/products/8581669724?itemId=24874498963&vendorItemId=4872156860&q=%EB%B9%84%EC%97%94%EB%82%A0%EC%94%AC&searchId=6e8eed731533015&sourceType=search&itemsCount=36&searchRank=0&rank=0&traceId=mfp0uoey")

MAX_REVIEWS = 10   # 딱 10개만 수집
SCROLL_ROUNDS = 4  # 너무 과하게 스크롤하지 않도록 제한

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

async def click_review_tab(page):
    """ '상품평/리뷰' 탭 누르기 (페이지에 따라 텍스트가 다를 수 있어 후보를 돌림) """
    candidates = ["상품평", "리뷰", "상품리뷰", "후기", "평가"]
    for txt in candidates:
        # role 기반 → 텍스트 기반 순으로 시도
        try:
            await page.get_by_role("tab", name=re.compile(txt)).click(timeout=800)
            await page.wait_for_timeout(700)
            return True
        except:
            pass
        try:
            await page.locator(f"text={txt}").first.click(timeout=800)
            await page.wait_for_timeout(700)
            return True
        except:
            pass
    return False

async def get_reviews_scope(page):
    """ 리뷰가 iframe 안에 뜨는 경우가 있어 프레임도 탐색 """
    for f in page.frames:
        try:
            html = await f.content()
            if ("sdp-review" in html) or ("상품평" in html) or ("리뷰" in html):
                return f
        except:
            pass
    return page

async def main():
    async with async_playwright() as p:
        # 창을 띄워서 실제 크롬으로 열기
        browser = await p.chromium.launch(
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = await browser.new_context(locale="ko-KR", timezone_id="Asia/Seoul")
        page = await ctx.new_page()

        # 홈을 먼저 들렀다가 상세 진입(차단 회피에 조금 더 유리)
        await page.goto("https://www.coupang.com/", wait_until="domcontentloaded")
        await page.goto(URL, wait_until="domcontentloaded")

        # '상품평/리뷰' 탭 시도 (없으면 패스)
        await click_review_tab(page)

        # 스크롤 몇 번 (동적 로딩 대비)
        for _ in range(SCROLL_ROUNDS):
            await page.mouse.wheel(0, 20000)
            await page.wait_for_timeout(1200)

        scope = await get_reviews_scope(page)

        # ─────────────────────────────────────────────────────────
        # 쿠팡 리뷰 DOM (자주 쓰이는 클래스들 기반 후보)
        # 카드
        card_sel_candidates = [
            "div.sdp-review__article__list__review",
            "article.sdp-review__article__list__review",
            "li.sdp-review__article__list__review"
        ]
        card_locator = None
        for sel in card_sel_candidates:
            loc = scope.locator(sel)
            if await loc.count() > 0:
                card_locator = loc
                break
        if card_locator is None:
            print("[ERROR] 리뷰 카드를 찾지 못했어요. 탭 클릭/스크롤이 충분한지 확인하세요.")
            await page.screenshot(path="screenshot.png", full_page=True)
            with open("debug_page.html", "w", encoding="utf-8") as f:
                f.write(await scope.content())
            print("Saved screenshot.png & debug_page.html")
            await ctx.close(); await browser.close()
            return

        total = await card_locator.count()
        print(f"[INFO] 리뷰 카드 감지: {total}개 (그 중 {MAX_REVIEWS}개만 수집)")

        # 본문/별점 셀렉터 후보
        content_sel_candidates = [
            "div.sdp-review__article__list__review__content",
            "div.sdp-review__article__list__review__content__review-text",
            "div[class*='review__content']",
            "p"
        ]
        rating_sel_candidates = [
            "div.sdp-review__article__list__info__product-info__star-orange",
            "span.sdp-review__article__list__info__product-info__star-orange",
            "span[class*='star']",
        ]
        rows = []
        n = min(MAX_REVIEWS, total)
        for i in range(n):
            c = card_locator.nth(i)

            # 본문
            content = ""
            for sel in content_sel_candidates:
                try:
                    content = await c.locator(sel).inner_text(timeout=600)
                    if content and content.strip():
                        break
                except:
                    pass
            if not content:
                try:
                    content = await c.inner_text(timeout=600)
                except:
                    content = ""

            # 별점
            rating = ""
            for sel in rating_sel_candidates:
                try:
                    rating = await c.locator(sel).inner_text(timeout=400)
                    if rating and rating.strip():
                        break
                except:
                    pass

            rows.append({"rating": clean(rating), "content": clean(content)})

        df = pd.DataFrame(rows)
        df.to_csv("reviews10.csv", index=False, encoding="utf-8-sig")
        print("Saved reviews10.csv")

        await ctx.close(); await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
