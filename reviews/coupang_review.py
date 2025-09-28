# -*- coding: utf-8 -*-
"""
coupang_review.py
- 1차: Playwright APIRequestContext(브라우저 네트워크 스택)로 쿠팡 리뷰 API 직통 호출
- 2차: 응답이 JSON이 아니라 HTML(리뷰 DOM)일 때, HTML 파싱으로 리뷰 추출
- 3차: 그래도 부족/실패 시, 브라우저 띄워서 페이지 내부 fetch() → XHR 스니핑 폴백
- CSV 저장(utf-8-sig), 상세 로그, pagesize/timeout 조절, headful/헤드리스 선택
"""

import os, re, math, time, asyncio, argparse, pandas as pd
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ===== 기본 설정 =====
DEFAULT_URL    = "https://www.coupang.com/vp/products/8581669724?itemId=24874498963"
DEFAULT_COUNT  = 100
DEFAULT_PAGESZ = 20      # 먼저 20으로 성공 확인 → 이후 50/100으로 올리기
HEADLESS       = True    # --headful 옵션 주면 False로 변경
USER_DATA_DIR  = os.path.expanduser("~/.pw-coupang")
COUPANG_COOKIE = os.getenv("COUPANG_COOKIE", "").strip()
DEBUG_DUMP = False 

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

BASE = "https://www.coupang.com/vp/product/reviews"

# ===== 유틸 =====
def extract_product_id(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    return parts[parts.index("products")+1] if "products" in parts else ""

def norm_date(s: str) -> str:
    if not s: return s
    m = re.search(r"(\d{4})[.\-\/](\d{1,2})[.\-\/](\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    return s

def to_rows(product_id: str, content: list):
    rows = []
    for rv in content or []:
        rows.append({
            "product_id": product_id,
            "date": norm_date(rv.get("createdAt", "")),
            "platform": "Coupang",
            "product_name": rv.get("productName", ""),
            "rating": rv.get("rating", ""),
            "review_text": rv.get("reviewText", ""),
            "likes": rv.get("likeCount", 0),
        })
    return rows

def save_csv(rows, product_id_for_id=None):
    """
    rows: 수집된 리뷰 dict 리스트
    product_id_for_id: review_id 생성 시 사용할 product_id (없으면 각 row의 product_id 사용)
    """
    # review_id 채워 넣기: cp_<product_id>_<000001> …
    fixed_rows = []
    for i, r in enumerate(rows, start=1):
        pid = product_id_for_id or r.get("product_id", "")
        rid = r.get("review_id")
        if not rid:
            rid = f"cp_{pid}_{i:06d}"
        # 컬럼/타입 가볍게 정리
        fixed_rows.append({
            "review_id": rid,
            "product_id": pid,
            "date": r.get("date", ""),
            "platform": r.get("platform", "Coupang"),
            "product_name": r.get("product_name", ""),
            "rating": int(r.get("rating") or 0) if str(r.get("rating", "")).strip() else None,
            "review_text": r.get("review_text", ""),
            "likes": int(r.get("likes") or 0),
        })

    df = pd.DataFrame(fixed_rows, columns=[
        "review_id","product_id","date","platform","product_name","rating","review_text","likes"
    ])

    day_tag = datetime.now().strftime("%Y%m%d")
    out = f"reviews_coupang_{day_tag}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out, len(df)


def parse_cookie_string(cookie_str: str):
    """'a=1; b=2' → playwright add_cookies 형식으로 변환"""
    cookies = []
    if not cookie_str:
        return cookies
    pairs = [p.strip() for p in cookie_str.split(";") if p.strip()]
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
            cookies.append({
                "name": k.strip(),
                "value": v.strip(),
                "domain": ".coupang.com",
                "path": "/",
                "httpOnly": False,
                "secure": True,
            })
    return cookies

# ===== (핵심) HTML 응답 파서 =====
def parse_reviews_from_html(html: str):
    """
    쿠팡 리뷰 리스트 HTML(Non-JSON 응답)을 파싱해서 JSON과 동일한 구조로 반환.
    반환: [{"createdAt": "...", "productName": "...", "rating": 4, "reviewText": "...", "likeCount": 0}, ...]
    """
    soup = BeautifulSoup(html, "lxml")
    items = []

    # 리뷰 블록 컨테이너 선택
    review_blocks = soup.select(".sdp-review__article__list.js_reviewArticleReviewList")
    if not review_blocks:
        review_blocks = soup.select(".sdp-review__article__list")

    # 개별 리뷰 article 선택 (페이지 버전에 따라 article 태그가 직접 나올 수 있음)
    if not review_blocks:
        review_blocks = soup.select("article.sdp-review__article__list")

    for block in review_blocks:
        # 평점
        rating = None
        rating_el = block.select_one(".js_reviewArticleRatingValue")
        if rating_el:
            val = rating_el.get("data-rating") or rating_el.get_text(strip=True)
            if val and re.match(r"^\d+$", val):
                rating = int(val)

        # 날짜
        date_el = block.select_one(".sdp-review__article__list__info__product-info__reg-date")
        created_at = date_el.get_text(strip=True) if date_el else ""

        # 상품명
        name_el = block.select_one(".sdp-review__article__list__info__product-info__name")
        product_name = name_el.get_text(" ", strip=True) if name_el else ""

        # 본문
        text_el = (
            block.select_one(".sdp-review__article__list__review__content") or
            block.select_one(".js_reviewArticleContent") or
            block.select_one(".sdp-review__article__list__review")
        )
        review_text = text_el.get_text("\n", strip=True) if text_el else ""

        # 도움돼요(좋아요)
        like_el = block.select_one(".js_reviewArticleHelpfulCount")
        like_txt = (like_el.get_text(strip=True) if like_el else "") or (like_el.get("data-count") if like_el else "")
        try:
            like_count = int(re.sub(r"[^\d]", "", like_txt)) if like_txt else 0
        except:
            like_count = 0

        # 하나의 아이템 구성
        items.append({
            "createdAt": created_at,
            "productName": product_name,
            "rating": rating,
            "reviewText": review_text,
            "likeCount": like_count,
        })

    return items

# ===== 1) API 직통 (JSON/HTML 모두 처리) =====
async def fetch_page_api(reqctx, product_url: str, product_id: str, page: int, size: int,
                         connect_timeout_ms=5000, read_timeout_ms=30000):
    params = {
        "productId": product_id,
        "page": page,
        "size": size,
        "sortBy": "ORDER_SCORE_ASC",
        "ratings": "",
        "q": "",
    }
    headers = {
        "user-agent": UA,
        "accept": "application/json, text/plain, */*",
        "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "referer": product_url,
        "origin": "https://www.coupang.com",
        "sec-fetch-site": "same-origin",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "sec-ch-ua": '"Chromium";v="124", "Not(A:Brand";v="24", "Google Chrome";v="124"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "x-requested-with": "XMLHttpRequest",
    }
    if not COUPANG_COOKIE:
        raise SystemExit("❌ COUPANG_COOKIE 환경변수가 비어 있습니다. 쿠키를 export 해주세요.")
    headers["cookie"] = COUPANG_COOKIE

    timeout_ms = connect_timeout_ms + read_timeout_ms
    print(f"→ requesting page={page} (size={size}, timeout={timeout_ms}ms)", flush=True)

    t0 = time.time()
    resp = await reqctx.get(BASE, params=params, headers=headers, timeout=timeout_ms)
    elapsed = time.time() - t0

    if resp.status == 403:
        raise SystemExit("❌ 403 Forbidden: 쿠키가 없거나 만료/부족합니다. 새 쿠키로 다시 export 해주세요.")
    if resp.status not in (200, 206):
        # 오류 응답일 때만 덤프 (옵션 켠 경우)
        if DEBUG_DUMP:
            try:
                text = await resp.text()
                with open(f"debug_http_{resp.status}_p{page}.html", "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
        raise RuntimeError(f"HTTP {resp.status} on page={page}")

    ctype = (resp.headers.get("content-type") or "").lower()

    if "application/json" in ctype:
        data = await resp.json()
        print(f"  • page={page} | status={resp.status} | items={len(data.get('content', []))} | {elapsed:.2f}s", flush=True)
        return data

    if "text/html" in ctype:
        # ✅ 정상 동작: 파일로 저장하지 않고 바로 파싱만
        html = await resp.text()
        parsed = parse_reviews_from_html(html)
        print(f"  • page={page} | status={resp.status} | HTML parsed items={len(parsed)} | {elapsed:.2f}s", flush=True)
        return {"content": parsed}

    # 그 외 타입: 오류로 처리. 원하면(--debug) 덤프 남김
    text = await resp.text()
    if DEBUG_DUMP:
        try:
            with open(f"debug_unknown_ct_p{page}.txt", "w", encoding="utf-8") as f:
                f.write(text[:2000])
        except Exception:
            pass
    raise RuntimeError(f"Unsupported content-type for reviews: {ctype}")

# ===== 2) 브라우저 폴백: 페이지 내부 fetch 우선, 이후 XHR 스니핑 =====
async def js_fetch_reviews(page, product_id: str, start_page: int, page_size: int, max_pages: int):
    """페이지 컨텍스트에서 fetch('/vp/product/reviews?...') 직접 호출"""
    results = []
    for pg in range(start_page, start_page + max_pages):
        data = await page.evaluate(
            """async ({pid, pg, sz}) => {
                const url = `/vp/product/reviews?productId=${pid}&page=${pg}&size=${sz}&sortBy=ORDER_SCORE_ASC&ratings=&q=`;
                const res = await fetch(url, {
                    headers: {
                        "x-requested-with": "XMLHttpRequest",
                        "accept": "application/json, text/plain, */*",
                    },
                    credentials: "include"
                });
                const ctype = res.headers.get('content-type') || '';
                if (!res.ok || !ctype.includes('application/json')) {
                    return { ok:false, status: res.status, ctype, text: await res.text() };
                }
                return { ok:true, json: await res.json() };
            }""",
            {"pid": product_id, "pg": pg, "sz": page_size}
        )
        if not data.get("ok"):
            break
        content = (data["json"] or {}).get("content", []) or []
        if not content:
            break
        results.extend(content)
        await page.wait_for_timeout(120)
    return results

async def fetch_by_sniff(product_url: str, target_count: int, page_size: int):
    """퍼시스턴트 컨텍스트로 접속 → (1) 페이지 내부 fetch → (2) 탭/스크롤로 XHR 스니핑"""
    rows = []
    product_id = extract_product_id(product_url)

    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-dev-shm-usage",
            ],
            viewport={"width": 1280, "height": 900},
            user_agent=UA,
        )
        page = await browser.new_page()

        # 쿠키 주입
        if COUPANG_COOKIE:
            try:
                cookies = parse_cookie_string(COUPANG_COOKIE)
                if cookies:
                    await browser.add_cookies(cookies)
                    await page.context.add_cookies(cookies)
            except Exception:
                pass

        # XHR 스니핑
        async def on_response(resp):
            try:
                if "/vp/product/reviews" in resp.url:
                    data = await resp.json()
                    got = to_rows(product_id, data.get("content", []))
                    if got:
                        rows.extend(got)
                        print(f"  • sniffed items += {len(got)} (total={len(rows)})", flush=True)
            except Exception:
                pass
        page.on("response", on_response)

        # 모바일 → 데스크톱 순서로 시도
        candidates = [
            product_url.replace("www.coupang.com", "m.coupang.com"),
            product_url,
        ]

        for url in candidates:
            try:
                print(f"→ [sniff] goto: {url}", flush=True)
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)

                # (1) 페이지 내부 fetch
                got_json = await js_fetch_reviews(
                    page, product_id, start_page=0,
                    page_size=page_size, max_pages=math.ceil(target_count/page_size)
                )
                if got_json:
                    rows.extend(to_rows(product_id, got_json))
                    print(f"  • fetch() items += {len(got_json)} (total={len(rows)})", flush=True)

                if len(rows) >= target_count:
                    break

                # (2) 리뷰 탭 열고 더보기/스크롤
                for sel in [
                    'a:has-text("리뷰")',
                    'a[data-more-tab="review"]',
                    '#btfTab a[href*="review"]',
                    'button:has-text("리뷰")',
                ]:
                    try:
                        el = await page.query_selector(sel, timeout=2500)
                        if el:
                            await el.click()
                            break
                    except Exception:
                        continue

                attempt = 0
                while len(rows) < target_count and attempt < 120:
                    clicked = False
                    for sel in [
                        'button:has-text("더보기")',
                        'a:has-text("더보기")',
                        'button:has-text("리뷰 더보기")',
                        'a:has-text("리뷰 더보기")',
                    ]:
                        try:
                            btn = await page.query_selector(sel, timeout=800)
                            if btn:
                                await btn.click()
                                clicked = True
                                break
                        except Exception:
                            pass
                    if not clicked:
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                    await page.wait_for_timeout(180)
                    attempt += 1

                break
            except PWTimeout:
                print("  ! [sniff] goto timeout, next candidate...", flush=True)
                continue
            except Exception as e:
                print(f"  ! [sniff] error: {e}", flush=True)
                continue

        await page.close()
        await browser.close()

    return rows[:target_count]

# ===== 메인 실행 =====
async def run(url: str, target_count: int, page_size: int):
    print("=== Coupang Reviews (API→Sniff fallback) ===", flush=True)
    print(f"- URL       : {url}", flush=True)
    print(f"- Target    : {target_count} (page size {page_size})", flush=True)
    print(f"- Cookie    : len={len(COUPANG_COOKIE)} (set={'yes' if COUPANG_COOKIE else 'no'})", flush=True)

    product_id = extract_product_id(url)
    print(f"- product_id: {product_id}", flush=True)
    if not product_id:
        raise SystemExit("PRODUCT_URL에서 productId 추출 실패")

    rows = []
    t_start = time.time()

    # (A) API 직통 (JSON/HTML 모두 처리)
    try:
        async with async_playwright() as p:
            req = await p.request.new_context(
                extra_http_headers={
                    "user-agent": UA,
                    "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                }
            )
            need_pages = max(1, math.ceil(target_count / page_size))
            total = 0
            for pg in range(need_pages):
                attempt, max_attempts = 1, 3
                while True:
                    try:
                        data = await fetch_page_api(req, url, product_id, pg, page_size)
                        break
                    except Exception as e:
                        if attempt >= max_attempts:
                            print(f"  ! page={pg} error(final): {e}", flush=True)
                            data = {"content": []}
                            break
                        backoff = 1.5 ** attempt
                        print(f"  ! page={pg} error: {e} → retry in {backoff:.1f}s (attempt {attempt}/{max_attempts-1})", flush=True)
                        await asyncio.sleep(backoff)
                        attempt += 1

                content = data.get("content", [])
                if not content:
                    print("  ! 더 이상 컨텐츠가 없습니다. (API)", flush=True)
                    break

                got = to_rows(product_id, content)
                rows.extend(got)
                total += len(got)
                if total % 20 == 0:
                    print(f"    - collected(API): {total}", flush=True)
                if total >= target_count:
                    break

            await req.dispose()
    except Exception as e:
        print(f"  ! API path failed early: {e}", flush=True)

    # (B) 부족하면 브라우저 폴백
    if len(rows) < target_count:
        need = target_count - len(rows)
        print(f"→ Fallback to browser (need {need} more)...", flush=True)
        more = await fetch_by_sniff(url, need, page_size)
        rows.extend(more)

    out_path, saved = save_csv(rows[:target_count], product_id_for_id=product_id)
    print(f"\n✅ Saved {saved} rows → {out_path}", flush=True)
    print(f"⏱️ Elapsed: {time.time() - t_start:.2f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="오류 발생 시에만 응답 덤프 파일 저장")
    ap.add_argument("--url", type=str, default=DEFAULT_URL)
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--pagesize", type=int, default=DEFAULT_PAGESZ)
    ap.add_argument("--headful", action="store_true", help="폴백 스니핑 시 브라우저 창 표시")
    args = ap.parse_args()

    global HEADLESS, DEBUG_DUMP
    if args.headful:
        HEADLESS = False
    DEBUG_DUMP = bool(args.debug)
    url = args.url
    target = max(1, args.count)
    pagesz = max(1, min(100, args.pagesize))

    asyncio.run(run(url, target, pagesz))

if __name__ == "__main__":
    main()
