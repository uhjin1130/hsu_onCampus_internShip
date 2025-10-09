import time
import random
import pandas as pd
import re
import datetime
import os

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains


def get_product_info(driver, wait, asin):
    """상품 상세 페이지에서 상품 정보를 추출합니다."""
    print(f"\n--- 상품 정보 수집 (ASIN: {asin}) ---")
    product_url = f"https://www.amazon.com/dp/{asin}"
    driver.get(product_url)

    product_data = {
        "product_id": asin,
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "platform": "Amazon",
        "product_name": "N/A",
        "price": 0.0,
        "is_stockout": True,
    }

    try:
        product_data["product_name"] = wait.until(
            EC.presence_of_element_located((By.ID, "productTitle"))
        ).text.strip()

        price_found = False
        price_selectors = [
            "div#corePrice_feature_div span.a-offscreen",
            "div#corePriceDisplay_desktop_feature_div span.a-offscreen",
            "span.priceToPay span.a-offscreen",
        ]
        for selector in price_selectors:
            try:
                price_text = driver.find_element(
                    By.CSS_SELECTOR, selector
                ).get_attribute("innerHTML")
                price_value = float(re.sub(r"[^\d.]", "", price_text))
                if price_value > 0:
                    product_data["price"] = price_value
                    price_found = True
                    break
            except (NoSuchElementException, ValueError):
                continue

        stock_info_element = driver.find_element(By.ID, "availability")
        if "in stock" in stock_info_element.text.lower():
            product_data["is_stockout"] = False

        return product_data

    except Exception as e:
        print(f"상품 정보 수집 중 오류 발생: {e}")
        return product_data


def scrape_reviews_for_product(
    driver,
    wait,
    actions,
    product_info,
    max_reviews_per_product,
    is_first_product,
    start_review_id,
):
    """주어진 단일 상품에 대한 리뷰를 여러 페이지에 걸쳐 수집합니다."""
    asin = product_info["product_id"]
    product_name = product_info["product_name"]

    print(f"--- '{product_name}' 리뷰 수집 시작 (최대 {max_reviews_per_product}개) ---")
    reviews_url = f"https://www.amazon.com/product-reviews/{asin}"
    driver.get(reviews_url)

    if is_first_product:
        print("\n" + "=" * 60)
        print("첫 상품 리뷰 페이지입니다. 로그인/CAPTCHA를 해결해주세요.")
        input("완료 후, 터미널로 돌아와 Enter 키를 누르세요: ")
        print("사용자 확인 완료. 첫 리뷰 수집을 시작합니다...")

    reviews_data = []
    page_count = 1
    current_review_id = start_review_id

    while len(reviews_data) < max_reviews_per_product:
        print(f"\n{page_count} 페이지 리뷰 수집 중...")
        try:
            wait.until(EC.presence_of_element_located((By.ID, "cm_cr-review_list")))
            time.sleep(random.uniform(2, 4))
        except TimeoutException:
            print("리뷰 목록을 찾을 수 없습니다.")
            break

        review_elements = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')
        if not review_elements:
            break

        for review in review_elements:
            try:
                body = (
                    review.find_element(
                        By.CSS_SELECTOR, '[data-hook="review-body"] span'
                    )
                    .text.replace("\n", " ")
                    .strip()
                )

                if body and not any(d["review_text"] == body for d in reviews_data):
                    likes = 0
                    try:
                        likes_text = review.find_element(
                            By.CSS_SELECTOR, '[data-hook="helpful-vote-statement"]'
                        ).text
                        match = re.search(r"(\d+)", likes_text)
                        if match:
                            likes = int(match.group(1))
                    except NoSuchElementException:
                        pass

                    reviews_data.append(
                        {
                            "review_id": current_review_id,
                            "product_id": asin,
                            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                            "platform": "Amazon",
                            "product_name": product_name,
                            "rating": float(
                                review.find_element(
                                    By.CSS_SELECTOR,
                                    '[data-hook="review-star-rating"] span.a-icon-alt',
                                )
                                .get_attribute("innerHTML")
                                .split(" ")[0]
                            ),
                            "review_text": body,
                            "likes": likes,
                        }
                    )
                    current_review_id += 1

                if len(reviews_data) >= max_reviews_per_product:
                    break
            except NoSuchElementException:
                continue

        print(
            f"'{product_name}' 상품의 리뷰를 현재까지 {len(reviews_data)}개 수집했습니다."
        )

        if len(reviews_data) >= max_reviews_per_product:
            break

        try:
            next_page_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a"))
            )
            last_known_reviews = driver.find_elements(
                By.CSS_SELECTOR, '[data-hook="review"]'
            )

            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", next_page_button
            )
            time.sleep(random.uniform(0.5, 1.5))
            actions.move_to_element(next_page_button).click().perform()

            if last_known_reviews:
                wait.until(EC.staleness_of(last_known_reviews[0]))

            page_count += 1
        except TimeoutException:
            print("'Next Page' 버튼을 찾을 수 없습니다. 마지막 페이지입니다.")
            break

    return reviews_data, current_review_id


def scrape_top_products(keyword, num_products_to_scrape=3, max_reviews_per_product=50):
    """아마존에서 Best Seller 상위 N개 상품의 정보와 리뷰를 수집합니다."""

    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=en-US")
    options.add_experimental_option("prefs", {"intl.accept_languages": "en,en_US"})

    driver = None
    try:
        driver = uc.Chrome(options=options)
        wait = WebDriverWait(driver, 20)
        actions = ActionChains(driver)

        driver.get("https://www.amazon.com/")
        print("Amazon.com에 접속합니다...")

        try:
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        '//button[text()="Continue shopping"] | //input[@data-action-type="DISMISS"]',
                    )
                )
            ).click()
            print("관문 페이지 버튼 클릭.")
        except TimeoutException:
            print("관문 페이지가 나타나지 않았습니다.")

        search_box = wait.until(
            EC.element_to_be_clickable((By.ID, "twotabsearchtextbox"))
        )
        search_box.clear()
        search_box.send_keys(keyword)
        search_box.submit()
        print(f"'{keyword}' 검색 완료.")

        try:
            print("\n'Best Sellers'로 정렬을 시도합니다...")
            sort_by_button = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "span.a-dropdown-container")
                )
            )
            sort_by_button.click()
            best_seller_option = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[text()='Best Sellers']"))
            )
            best_seller_option.click()
            print("'Best Sellers'로 정렬했습니다. 페이지 로드를 기다립니다...")
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div[data-component-type='s-search-result']")
                )
            )
            time.sleep(random.uniform(2, 4))
        except TimeoutException:
            print("오류: 'Best Sellers' 정렬 버튼을 찾지 못했습니다.")

        print(f"\n상위 {num_products_to_scrape}개 상품의 ASIN을 수집합니다...")
        product_containers = wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "div[data-component-type='s-search-result']")
            )
        )

        asins_to_scrape = [
            container.get_attribute("data-asin")
            for container in product_containers[:num_products_to_scrape]
            if container.get_attribute("data-asin")
        ]

        if not asins_to_scrape:
            return

        print(f"수집할 ASIN 목록: {asins_to_scrape}")

        all_products_data = []
        all_reviews_data = []

        is_first_product = True
        review_id_counter = 1

        for asin in asins_to_scrape:
            product_info = get_product_info(driver, wait, asin)
            if product_info:
                all_products_data.append(product_info)

            reviews, review_id_counter = scrape_reviews_for_product(
                driver,
                wait,
                actions,
                product_info,
                max_reviews_per_product,
                is_first_product,
                start_review_id=review_id_counter,
            )
            if reviews:
                all_reviews_data.extend(reviews)

            if is_first_product:
                is_first_product = False

        print("\n--- 모든 데이터 수집 완료. 파일 저장 시작 ---")

        # 저장 폴더 생성 및 경로 지정
        output_folder = "Amazon_review"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"'{output_folder}' 폴더를 생성했습니다.")

        if all_products_data:
            df_product = pd.DataFrame(all_products_data)
            df_product_to_save = df_product[
                [
                    "product_id",
                    "date",
                    "platform",
                    "product_name",
                    "price",
                    "is_stockout",
                ]
            ]

            # 파일 경로를 폴더와 함께 지정
            product_filepath = os.path.join(output_folder, "amazon_products.csv")
            df_product_to_save.to_csv(
                product_filepath, index=False, encoding="utf-8-sig"
            )
            print(f"'{product_filepath}' 파일 저장 완료.")

        if all_reviews_data:
            df_reviews = pd.DataFrame(all_reviews_data)
            df_reviews_to_save = df_reviews[
                [
                    "review_id",
                    "product_id",
                    "date",
                    "platform",
                    "product_name",
                    "rating",
                    "review_text",
                    "likes",
                ]
            ]

            # 파일 경로를 폴더와 함께 지정
            reviews_filepath = os.path.join(output_folder, "amazon_reviews.csv")
            df_reviews_to_save.to_csv(
                reviews_filepath, index=False, encoding="utf-8-sig"
            )
            print(f"'{reviews_filepath}' 파일 저장 완료.")

    except Exception as e:
        print(f"스크립트 실행 중 오류가 발생했습니다: {e}")
        if driver:
            driver.save_screenshot("debug_final_error.png")
    finally:
        if driver:
            input(
                "모든 작업이 완료되었습니다. Enter 키를 누르면 브라우저를 닫고 종료합니다..."
            )
            driver.quit()
            print("브라우저를 종료했습니다.")


if __name__ == "__main__":
    search_keyword = "diet probiotics"
    scrape_top_products(
        keyword=search_keyword, num_products_to_scrape=20, max_reviews_per_product=2000
    )
