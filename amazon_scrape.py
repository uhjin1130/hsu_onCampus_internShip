import time
import random
import pandas as pd
import re

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

def scrape_amazon_reviews_with_manual_auth(keyword, max_reviews=50):
    print(f"'{keyword}' 키워드로 'Next Page' 버튼을 클릭하며 리뷰를 수집합니다 (최대 {max_reviews}개 목표).")
    
    options = uc.ChromeOptions()
    # 자동 종료 관련 옵션을 비활성화하여 안정성 확보
    options.add_argument('--disable-features=RendererCodeIntegrity')
    options.add_argument('--start-maximized')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument('--lang=en-US')
    options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
    
    driver = None
    try:
        driver = uc.Chrome(options=options)
        wait = WebDriverWait(driver, 20)
        actions = ActionChains(driver)

        # ... (접속, 관문 페이지 처리, 검색, ASIN 추출 로직은 이전과 동일) ...
        print("Amazon.com에 접속합니다...")
        driver.get("https://www.amazon.com/")

        try:
            continue_button_wait = WebDriverWait(driver, 10)
            continue_button = continue_button_wait.until(EC.element_to_be_clickable((By.XPATH, '//button[text()="Continue shopping"] | //input[@data-action-type="DISMISS"]')))
            print("관문 페이지 버튼 클릭.")
            continue_button.click()
            time.sleep(random.uniform(2, 3))
        except TimeoutException:
            print("관문 페이지가 나타나지 않았습니다.")

        print("검색창을 찾고 키워드를 입력합니다...")
        search_box = wait.until(EC.element_to_be_clickable((By.ID, "twotabsearchtextbox")))
        search_box.clear()
        search_box.send_keys(keyword)
        search_box.submit()
        print(f"'{keyword}' 검색을 완료했습니다.")

        print("상품 고유 ID(ASIN)를 추출합니다...")
        asin = None
        try:
            first_product_container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-component-type='s-search-result']")))
            asin = first_product_container.get_attribute('data-asin')
            if asin:
                print(f"성공: ASIN을 직접 찾았습니다: {asin}")
        except TimeoutException:
            print("오류: 검색 결과에서 상품을 찾지 못했습니다.")
            return

        if not asin or len(asin) != 10:
            print("오류: 유효한 ASIN을 추출하지 못했습니다.")
            return

        reviews_url = f"https://www.amazon.com/product-reviews/{asin}"
        print(f"리뷰 1페이지로 이동합니다: {reviews_url}")
        driver.get(reviews_url)

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★★ 캡차 또는 로그인 처리를 위한 사용자 대기 (핵심 변경점) ★★★★★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        print("\n" + "="*60)
        print("브라우저를 확인하고, 만약 로그인 또는 CAPTCHA(퍼즐) 화면이 나타났다면")
        print("직접 해결하신 후, 아래 터미널(명령 프롬프트)로 돌아와 'y'를 입력하고 Enter 키를 누르세요.")
        print("정상적인 리뷰 페이지가 보인다면 바로 'y'를 입력하고 Enter를 누르시면 됩니다.")
        print("="*60)
        
        while True:
            user_input = input("준비가 되면 'y'를 입력하고 Enter를 누르세요: ")
            if user_input.lower() == 'y':
                print("사용자 확인 완료. 리뷰 수집을 시작합니다...")
                break
        
        reviews_data = []
        page_count = 1
        while len(reviews_data) < max_reviews:
            print(f"\n{page_count} 페이지 리뷰 수집을 시작합니다...")
            # (이후 리뷰 수집 및 페이지 이동 로직은 이전과 동일)
            try:
                review_list_container = wait.until(
                    EC.presence_of_element_located((By.ID, "cm_cr-review_list"))
                )
                time.sleep(random.uniform(2, 4))
            except TimeoutException:
                print("리뷰 목록을 찾을 수 없습니다. 수집을 종료합니다.")
                break

            review_elements = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')
            if not review_elements:
                print("현재 페이지에 리뷰가 없습니다.")
                break
                
            for review in review_elements:
                try:
                    reviewer = review.find_element(By.CSS_SELECTOR, '.a-profile-name').text
                    rating = review.find_element(By.CSS_SELECTOR, '[data-hook="review-star-rating"] span.a-icon-alt').get_attribute('innerHTML').split(' ')[0]
                    title = review.find_element(By.CSS_SELECTOR, '[data-hook="review-title"] span').text
                    date = review.find_element(By.CSS_SELECTOR, '[data-hook="review-date"]').text
                    body = review.find_element(By.CSS_SELECTOR, '[data-hook="review-body"] span').text.replace('\n', ' ')
                    
                    is_duplicate = any(d['Body'] == body and d['Reviewer'] == reviewer for d in reviews_data)
                    if not is_duplicate:
                        reviews_data.append({'Reviewer': reviewer, 'Rating': rating, 'Title': title, 'Date': date, 'Body': body})

                    if len(reviews_data) >= max_reviews:
                        break
                except NoSuchElementException:
                    continue
            
            print(f"현재까지 {len(reviews_data)}개의 리뷰를 수집했습니다.")

            if len(reviews_data) >= max_reviews:
                break

            try:
                next_page_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a")))
                last_known_reviews = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')
                
                print("'Next Page' 버튼을 클릭합니다...")
                driver.execute_script("arguments[0].scrollIntoView();", next_page_button)
                time.sleep(random.uniform(0.5, 1))
                actions.move_to_element(next_page_button).click().perform()
                
                if last_known_reviews:
                    print("페이지가 완전히 로드될 때까지 대기합니다...")
                    wait.until(EC.staleness_of(last_known_reviews[0]))
                
                page_count += 1
                time.sleep(random.uniform(2, 5))
            except TimeoutException:
                print("'Next Page' 버튼을 찾을 수 없습니다. 마지막 페이지인 것 같습니다.")
                break

        print(f"\n총 {len(reviews_data)}개의 리뷰 수집을 완료했습니다.")
        
        if reviews_data:
            df = pd.DataFrame(reviews_data)
            output_filename = 'amazon_reviews_manual_auth.csv'
            df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"'{output_filename}' 파일로 저장을 완료했습니다.")

    except Exception as e:
        print(f"스크립트 실행 중 오류가 발생했습니다: {e}")
        if driver: 
            driver.save_screenshot('debug_final_error.png')
    finally:
        if driver:
            # 스크립트가 끝나도 브라우저가 바로 닫히지 않도록 input() 추가 (결과 확인용)
            input("모든 작업이 완료되었습니다. Enter 키를 누르면 브라우저를 닫고 종료합니다...")
            driver.quit()
            print("브라우저를 종료했습니다.")

if __name__ == '__main__':
    search_keyword = "diet probiotics"
    reviews_to_scrape = 50 
    scrape_amazon_reviews_with_manual_auth(search_keyword, reviews_to_scrape)