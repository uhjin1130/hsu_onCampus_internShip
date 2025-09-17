import time
import random
import pandas as pd
import re

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
# ActionChains 추가: 더 사람다운 클릭을 위해
from selenium.webdriver.common.action_chains import ActionChains

def scrape_amazon_reviews_by_clicking(keyword, max_reviews=50):
    print(f"'{keyword}' 키워드로 'Next Page' 버튼을 클릭하며 리뷰를 수집합니다 (최대 {max_reviews}개 목표).")
    
    options = uc.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument('--lang=en-US')
    options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
    
    driver = None
    try:
        driver = uc.Chrome(options=options)
        wait = WebDriverWait(driver, 20)
        actions = ActionChains(driver)

        print("Amazon.com에 접속합니다...")
        driver.get("https://www.amazon.com/")

        # 관문 페이지 처리 및 검색, ASIN 추출 로직
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
        
        reviews_data = []
        
        # 클릭과 스마트 대기를 이용한 페이지 이동 로직
        page_count = 1
        while len(reviews_data) < max_reviews:
            print(f"\n{page_count} 페이지 리뷰 수집을 시작합니다...")
            try:
                # 현재 페이지의 리뷰 목록이 완전히 로드될 때까지 대기
                review_list_container = wait.until(
                    EC.presence_of_element_located((By.ID, "cm_cr-review_list"))
                )
                # 페이지 로딩 후 자바스크립트가 리뷰를 렌더링할 시간을 추가로 줌
                time.sleep(random.uniform(2, 4))

            except TimeoutException:
                print("리뷰 목록을 찾을 수 없습니다. 수집을 종료합니다.")
                break

            review_elements = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')
            if not review_elements:
                print("현재 페이지에 리뷰가 없습니다.")
                break
                
            # 현재 페이지 리뷰 수집
            for review in review_elements:
                try:
                    reviewer = review.find_element(By.CSS_SELECTOR, '.a-profile-name').text
                    rating = review.find_element(By.CSS_SELECTOR, '[data-hook="review-star-rating"] span.a-icon-alt').get_attribute('innerHTML').split(' ')[0]
                    title = review.find_element(By.CSS_SELECTOR, '[data-hook="review-title"] span').text
                    date = review.find_element(By.CSS_SELECTOR, '[data-hook="review-date"]').text
                    body = review.find_element(By.CSS_SELECTOR, '[data-hook="review-body"] span').text.replace('\n', ' ')
                    
                    # 중복 리뷰 방지 (리뷰 텍스트와 작성자 기준)
                    is_duplicate = any(d['Body'] == body and d['Reviewer'] == reviewer for d in reviews_data)
                    if not is_duplicate:
                        reviews_data.append({'Reviewer': reviewer, 'Rating': rating, 'Title': title, 'Date': date, 'Body': body})

                    if len(reviews_data) >= max_reviews:
                        break
                except NoSuchElementException:
                    continue # 리뷰 내용 중 일부가 없는 경우 건너뛰기
            
            print(f"현재까지 {len(reviews_data)}개의 리뷰를 수집했습니다.")

            if len(reviews_data) >= max_reviews:
                print(f"목표한 리뷰 개수({max_reviews}개)에 도달하여 수집을 중단합니다.")
                break

            # 'Next Page' 버튼 찾기 및 클릭
            try:
                # 다음 페이지 버튼을 찾음 (보통 li.a-last a 태그)
                next_page_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a"))
                )
                
                # 클릭하기 전, 현재 페이지의 리뷰들을 기억
                last_known_reviews = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')
                
                print("'Next Page' 버튼을 클릭합니다...")
                # 사람처럼 보이게 스크롤 후 클릭
                driver.execute_script("arguments[0].scrollIntoView();", next_page_button)
                time.sleep(random.uniform(0.5, 1))
                actions.move_to_element(next_page_button).click().perform()
                
                # 페이지가 실제로 바뀔 때까지 (이전 페이지의 요소가 사라질 때까지) 대기
                if last_known_reviews:
                    print("페이지가 완전히 로드될 때까지 대기합니다...")
                    wait.until(EC.staleness_of(last_known_reviews[0]))
                
                page_count += 1
                time.sleep(random.uniform(2, 5))

            except TimeoutException:
                print("'Next Page' 버튼을 찾을 수 없습니다. 마지막 페이지인 것 같습니다.")
                break # 루프 종료

        print(f"\n총 {len(reviews_data)}개의 리뷰 수집을 완료했습니다.")
        
        if reviews_data:
            df = pd.DataFrame(reviews_data)
            output_filename = 'amazon_reviews_by_clicking.csv'
            df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"'{output_filename}' 파일로 저장을 완료했습니다.")

    except Exception as e:
        print(f"스크립트 실행 중 오류가 발생했습니다: {e}")
        if driver: 
            driver.save_screenshot('debug_final_error.png')
    finally:
        if driver:
            driver.quit()
            print("브라우저를 종료했습니다.")

if __name__ == '__main__':
    search_keyword = "diet probiotics"
    reviews_to_scrape = 50 
    scrape_amazon_reviews_by_clicking(search_keyword, reviews_to_scrape)