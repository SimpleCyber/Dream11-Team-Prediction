from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import csv
import time

# Configure Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Disabled for debugging
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920x1080")

# Set path to chromedriver
service = Service(r'E:\pc\python\srab\player data\chromedriver.exe')

def scrape_partnership_data():
    driver = webdriver.Chrome(service=service, options=chrome_options)
    url = "https://www.espncricinfo.com/records/trophy/fow-highest-partnerships-for-any-wicket/indian-premier-league-117"
    
    try:
        driver.get(url)
        print("Page loaded successfully")
        time.sleep(3)  # Initial page load wait

        # Handle cookie consent if it appears
        try:
            cookie_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button#wzrk-cancel"))
            )
            cookie_btn.click()
            print("Cookie consent dismissed")
            time.sleep(1)
        except:
            print("No cookie consent found")

        # Click the "Last 10 Years" filter using the provided XPath
        try:
            last_10_years = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="main-container"]/div[4]/div/div[3]/div[1]/div[1]/div/div[2]/div/div/span[6]'))
            )
            print("Found Last 10 Years filter")
            last_10_years.click()
            print("Clicked Last 10 Years filter")
            time.sleep(3)  # Wait for data to load
        except Exception as e:
            print(f"Error clicking Last 10 Years filter: {str(e)}")
            return

        # Get table data
        try:
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.ds-table"))
            )
            print("Table located successfully")
            
            # Extract headers
            headers = [th.text for th in table.find_elements(By.CSS_SELECTOR, "thead th")][:8]
            print(f"Headers: {headers}")
            
            # Extract rows
            data = []
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            print(f"Found {len(rows)} rows")
            
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 8:
                    row_data = [col.text.replace('\n', ' ') for col in cols[:8]]
                    data.append(row_data)
                    print(f"Added row: {row_data}")

            # Save to CSV
            csv_file = 'ipl_partnerships_last10years.csv'
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(data)
            
            print(f"Successfully saved data to {csv_file}")

        except Exception as e:
            print(f"Error extracting table data: {str(e)}")

    except Exception as e:
        print(f"General error occurred: {str(e)}")
    finally:
        driver.quit()
        print("Browser closed")

if __name__ == "__main__":
    scrape_partnership_data()