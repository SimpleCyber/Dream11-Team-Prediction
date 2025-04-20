from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import re
import time

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    service = Service('chromedriver.exe')  # Update path
    return webdriver.Chrome(service=service, options=chrome_options)

def convert_ordinal_to_int(ordinal_str):
    """Convert '1st', '2nd', etc. to integer"""
    if not ordinal_str:
        return None
    return int(re.sub(r'\D', '', ordinal_str))

def scrape_all_matches():
    driver = setup_driver()
    try:
        url = "https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/matches"
        driver.get(url)
        
        # Wait for matches to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="series-matches"]'))
        )
        time.sleep(3)

        matches = []
        match_elements = driver.find_elements(By.XPATH, '//*[@id="series-matches"]/div[position()>2]')
        
        for idx, element in enumerate(match_elements, start=1):
            try:
                # Extract date
                date = element.find_element(By.XPATH, './/span[1]').text.strip()
                
                # Extract match details - more robust selector
                match_text = element.find_element(By.XPATH, './/a[contains(@href, "cricket-scores")]//span').text.strip()
                
                # Extract venue - more flexible selector
                venue = element.find_element(By.XPATH, './/div[contains(@class, "cb-font-12") or contains(@class, "text-gray")]').text.strip()
                
                # Extract time - more flexible selector
                time_element = element.find_element(By.XPATH, './/div[contains(@class, "text-gray") or contains(@class, "schedule-time")]')
                match_time = time_element.text.split('\n')[0].strip()
                
                # Parse teams and match number with better error handling
                if ' vs ' in match_text:
                    team1, rest = match_text.split(' vs ', 1)
                    team2 = rest.split(',')[0].strip()
                    
                    # Handle ordinal numbers (1st, 2nd, etc.)
                    match_num_str = rest.split(',')[-1].replace('Match', '').strip()
                    match_num = convert_ordinal_to_int(match_num_str) or idx
                else:
                    team1, team2, match_num = "TBD", "TBD", idx
                
                matches.append({
                    'Match Number': match_num,
                    'Date': date,
                    'Team 1': team1,
                    'Team 2': team2,
                    'Venue': venue,
                    'Time': match_time
                })
                
            except Exception as e:
                print(f"Skipping match {idx} due to error: {str(e)}")
                continue
                
        return pd.DataFrame(matches)
    
    finally:
        driver.quit()

if __name__ == "__main__":
    print("Scraping all IPL 2025 matches...")
    df = scrape_all_matches()
    
    if not df.empty:
        # Sort by match number and reset index
        df = df.sort_values('Match Number').reset_index(drop=True)
        
        df.to_csv('ipl_2025_full_schedule.csv', index=False)
        print(f"\nSuccessfully scraped {len(df)} matches!")
        print(df[['Match Number', 'Date', 'Team 1', 'Team 2']].head(76))
        print("\nFull schedule saved to 'ipl_2025_full_schedule.csv'")
    else:
        print("Failed to scrape matches. Website structure may have changed.")