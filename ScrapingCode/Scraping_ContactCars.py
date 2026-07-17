from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
from datetime import datetime, timedelta 
import undetected_chromedriver as uc
import random
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = uc.Chrome(version_main=147)
driver.get("https://www.contactcars.com/en/used-cars")
driver.maximize_window()
WebDriverWait(driver, 8).until(
    EC.presence_of_element_located((By.ID, "vehicles-tabs"))
)
time.sleep(random.uniform(0.5, 1))

cars_list = []
mileage_list = []
Transmission_list = []
price_list = [] 
date_list = [] 
year_list = []
car_links = [] 

for page in range(1):
    driver.get(f"https://www.contactcars.com/en/used-cars?page={page+1}")
    
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "vehicles-tabs"))
    )
    time.sleep(random.uniform(0.5, 1))
    
    cars = driver.find_elements(By.ID, "vehicles-tabs")
    
    for car in cars:
        boxes1 = car.find_elements(By.CLASS_NAME, 'flex-wrap')
        for box in boxes1:
            lines = box.text.split("\n")
            if len(lines) >= 2:
                brand = lines[0]
                model = lines[1]
                if brand == "Premium":
                    continue 
                cars_list.append({"Brand": brand, "Model": model})
                print(f"Car #{len(cars_list)} — {brand} {model}")
        boxes2 = car.find_elements(By.CSS_SELECTOR, 'div.flex.items-center.gap-1.mt-2')
        for box2 in boxes2:
            try:
                Millage = box2.find_element(By.CSS_SELECTOR, 'div:nth-child(3) > span').text
                mileage_list.append(Millage)
            except:
                mileage_list.append("N/A")

            try:
                Transmission = box2.find_element(By.XPATH, 'div[2]/span').text
                Transmission_list.append(Transmission)
            except:
                Transmission_list.append("N/A")
                
        boxes3 = car.find_elements(By.CSS_SELECTOR, 'div.flex.items-center.justify-between.h-7')
        for box3 in boxes3:
            price = box3.find_element(By.CSS_SELECTOR, 'div.flex.items-center.justify-between.h-7 > span').text
            price_list.append(price)

        boxes4 = car.find_elements(By.CLASS_NAME, 'rounded-lg')
        for box4 in boxes4:
            try:
                relative_time = box4.find_element(By.CSS_SELECTOR, 'div.flex.items-center.gap-1 > span').text
                current_date = datetime.now()
                if "hour" in relative_time:
                    value = int(relative_time.split()[0])
                    posted_date = current_date - timedelta(hours=value)
                elif "day" in relative_time:
                    value = int(relative_time.split()[0])
                    posted_date = current_date - timedelta(days=value)
                else:
                    posted_date = current_date
                date_list.append(posted_date.strftime("%Y/%m/%d"))
            except:
                date_list.append("N/A")

        boxes5 = car.find_elements(By.CSS_SELECTOR, 'a.px-3.pt-2.bg-white-900.flex.flex-col')
        for box5 in boxes5:       
            titles = box5.find_elements(By.CLASS_NAME, 'sub-h-lg ')
            for title in titles:
                try:
                    year = title.text.split()[-1]
                    if len(year) == 4:
                        year_list.append(year)
                    else:
                        year_list.append("N/A")
                except:
                    year_list.append("N/A")

        anchors = car.find_elements(By.CSS_SELECTOR, 'a.px-3.pt-2.bg-white-900.flex.flex-col')
        for anchor in anchors:
            href = anchor.get_attribute('href')
            if href:
                car_links.append(href)

cc_list = []
for url in car_links:
    driver.get(url)
    time.sleep(random.uniform(0.5, 1))
    try:
        cc_element = driver.find_element(By.CSS_SELECTOR,"div:nth-child(6) > div.flex.flex-col > h5 > span.whitespace-nowrap")
        cc_list.append(cc_element.text)
    except:
        cc_list.append("N/A")
    driver.back()
    time.sleep(random.uniform(0.5, 1))

df = pd.DataFrame(cars_list)

min_len = min(len(df), len(mileage_list), len(Transmission_list), 
              len(date_list), len(price_list), len(year_list), len(cc_list))

df = df.iloc[:min_len]

df['Mileage']       = mileage_list[:min_len]
df['Transmission']  = Transmission_list[:min_len]
df['Price']         = price_list[:min_len]
df['Year']          = year_list[:min_len]
df['Engine CC']     = cc_list[:min_len]
df['Date Posted']   = date_list[:min_len]
df.to_csv('ContactCars.csv', index=False)