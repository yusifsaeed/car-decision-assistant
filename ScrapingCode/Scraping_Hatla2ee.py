from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.options import Options
import pandas as pd

# ------------------ OPTIONS ------------------
options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
# options.add_argument("--headless")  # disable for stability first

# ------------------ INPUT ------------------
start = int(input("Enter The Num of first page : "))      
end = int(input("Enter The Num of last page : "))

# ------------------ DRIVERS ------------------
driver = webdriver.Chrome(options=options)
details_driver = webdriver.Chrome(options=options)

data = []

# ------------------ MAIN LOOP ------------------
for i in range(start, end + 1):

    # Open page
    if i == 1:
        driver.get("https://eg.hatla2ee.com/en/car/search?sortby=updated_at_desc")
    else:
        driver.get(f"https://eg.hatla2ee.com/en/car/search?sortby=updated_at_desc&page={i}")

    time.sleep(2)

    cars = driver.find_elements(By.CLASS_NAME , "my-0")

    # ------------------ GET DATE FROM FIRST CAR ------------------
    page_date = "Null"

    try:
        if len(cars) > 0:
            first_car = cars[0]

            link = first_car.find_element(By.TAG_NAME, "a").get_attribute("href")

            details_driver.get(link)
            time.sleep(2)

            # details = details_driver.find_element(By.id, "car-details")
            page_date = details_driver.find_element(By.XPATH, "//div[contains(@class,'font-medium') and contains(text(),'2026')]").text
            
    except:
        page_date = "Null"

    # ------------------ LOOP ALL CARS ------------------
    for car in cars:

        # -------- info --------
        try: 
            info = car.find_element(By.CLASS_NAME , "flex").text.split("\n")
            year = info[0]
            kilo = info[1]
            transmission = info[2]
            fuel_type = info[3]
        except:
            year, kilo, transmission, fuel_type = "Null", "Null", "Null", "Null"

        # -------- price --------
        try:
            price = car.find_element(By.CLASS_NAME , "text-lg").text
        except:
            price = "Null"

        # -------- location, brand, model --------
        try: 
            info2 = car.find_elements(By.CLASS_NAME , "inline-flex")
            location = info2[0].text
            Brand = info2[1].text
            Model = info2[2].text
        except:
            location , Brand , Model = "Null" , "Null" , "Null"

        # -------- link --------
        try:
            link = car.find_element(By.TAG_NAME, "a").get_attribute("href")
        except:
            link = "Null"

        # -------- same date for all --------
        

        # -------- store --------
        data.append({
            "Brand": Brand,
            "Model": Model,
            "Price": price,
            "Year": year,
            "Mileage": kilo,
            "Transmission type": transmission,
            "Fuel type": fuel_type,
            "Location": location,
            "Link": link,
            "Posted On": page_date
        })

    print(f"✅ Page {i} done | Date used: {page_date}")

# ------------------ CLOSE ------------------
driver.quit()
details_driver.quit()

# ------------------ SAVE ------------------
df = pd.DataFrame(data)
df.to_csv(f"data_{start}_{end}.csv", index=False)
