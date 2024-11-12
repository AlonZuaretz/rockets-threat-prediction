import glob
from datetime import datetime, timedelta
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import bs4
import time

TIME_DELTA = 5

def driver():
    end_date_obj = datetime.today()
    start_date_obj = end_date_obj - timedelta(days=TIME_DELTA)

    # Changing end and start date to preferred date, to do this change the number of iterations
    for i in range(20):
        end_date_obj = start_date_obj - timedelta(days=1)
        start_date_obj = end_date_obj - timedelta(days=TIME_DELTA)

    driver = webdriver.Chrome()  # or webdriver.Firefox()
    driver.get("https://www.oref.org.il/heb/alerts-history")
    iframe = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "IframeId"))
    )
    driver.switch_to.frame(iframe)

    for i in range(30):
        end_date_obj = start_date_obj - timedelta(days=1)
        start_date_obj = end_date_obj - timedelta(days=TIME_DELTA)

        start_date = start_date_obj.strftime("%d.%m.%Y")
        end_date = end_date_obj.strftime("%d.%m.%Y")

        try:
            # Wait for the calendar button to appear and then click it
            calendar_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//img[@src='/images/SVG/General/calendar.svg']"))
            )
            # Use JavaScript to click if a normal click doesn’t work
            driver.execute_script("arguments[0].click();", calendar_button)
            # Inject dates directly into the date fields using JavaScript
            driver.execute_script(f"document.getElementById('txtDateFrom').value = '{start_date}';")
            print(f"Please input end date ({end_date})")

            TIME_LIMIT = 15
            for i in range(TIME_LIMIT):
                print(f"{TIME_LIMIT-i} seconds left")
                time.sleep(1)

            # WebDriverWait(driver, 20).until(
            #     EC.presence_of_all_elements_located((By.XPATH, "//span[text()='הצג עוד התרעות']"))
            # )

            # Find all span elements with the text "הצג עוד התרעות" and click each one
            span_elements = driver.find_elements(By.XPATH, "//span[text()='הצג עוד התרעות']")

            if len(span_elements) > 0:
                for span in span_elements:
                    driver.execute_script("arguments[0].click();", span)


            # Get the page source
            page_html = driver.page_source

            # Save to a file
            with open(f"{start_date}-{end_date}.html", "w", encoding="utf-8") as file:
                file.write(page_html)


        except Exception as e:
            print(e)

    driver.quit()


def parser(file_name):
    data = ""
    with open(file_name, "r", encoding="utf-8") as p:
        data = p.read()

    # Parse the HTML
    soup = bs4.BeautifulSoup(data, 'html.parser')

    # Initialize lists to collect data
    dates, categories, times, locations = [], [], [], []

    # Track the current date as we iterate
    current_date = None

    # Loop through elements to capture dates and alerts
    for element in soup.find_all(['h3', 'div'], {'class': ['alertTableDate', 'alert_table']}):
        # Check if the element is a date
        if 'alertTableDate' in element.get('class', []):
            current_date = element.get_text(strip=True)

        # Otherwise, it should be an alert table
        elif 'alert_table' in element.get('class', []):
            # Extract the category
            category = element.find('h4', class_='alertTableCategory').get_text(strip=True)

            # Loop through each alert detail
            for alert_detail in element.find_all('div', class_='alertDetails'):
                time = alert_detail.find('h5', class_='alertTableTime').get_text(strip=True)
                location = alert_detail.get_text(strip=True).replace(time, '').strip()

                # Append data to lists
                dates.append(current_date)
                categories.append(category)
                times.append(time)
                locations.append(location)

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Time': times,
        'Category': categories,
        'Location': locations
    })

    # Display DataFrame
    df.to_csv(f"{file_name}.csv")

def parse_all():
    files = glob.glob("*.html")
    for file in files:
        parser(file)

if __name__ == '__main__':
    driver()