import glob
from datetime import datetime, timedelta
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import os

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



def parse_alerts_from_file(html_filename):
    # Open and read the HTML file
    with open(html_filename, 'r', encoding='utf-8') as file:
        html_code = file.read()

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_code, 'html.parser')

    # Initialize an empty list to store the parsed data
    parsed_data = []

    # Initialize the variables for date and day (they will be updated during iteration)
    date_section = None
    day_section = None
    type_section = None

    # Iterate over all the sections where alerts are located
    for element in soup.find_all(['h3', 'h4', 'h5', 'div']):
        # Detect a new date and day section
        if element.name == 'h3' and 'alertTableDate' in element.get('class', []):
            date_day_text = element.text.strip()  # Example: "יום רביעי 10.07.2024"

            # Correctly split the text into day and date
            # Step 1: Remove "יום" from the beginning
            day_date_parts = date_day_text.replace('יום', '').strip().split(' ', 1)
            if len(day_date_parts) == 2:
                # Step 2: day is the first part, date is the second part
                day_section, date_section = day_date_parts
            else:
                # Handle unexpected format if there's only a single part (shouldn't happen based on the provided HTML)
                day_section = day_date_parts[0]
                date_section = ''

        # Detect a new type section
        if element.name == 'h4' and 'alertTableCategory' in element.get('class', []):
            type_section = element.text.strip()

        # Detect alert details (time and location)
        if element.name == 'div' and 'alertDetails' in element.get('class', []):
            time_tag = element.find('h5', class_='alertTableTime')
            if time_tag:
                time_text = time_tag.text.strip()  # Example: "22:09"
                location_text = element.text.replace(time_text, '').strip()  # Example: "משגב עם"

                # If location has multiple entries (comma-separated), split into separate rows
                locations = [loc.strip() for loc in location_text.split(',')]

                for location in locations:
                    # Append a new row to the results
                    parsed_data.append({
                        'Date': date_section,
                        'Day': day_section,
                        'Type': type_section,
                        'Time': time_text,
                        'Location': location
                    })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(parsed_data)

    # Generate the CSV file name based on the HTML file name
    csv_filename = os.path.splitext(html_filename)[0] + '.csv'

    # Save the DataFrame to CSV
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    print(f"CSV file has been saved as '{csv_filename}'.")
    return df  # Optional: return the DataFrame for further inspection or processing

def parse_all():
    files = glob.glob("./data/*.html")
    for file in files:
        parse_alerts_from_file(file)

def sort_files_by_date(files):
    # Function to extract the start date from the file name
    def get_start_date(file):
        # Extracts the start date from file name assuming the format "dd.mm.yyyy-dd.mm.yyyy"
        date_range = os.path.basename(file).split('-')[0]  # Extract the start date part
        return datetime.strptime(date_range, "%d.%m.%Y")  # Convert to datetime object

    # Sort the files by their start date
    return sorted(files, key=get_start_date, reverse=True)


def combine_csv_files(folder_path, output_filename):
    # Get all CSV files from the folder
    files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Sort the files by date range (based on the start date)
    sorted_files = sort_files_by_date(files)

    # Initialize an empty list to store DataFrames
    df_list = []

    # Iterate over each sorted CSV file
    for file in sorted_files:
        # Read the CSV file into a DataFrame (skip the first row)
        df = pd.read_csv(file, encoding='utf-8-sig')

        # Debugging: Check the shape and columns of each DataFrame before appending
        print(f"Processing file: {file}")
        print(f"Shape of {file}: {df.shape}")
        print(f"Columns in {file}: {df.columns.tolist()}")

        # Force column order if necessary (assuming the expected column names are known)
        expected_columns = ['Date', 'Day', 'Type', 'Time', 'Location']
        df = df[expected_columns]  # This will reorder columns if necessary

        # Check for empty DataFrames and skip them
        if df.empty:
            print(f"Warning: {file} is empty and will be skipped.")
            continue

        # Append the DataFrame to the list
        df_list.append(df)

    # Check if df_list is empty after processing
    if not df_list:
        print("No valid CSV files to combine.")
        return

    # Concatenate all DataFrames in the list into one large DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Debugging: Check the shape of the final combined DataFrame
    print(f"Combined DataFrame shape: {combined_df.shape}")

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"Combined CSV has been saved as '{output_filename}'.")


def round_time_to_nearest_hour(file_name, output_file):
    """
    Round down all times in the given time column to the nearest hour.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    time_column (str): The column name that contains the time values.

    Returns:
    pandas.DataFrame: A new dataframe with the time column rounded to the nearest hour.
    """
    df = pd.read_csv(file_name)

    time_column = 'Time'
    # Convert the 'Time' column to datetime objects (if not already)
    df[time_column] = pd.to_datetime(df[time_column], format='%H:%M', errors='coerce')

    # Round down the time to the nearest hour
    df[time_column] = df[time_column].dt.floor('H').dt.strftime('%H:%M')

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"File saved as {output_file}")

def remove_duplicate_rows(input_file, output_file):
    """
    Removes duplicate rows (rows that are exactly identical across all columns) from a CSV file.

    Parameters:
    input_file (str): Path to the input CSV file.
    output_file (str): Path where the cleaned CSV file will be saved.
    """
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_file)

    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Save the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"Cleaned file saved as '{output_file}'.")

if __name__ == '__main__':
    # parse_all()
    driver()
    # folder_path = "./data"  # Replace with your folder path
    # output_filename = "combined_alerts.csv"  # Output filename for the combined CSV
    #
    # # Example usage
    # combine_csv_files("./data", "combined_output.csv")
    # round_time_to_nearest_hour("combined_output_cleaned.csv")
    # remove_duplicate_rows("combined_output_Time_rounded.csv")

