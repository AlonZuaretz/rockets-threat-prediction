
import pandas as pd
import re
import os

from datetime import datetime


def read_from_csv(alerts_path, articles_path, time_resolution):
    # Articles
    articles_df = read_preprocess_articles(articles_path)

    # Threats
    threats_csv_path = alerts_path
    types_to_keep = ["ירי רקטות וטילים"]
    threats_df, location_mapping = read_preprocess_threats(threats_csv_path,
                                                                     types_to_keep, time_resolution)

    return articles_df, threats_df, location_mapping


def read_preprocess_articles(path):
    date_time_file = os.path.join(path, "Date_Time.txt")
    main_titles_file = os.path.join(path, "Main_Titles.txt")
    sub_titles_file = os.path.join(path, "Sub_Titles.txt")

    # Parse each of the files
    date_time_df = parse_date_time_file(date_time_file)
    main_titles_df = parse_title_file(main_titles_file).rename(columns={'Title': 'Main_Titles'})
    sub_titles_df = parse_title_file(sub_titles_file).rename(columns={'Title': 'Sub_Titles'})

    # Process:
    # Create new columns for timestamp, day of the week, hour, day, month, year
    date_time_df['timestamp'] = date_time_df['Date_Time'].apply(lambda x: int(x.timestamp()))  # Timestamp in seconds
    date_time_df['week day'] = date_time_df['Date_Time'].dt.day_name()  # Day of the week (e.g., Sunday)
    date_time_df['hour'] = date_time_df['Date_Time'].dt.hour  # Hour of the day
    date_time_df['day'] = date_time_df['Date_Time'].dt.day  # Day of the month
    date_time_df['month'] = date_time_df['Date_Time'].dt.month  # Month
    date_time_df['year'] = date_time_df['Date_Time'].dt.year  # Year
    date_time_df = date_time_df.drop(columns=['Date_Time'])

    day_mapping = {
        'Sunday': 1,  # Sunday
        'Monday': 2,  # Monday
        'Tuesday': 3,  # Tuesday
        'Wednesday': 4,  # Wednesday
        'Thursday': 5,  # Thursday
        'Friday': 6,  # Friday
        'Saturday': 7  # Saturday
    }
    date_time_df['week day'] = date_time_df['week day'].map(day_mapping)

    # Combine all data into a single DataFrame
    combined_df = pd.merge(date_time_df, main_titles_df, on='Sample_Number')
    combined_df = pd.merge(combined_df, sub_titles_df, on='Sample_Number')
    combined_df = combined_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    combined_df = combined_df.drop_duplicates(keep=False)

    return combined_df

def group_time(hour_str, resolution):
    hour = int(hour_str.split(':')[0])  # Extract the hour
    return (hour // resolution) * resolution


def read_preprocess_threats(file_path, types_to_keep, time_resolution):
        # Read the CSV file with UTF-8 encoding, which supports Hebrew
        df = pd.read_csv(file_path, encoding='utf-8')

        # Flip the dataset so that the last row becomes the first
        df = df.iloc[::-1].reset_index(drop=True)

        # Keep only rows with predefined types:
        df = df[df['type'].isin(types_to_keep)]
        df = df.drop(columns=['type'])

        # Round with the selected time resolution
        df['hour'] = df['hour'].apply(lambda x: group_time(x, time_resolution))

        # Transform Locations into numbers
        unique_locations = df['location'].unique()
        location_mapping = {location: idx + 1 for idx, location in enumerate(unique_locations)}
        df['location'] = df['location'].map(location_mapping)

        # Transform Day column into numbers representing days of the week
        day_mapping = {
            'ראשון': 1,  # Sunday
            'שני': 2,    # Monday
            'שלישי': 3,  # Tuesday
            'רביעי': 4,  # Wednesday
            'חמישי': 5,  # Thursday
            'שישי': 6,   # Friday
            'שבת': 7     # Saturday
        }
        df['week day'] = df['week day'].map(day_mapping)

        # Split the Date column into day, month, and year components and convert them to integers
        df['day'] = pd.to_datetime(df['date'], format='%d.%m.%Y').dt.day.astype(int)
        df['month'] = pd.to_datetime(df['date'], format='%d.%m.%Y').dt.month.astype(int)
        df['year'] = pd.to_datetime(df['date'], format='%d.%m.%Y').dt.year.astype(int)
        df = df.drop(columns=['date'])

        # Reset index to start from 0 again
        df.reset_index(drop=True, inplace=True)

        return df, location_mapping


def parse_date_time_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize an empty list to store parsed data
    data = []

    # Iterate over lines and parse information
    sample_number = None
    for line in lines:
        line = line.strip()  # Remove trailing newline and spaces
        if line.startswith("Sample number:"):
            sample_number = int(line.split(":")[1].strip())
        elif re.match(r'\d{2}\.\d{2}\.\d{2} \| \d{2}:\d{2}', line):
            # Convert the date-time string to a datetime object
            date_time_str = line.replace('|', '').strip()
            date_time = datetime.strptime(date_time_str, "%d.%m.%y %H:%M")
            data.append({'Sample_Number': sample_number, 'Date_Time': date_time})

    return pd.DataFrame(data)

# Function to parse the Main_Titles and Sub_Titles files
def parse_title_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    # Initialize an empty list to store parsed data
    data = []

    # Iterate over lines and parse information
    sample_number = None
    for line in lines:
        line = line.strip()  # Remove trailing newline and spaces
        if line.startswith("Sample number:"):
            sample_number = int(line.split(":")[1].strip())
        elif line and not line.startswith("Page Number:"):
            # Assuming that the non-empty line after "Sample number" is the title
            title = line
            data.append({'Sample_Number': sample_number, 'Title': title})

    return pd.DataFrame(data)


# if __name__ == "__main__":
# #         # Articles
# #         articles_csv_path = r""
# #         articles_df = read_preprocess_articles(articles_csv_path)
# #
# #         # Threats
# #         threats_csv_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\rockets-threat-prediction\Data\Data_07_10_23_11_11_24\combined_data\alerts_dataset.csv"
# #         locations_to_keep = ["קריית שמונה", "חיפה - נווה שאנן ורמות כרמל", "צפת - עיר",
# #                              "תל אביב - מרכז העיר"]
# #         types_to_keep = ["ירי רקטות וטילים"]
# #         threats_df, location_mapping, type_mapping = read_preprocess_threats(threats_csv_path,
# #                                                                                      locations_to_keep, types_to_keep)













