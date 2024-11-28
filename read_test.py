
import re
import pandas as pd
from datetime import datetime

# Define file names
date_time_file = r".\Data\Ynet stuff\Ynet_Articles_Data\Full\Date_Time.txt"
main_titles_file = r".\Data\Ynet stuff\Ynet_Articles_Data\Full\Main_Titles.txt"
sub_titles_file = r".\Data\Ynet stuff\Ynet_Articles_Data\Full\Sub_Titles.txt"

# Function to parse the file
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

# Parse each of the files
date_time_df = parse_date_time_file(date_time_file)
main_titles_df = parse_title_file(main_titles_file).rename(columns={'Title': 'Main_Titles'})
sub_titles_df = parse_title_file(sub_titles_file).rename(columns={'Title': 'Sub_Titles'})

# Combine all data into a single DataFrame
combined_df = pd.merge(date_time_df, main_titles_df, on='Sample_Number')
combined_df = pd.merge(combined_df, sub_titles_df, on='Sample_Number')
combined_df = combined_df.sort_values(by='Date_Time', ascending=True).reset_index(drop=True)

# Print the combined DataFrame
print(combined_df)

# Accessing each sample by row index
for idx, sample in combined_df.iterrows():
    print(f"Sample {sample['Sample_Number']}:")
    print(f"Date_Time: {sample['Date_Time']}")
    print(f"Main_Title: {sample['Main_Titles']}")
    print(f"Sub_Title: {sample['Sub_Titles']}")
    print()  # Add spacing between samples