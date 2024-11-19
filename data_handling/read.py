
import pandas as pd


def read_preprocess_articles(file_path):
        x=5
        return pd.DataFrame()


def read_preprocess_threats(file_path, locations_to_keep, types_to_keep):
        # Read the CSV file with UTF-8 encoding, which supports Hebrew
        df = pd.read_csv(file_path, encoding='utf-8')

        # Flip the dataset so that the last row becomes the first
        df = df.iloc[::-1].reset_index(drop=True)

        # Filter rows
        df = df[df['Location'].isin(locations_to_keep)]
        df = df[df['Type'].isin(types_to_keep)]

        # Modify the Time column to keep only the hour and convert it to a number
        df['Time'] = df['Time'].str.split(':').str[0].astype(int)

        # Transform Locations into numbers
        unique_locations = df['Location'].unique()
        location_mapping = {location: idx + 1 for idx, location in enumerate(unique_locations)}
        df['Location'] = df['Location'].map(location_mapping)

        # Transform Type column into numbers
        unique_types = df['Type'].unique()
        type_mapping = {type_: idx + 1 for idx, type_ in enumerate(unique_types)}
        df['Type'] = df['Type'].map(type_mapping)

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
        df['Day'] = df['Day'].map(day_mapping)

        # Split the Date column into day, month, and year components and convert them to integers
        df['Day_of_Month'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.day.astype(int)
        df['Month'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.month.astype(int)
        df['Year'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.year.astype(int)
        df = df.drop(columns=['Date'])

        # Reset index to start from 0 again
        df.reset_index(drop=True, inplace=True)

        return df, location_mapping, type_mapping

def read_from_csv():
        # Articles
        articles_csv_path = r""
        articles_df = read_preprocess_articles(articles_csv_path)

        # Threats
        threats_csv_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\rockets-threat-prediction\Data\Data_07_10_23_11_11_24\combined_data\combined_output_no_dups_Time_rounded_v4.csv"
        locations_to_keep = ["קריית שמונה", "חיפה - נווה שאנן ורמות כרמל", "צפת - עיר",
                             "תל אביב - מרכז העיר"]
        types_to_keep = ["ירי רקטות וטילים"]
        threats_df, location_mapping, type_mapping = read_preprocess_threats(threats_csv_path, locations_to_keep, types_to_keep)

        return articles_df, threats_df

# if __name__ == "__main__":
#         # Articles
#         articles_csv_path = r""
#         articles_df = read_preprocess_articles(articles_csv_path)
#
#         # Threats
#         threats_csv_path = r"C:\Users\alonz\OneDrive - Technion\Documents\GitHub\rockets-threat-prediction\Data\Data_07_10_23_11_11_24\combined_data\combined_output_no_dups_Time_rounded_v4.csv"
#         locations_to_keep = ["קריית שמונה", "חיפה - נווה שאנן ורמות כרמל", "צפת - עיר",
#                              "תל אביב - מרכז העיר"]
#         types_to_keep = ["ירי רקטות וטילים"]
#         threats_df, location_mapping, type_mapping = read_preprocess_threats(threats_csv_path,
#                                                                                      locations_to_keep, types_to_keep)










