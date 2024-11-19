
POScrape.py Readme


driver():
Opens webpage and allows you to pick date range. After some
time, saves the html of the entire page
and allows you to pick another date range.


parse_alerts_from_file(html_filename):
parses html page and removes from it only the interesting data.
Knows to organize the data into 5 columns:
Date, Day, Type, Time, Location (in this order)
If there are multiple locations in one sample knows to deal with
that.
Saves the parsed table into a csv file with the html_filename as its
name.


parse_all():
Just runs parse_alerts_from_file on all the html files in folder data


sort_files_by_date(files):
receives a list, the list's elements are names of files. each name
is a range of dates like 02.07.2024-07.07.2024, and this function
just sorts the list. Returns a list with the same elements, but sorted
such that the files are organized by their names' descending order.


combine_csv_files(folder_path, output_filename):
concatenates all the csv files (parsed tables) previously created.
Calls sort_files_by_date inside of it to first sort the csv files
and then concatenate them. Assumes that the csv files are properly
named (like above) and that all the csv files have tables with 5
columns (like above).
Saves the new csv file into output_file.csv


round_time_to_nearest_hour(file_name, output_file):
Receives a csv file name and rounds all the times in the column 
'Time' down. Assumes the csv file has a column 'Time'
Saves the new csv file into output_file.csv


remove_duplicate_rows(input_file, output_file):
Receives csv file name and removes all the duplicate rows.
Saves the new csv file into output_file.csv

