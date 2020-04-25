#!/usr/bin/env python3
"""
util.py
python file with utilities for reading and working with data from reviews.csv

Ryan A. Mannion
Written for LING472 Final Project
"""
import argparse
import csv


def read_data(csv_file):
    """
    This function reads the data from the csv file and returns a dictionary with column name as keys and
    cell contents as values

    The columns are labeled as follows:
    Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text

    :param csv_file: data file of type csv
    :return header_key: lst, a list of headers in csv_file
    :return review_data: dict, a dictionary with headers as keys and lists of their respective columns as values
    """
    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        i = 0
        for row in csv_reader:
            if i == 0:
                # header row
                header_key = [header for header in row]
                review_data = {col: [] for col in header_key}
                i += 1
            elif i >= 0:
                assert len(row) == len(header_key), "Row is longer than it should be!"
                for j, cell_value in enumerate(row):
                    col_name = header_key[j]
                    review_data[col_name].append(cell_value)
    return header_key, review_data


def data_to_tsv(data_dict, columns):
    with open("review_data.tab", 'w') as f:
        # prints header row
        header_row_values = [column for column in columns]
        new_row = "\t".join(header_row_values) + "\n"
        f.write(new_row)
        # picks sample from data_dict for length, should be the same across values
        sample = list(data_dict.values())[0]
        for i in range(len(sample)):
            row_values = []
            for column in columns:
                values = data_dict[column]
                value = values[i]
                row_values.append(value)
            new_row = "\t".join(row_values) + "\n"
            f.write(new_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../helpfulnessprediction-1/data/reviews.csv",
                        help="csv file containing data to be read")
    args = parser.parse_args()

    headers, data = read_data(args.data_path)

    # select some number of headers for export to tsv
    select_headers = ["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]
    data_to_tsv(data, select_headers)
