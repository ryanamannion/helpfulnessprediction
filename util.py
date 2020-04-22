#!/usr/bin/env python3
"""
util.py

python file with utilities for reading and working with data from reviews.csv
"""
import argparse
import csv


def read_data(csv_file, header=True):
    """
    This function reads the data from the csv file and returns a dictionary with column name as keys and
    cell contents as values

    The columns are labeled as follows:
    Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text

    :param csv_file: data file of type csv
    :return review_data:
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
    return review_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/reviews.csv",
                        help="csv file containing data to be read")
    args = parser.parse_args()

    data = read_data(args.data_path)
