#!/usr/bin/env python3
"""
util.py
python file with utilities for reading and working with data from reviews.csv

Ryan A. Mannion
Written for LING472 Final Project
"""
import argparse
import csv
import numpy as np


def read_data(csv_file):
    """
    Reads data from the csv file and returns a dictionary with column name as keys and cell contents as values

    The columns are labeled as follows:
    Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text

    :param csv_file: data file of type csv
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
    return review_data


def data_to_tsv(data_dict, all_columns=True, columns=None, output_name="review_data.tab"):
    """
    Selects columns from the data and output them to a tsv file for use in R and other functions
    :param data_dict: dict, dictionary containing the data
    :param all_columns: bool, if true, all columns from data_dict are output to tsv
    :param columns: if all is False, list of column names to be output to tsv, must match headers in data_dict
    :param output_name: name of file to be output to cwd, defaults to review_data.tab
    :return: outputs file called value of output_name to cwd
    """
    with open(output_name, 'w') as f:
        if all_columns:
            header_row_values = data_dict.keys()
        else:
            header_row_values = [column for column in columns]
        # prints header row to file
        header_row = "\t".join(header_row_values) + "\n"
        f.write(header_row)
        # picks sample from data_dict for length, should be the same across values
        sample = list(data_dict.values())[0]
        for i in range(len(sample)):
            row_values = []
            for column_name in header_row_values:
                values = data_dict[column_name]
                value = values[i]
                row_values.append(value)
            new_row = "\t".join(row_values) + "\n"
            f.write(new_row)


def split_data(data, test=10, dev_test=True):
    """
    Shuffles and splits data into train and test sets for experimental use
    :param data: data to be split TODO: variable type?
    :param test: int, percentage of data to be withheld for testing, defaults to 10%
    :param dev_test: bool, whether or not to create a dev test set of the same size as test
    :return: TODO
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../helpfulnessprediction-1/data/reviews.csv",
                        help="csv file containing data to be read")
    args = parser.parse_args()

    my_data = read_data(args.data_path)

    # select some number of headers for export to tsv
    select_headers = ["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]
    data_to_tsv(my_data, select_headers)
