#!/usr/bin/env python3
"""
util.py
python file with utilities for reading and working with data from reviews.csv

Ryan A. Mannion
Written for LING472 Final Project

Envisioned workflow:
    - Kaggle csv containing loaded into data_dict via read_data()
    - data_dict split into dev and test sets with split_data(), saved as tsv
    - TODO: write function to load data from tsv to data_dict
"""
import argparse
import csv
import numpy as np
import random
from collections import defaultdict


def read_data(csv_file, delimiter):
    """
    Reads data from csv/tsv file and returns a dictionary with column name as keys and cell contents as values
    :param csv_file: data file of type csv
    :param delimiter: delimiter used in file, aka ',' for csv or '\t' for tsv
    :return review_data: dict, a dictionary with headers as keys and lists of their respective columns as values
    :return header_key: lst, a list of headers as they appear in review_data as keys
    """
    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        i = 0
        for row in csv_reader:
            if i == 0:
                # header row
                header_key = [header for header in row]
                review_data = {col: [] for col in header_key}
                i += 1
            elif i >= 0:
                # ensures delimiter did not bug out
                assert len(row) == len(header_key), "Row is longer than it should be!"
                for j, cell_value in enumerate(row):
                    col_name = header_key[j]
                    review_data[col_name].append(cell_value)
    return review_data, header_key


def data_to_tsv(data_dict, all_columns=True, columns=None, output_name="review_data.tab"):
    """
    Outputs data from dictionary output of read_data to tsv, allows for selection of only certain columns
    :param data_dict: dict, dictionary containing the data
    :param all_columns: bool, if true, all columns from data_dict are output to tsv
    :param columns: if all_columns is False, list of column names to be output to tsv, must match headers in data_dict
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


class ReviewerData:
    """ TODO """

    def __init__(self, data_file, delimiter):
        self.data_file_name = data_file
        self.delimiter_type = delimiter
        self.data_dict, self.headers = read_data(data_file, delimiter=delimiter)

    def split_data(self, test=10, dev_test=True, shuffle=True):
        """
        Shuffles and splits data in unison into train and test sets for experimental use
        :param test: int, 1-100 percentage of data to be withheld for testing
        :param dev_test: bool, whether or not to create a dev test set of the same size as test, if True split_data will
        return a third variable dev_test_split
        :param shuffle: bool, whether or not to shuffle the data before splitting
        :return dev_split: development split of data
        :return test_split: test_split of data
        :return dev_test_split: ONLY returns if dev_test=True
        """
        # Shuffles data if specified
        length_of_data = len(self.data_dict.values()[0])
        if shuffle:
            order = random.shuffle[list(range(length_of_data))]
        else:
            order = list(range(length_of_data))

        # determines split point and splits order indices
        test_percent = test / 100
        test_split_point = int(length_of_data * test_percent)
        test_indices = order[:test_split_point]
        dev_indices = order[test_split_point:]

        # further splits dev set into train and dev-test sets if dev_test is True
        if dev_test:
            dev_split_point = int(len(dev_indices) * test_percent)
            dev_test_indices = dev_indices[:dev_split_point]
            dev_indices = dev_indices[dev_split_point:]     # reassigns variable
            dev_test_split = defaultdict(list)
        test_split = defaultdict(list)
        dev_split = defaultdict(list)

        for header in self.headers:
            column_data = self.data_dict[header]
            test_split[header] = [column_data[i] for i in test_indices]
            dev_split[header] = [column_data[i] for i in dev_indices]       # will be its new value if dev_test is True
            if dev_test:
                dev_test_split[header] = [column_data[i] for i in dev_test_indices]

        if dev_test:
            return dev_split, dev_test_split, test_split
        else:
            return dev_split, test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../helpfulnessprediction-1/data/reviews.csv",
                        help="csv file containing data to be read")
    args = parser.parse_args()

    my_data = read_data(args.data_path)

    # select some number of headers for export to tsv
    select_headers = ["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]
    data_to_tsv(my_data, select_headers)
