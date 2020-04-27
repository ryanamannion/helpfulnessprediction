#!/usr/bin/env python3
"""
data_util.py

Contains functions for reading and saving data, as well as the class ReviewerData which handles loading and splitting
of data, and has attributes to help organization

Functions:
    read_data
    data_to_tsv

Class:
    ReviewerData

Ryan A. Mannion
Written for LING472 Final Project
2020
"""
import argparse
import csv
import random
from collections import defaultdict


def read_data(csv_file, delimiter=str):
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


def data_to_tsv(data_dict, output_name=str, all_columns=True, columns=None):
    """
    Outputs data from dictionary output of read_data to tsv, allows for selection of only certain columns
    :param data_dict: dict, dictionary containing the data
    :param all_columns: bool, if true, all columns from data_dict are output to tsv
    :param columns: if all_columns is False, list of column names to be output to tsv, must match headers in data_dict
    :param output_name: name of file to be output, passeed without file extension
    """
    output_name = f"{output_name}.tsv"
    with open(output_name, 'w') as f:
        if all_columns:
            header_row_values = list(data_dict.keys())
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
    """
    Class to handle loading and splitting of reviewer data

    Args:
        data_file (str): file name of the data to be loaded
        delimiter (str): delimiter type used in data_file_name, e.g. ',' or '\t'

    Attributes:
        data_file_name (str): value of arg data_file
        delimiter_type (str): value of arg delimiter
        data_dict (dict): dictionary containing data extracted from data_file_name, keys are header row
        headers (list): list containing header for of data_file_name
        train (dict): dictionary to contain train data after method split_data is run
        dev_test (dict): dictionary to contain dev_test data after method split_data is run
        test (dict): dictionary to contain test data after method split_data is run
    """
    def __init__(self, data_file, delimiter):
        self.data_file_name = data_file
        self.delimiter_type = delimiter
        self.data_dict, self.headers = read_data(data_file, delimiter=delimiter)

        # if split_data() method used
        self.train = None
        self.dev_test = None
        self.test = None

    def split_data(self, test=10, dev_test=True, shuffle=True):
        """
        Shuffles and splits data in unison into train and test sets for experimental use
        :param test: (int) 1-100 percentage of data to be withheld for testing
        :param dev_test: (bool) create a dev test set of the same size as test or not
        :param shuffle: (bool) shuffle the data before splitting or not
        """
        # Shuffles data if specified
        length_of_data = len(list(self.data_dict.values())[0])
        order = list(range(length_of_data))
        if shuffle:
            random.shuffle(order)

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
            self.dev_test = defaultdict(list)
        self.test = defaultdict(list)
        self.train = defaultdict(list)

        for header in self.headers:
            column_data = self.data_dict[header]
            self.test[header] = [column_data[i] for i in test_indices]
            self.train[header] = [column_data[i] for i in dev_indices]       # will be its new value if dev_test is True
            if dev_test:
                self.dev_test[header] = [column_data[i] for i in dev_test_indices]


def main():
    """
    Loads csv file, creates class instance and saves splits to tsv files for later use
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="reviews.csv", help="csv file containing data to be read")
    args = parser.parse_args()

    print("Loading data...")
    review_data = ReviewerData(data_file=args.data_path, delimiter=',')

    print("Splitting data...")
    review_data.split_data()

    print("Saving train...")
    data_to_tsv(review_data.train, output_name="./data/train")
    print("Saving dev_test...")
    data_to_tsv(review_data.dev_test, output_name="./data/dev_test")
    print("Saving test...")
    data_to_tsv(review_data.test, output_name="./data/test")

    print("Complete!")


if __name__ == "__main__":
    main()
