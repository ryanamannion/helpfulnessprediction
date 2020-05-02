#!/usr/bin/env python3
"""
ablation.py
includes functions and a class for running a feature ablation

Ryan A. Mannion
Written for LING472 Final Project
2020
"""
import argparse
import numpy as np
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from data_util import ReviewerData, filter_data
import datetime


def scale_data(data):
    """
    Scales data as preprocessing step for LogisticRegression() instance
    :param data: ndarray to be scaled
    :return scaled_data: the scaled data to be used in LogisticRegression() instance as X
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def create_ablation_sets(feature_array, data,  minimum_votes, help_boundary, scale=True):
    """
    Takes a full feature array with matching data_dict of ReviewerData instance as input and returns 5 sets of features:

    Ablation Feature Sets:
    X1 - Review Simple
    X2 - Review + Readability
    X3 - Review + Meta
    X4 - Review + Summary
    X5 - Review + Readability + Meta
    X6 - Review + Readability + Summary
    X7 - Review + Summary + Meta
    X8 - All

    :param feature_array: ndarray containing full set of features to be turned into ablation set
    :param data: ReviewerData instance's data_dict attribute, matches feature_array
    :param scale: (bool) whether or not to scale data before Logistic Regression
    :param minimum_votes: (float) minimum number of helpfulness votes to be included in ablation sets
    :param help_boundary: (float) percentage (0-1) of votes needed to be considered helpful
    :return ablation_sets: (lst) ndarrays for ablation
    :return y: gold standard tags for helpfulness, binary based on help_boundary
    """

    if isinstance(feature_array, np.ndarray):
        all_features = feature_array
    else:
        all_features = np.load(feature_array) if feature_array.endswith(".npy") else np.load(f"{feature_array}.npy")

    if isinstance(data, ReviewerData):
        data = data.data_dict
    else:
        data = ReviewerData(data_file=data, delimiter='\t').data_dict

    processed_feats, y = filter_data(data_dict=data, feature_array=all_features,
                                     minimum_votes=minimum_votes, help_boundary=help_boundary)

    summary_feats = processed_feats[:, 0:5]
    review_simple_feats = processed_feats[:, 6:11]
    readability_feats = processed_feats[:, 11:]

    # Creates feature sets and scales if scale=True
    set1 = scale_data(review_simple_feats) if scale else review_simple_feats
    # -
    set2_concat = np.concatenate((review_simple_feats, readability_feats), axis=1)
    set2 = scale_data(set2_concat) if scale else set2_concat
    # -
    set3 = scale_data(processed_feats[:, 5:11]) if scale else processed_feats[:, 5:11]
    # -
    set4_concat = np.concatenate((review_simple_feats, summary_feats), axis=1)
    set4 = scale_data(set4_concat) if scale else set4_concat
    # -
    set5 = scale_data(processed_feats[:, 5:]) if scale else processed_feats[:, 5:]
    # -
    set6_concat = np.concatenate((summary_feats, review_simple_feats, readability_feats), axis=1)
    set6 = scale_data(set6_concat) if scale else set6_concat
    # -
    set7 = scale_data(processed_feats[:, 0:11]) if scale else processed_feats[:, 0:11]
    # -
    set8 = scale_data(processed_feats) if scale else processed_feats

    ablation_sets = [set1, set2, set3, set4, set5, set6, set7, set8]

    return ablation_sets, y


def run_logreg(X, y, condition=str, save=True):
    """
    Fits logistic regression model for given condition and returns fitted model
    :param X: ndarray of size (n, f) with f number of features for n number of cases
    :param y: ndarray of length n with gold data
    :param condition: (str) condition being tested, used for command line output and file naming
    :param save: (bool)
    :return model: sklearn.linear_model.LogisticRegression() instance fitted for the given data
    """
    warnings.filterwarnings('ignore', r"The max_iter was reached.*")
    print(f"Fitting model for condition: {condition}")
    # model = LogisticRegression(random_state=0, solver='sag')
    model = LogisticRegression(max_iter=10000)
    model.fit(X, y)
    print(f"\tDone.")

    if save:
        with open(f"{condition}.pkl", 'wb') as f:
            pickle.dump(model, f)

    return model


def get_zero_rule(y):
    """
    For gold standard tags ndarray y, returns ndarray of length len(y) with zero rule tags, that is the most common tag
    as the prediction every time
    :param y: gold standard tags
    :return zero_rule: ndarray with zero rule tags for scoring
    """
    zero_rule = np.zeros((len(y)))
    most_common = 1 if sum(y) > 0.5 * len(y) else 0
    for i in range(len(y)):
        zero_rule[i] = most_common
    return zero_rule, most_common


def score_model(condition, test_y, predictions):
    """
    Scores models
    :param condition: (str) name of the condition being tested
    :param test_y: gold standard helpfulness (binary)
    :param predictions: predicted helpfulness (binary)
    :param save_out: (bool) if True, saves output of scorer to file
    :return:
    """
    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    f1 = f1_score(test_y, predictions)
    scores = f"""\t{condition}:
            Accuracy: {accuracy:0.03}
            Precision: {precision:0.03}
            Recall: {recall:0.03}
            F1: {f1:0.03}
            """
    print(scores)
    return scores


def ablation(train_file, train_features, test_file, test_features, scale=False, minimum_votes=10.0, help_boundary=0.6):
    """
    Runs ablation for train and test files
    :param train_file: either a path to tsv or data_dict, file to train LogReg
    :param train_features: feature array for train
    :param test_file: either a path to tsv or data_dict, file to test LogReg
    :param test_features: feature array for test
    :param scale: (bool) whether or not to scale data before Logistic Regression
    :param minimum_votes: (float) minimum number of helpfulness votes to be included in ablation sets
    :param help_boundary: (float) percentage (0-1) of votes needed to be considered helpful
    """
    # Loads data dicts
    if train_file is str and test_file is str:
        reviewer_data = ReviewerData(data_file=train_file, delimiter="\t")
        train_data_dict = reviewer_data.data_dict
        test_data = ReviewerData(data_file=test_file, delimiter='\t')
        test_data_dict = test_data.data_dict
    else:
        train_data_dict = train_file
        test_data_dict = test_file

    # Hyperparameters
    scale = scale
    minimum_votes = minimum_votes
    help_boundary = help_boundary

    # Writes to log file
    date = datetime.datetime.today()
    log_file = open(f"log_{date.month}_{date.day}_{date.hour}_{date.minute}.txt", 'w')
    log_file.write(f"Log file for experiment on {date.month}/{date.day} at {date.hour}:{date.minute}\n"
                   f"Train File: {train_file}\nTest File: {test_file}\n\n"
                   f"Hyperparameters:\n"
                   f"----------------\n"
                   f"\tScaled Data: {scale}\n"
                   f"\tMinimum Votes Needed: {minimum_votes}\n"
                   f"\tHelpfulness Cutoff Point: {help_boundary}\n\n")

    # Creates Ablation sets for Train and Test
    train_ablation_sets, y = create_ablation_sets(feature_array=train_features, data=train_data_dict, scale=scale,
                                            minimum_votes=minimum_votes, help_boundary=help_boundary)
    test_ablation_sets, test_y = create_ablation_sets(feature_array=test_features, data=test_data_dict,
                                                 scale=scale, minimum_votes=minimum_votes,
                                                 help_boundary=help_boundary)
    X1, X2, X3, X4, X5, X6, X7, X8 = train_ablation_sets
    tX1, tX2, tX3, tX4, tX5, tX6, tX7, tX8 = test_ablation_sets
    zero_rule, most_common = get_zero_rule(test_y)

    # Writes data to log file
    log_file.write(f"Model Information:\n"
                   f"------------------\n"
                   f"\tNumber of Train Reviews: {len(X1[:, 0])}\n"
                   f"\tNumber of Test Reviews: {len(tX1[:, 0])}\n"
                   f"\tZero Rule Value: {most_common}\n\n"
                   f"Results:\n"
                   f"--------\n")

    # Dictionary for looping models
    ablation = {"model1": (X1, tX1),
                "model2": (X2, tX2),
                "model3": (X3, tX3),
                "model4": (X4, tX4),
                "model5": (X5, tX5),
                "model6": (X6, tX6),
                "model7": (X7, tX7),
                "model8": (X8, tX8)}

    # iteratively fits, predicts, and scores each model
    for condition, features in ablation.items():
        X, test_X = features
        model = run_logreg(X, y, condition=condition, save=False)
        predictions = model.predict(test_X)
        print("\tPredicted Zero Rule") if len(predictions) == sum(predictions) else print("\tDidn't Predict Zero Rule")
        scores = score_model(condition=condition, test_y=test_y, predictions=predictions)
        log_file.write(scores + '\n')

    # Scores and writes zero rule to log file
    zero_rule_score = score_model(condition="Zero Rule", test_y=test_y, predictions=zero_rule)
    log_file.write(zero_rule_score)
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', type=str, default="./data/sample.tsv", help="train file csv/tsv")
    parser.add_argument('-f', '--train_feature_array', default="./data/feature_arrays/sample_features.npy",
                        help="feature array for train_file")
    parser.add_argument('-s', '--test_file', type=str, default="./data/sample.tsv", help="test file csv/tsv")
    parser.add_argument('-a', '--test_feature_array', default="./data/feature_arrays/sample_features.npy",
                        help="feature array for test_file")
    args = parser.parse_args()
    train = args.train_file
    train_feats = args.train_feature_array
    test = args.test_file
    test_feats = args.test_feature_array

    ablation(train, train_feats, test, test_feats, scale=False, help_boundary=0.75, minimum_votes=15)
