#!/usr/bin/env python3
"""
ablation.py


"""
import argparse
import numpy as np
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from data_util import ReviewerData, filter_data


def scale_data(data):
    """
    Scales data as preprocessing step for LogisticRegression() instance
    :param data: ndarray to be scaled
    :return scaled_data: the scaled data to be used in LogisticRegression() instance as X
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def create_ablation_sets(feature_array, data, scale=True):
    """"
    Takes a full feature array with matching data_dict of ReviewerData instance as input and returns 5 sets of features:

    Ablation Feature Sets:
    X1 - Review Simple
    X2 - Review + Readability
    X3 - AllReview + Meta
    X4 - AllReview + Summary
    X5 - All

    :param feature_array:
    :param data:
    :param scale:
    :returns X(1-5): feature sets for the five ablation conditions
    :returns y: gold standard labels

    """

    if isinstance(feature_array, str):
        all_features = np.load(feature_array) if feature_array.endswith(".npy") else np.load(f"{feature_array}.npy")
    else:
        all_features = feature_array

    processed_feats, y = filter_data(data_dict=data, feature_array=all_features, minimum_votes=10, help_boundary=0.9)

    summary_feats = processed_feats[:, 0:5]
    review_simple_feats = processed_feats[:, 6:11]
    readability_feats = processed_feats[:, 11:]

    # Creates feature sets and scales if scale=True
    set1 = scale_data(review_simple_feats) if scale else review_simple_feats
    # -
    set2_concat = np.concatenate((review_simple_feats, readability_feats), axis=1)
    set2 = scale_data(set2_concat) if scale else set2_concat
    # -
    set3 = scale_data(processed_feats[:, 5:]) if scale else processed_feats[:, 5:]
    # -
    set4_concat = np.concatenate((summary_feats, review_simple_feats, readability_feats), axis=1)
    set4 = scale_data(set4_concat) if scale else set4_concat
    # -
    set5 = scale_data(processed_feats) if scale else processed_feats

    return set1, set2, set3, set4, set5, y


def run_logreg(X, y, condition=str, save=True):
    """
    Fits logistic regression model for given condition and returns fitted model
    :param X: (np.ndarray) of size (n, f) with f number of features for n number of cases
    :param y: (np.ndarray) of length n with gold data
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
    return zero_rule


def score_model(condition, test_y, predictions, save_out=True):
    """
    Scores models and if save_out is True, saves output to file
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
    scores = f"""\tResults for model {condition}:
            Accuracy: {accuracy:0.03}
            Precision: {precision:0.03}
            Recall: {recall:0.03}
            F1: {f1:0.03}
            """
    print(scores)
    if save_out:
        with open(f"{condition}_scores.txt", 'w') as f:
            f.write(scores)


def main():
    # Argparse Routine
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', type=str, default="./data/sample.tsv", help="train file csv/tsv")
    parser.add_argument('-f', '--train_feature_array', default="./data/feature_arrays/sample_features.npy",
                        help="feature array for train_file")
    parser.add_argument('-s', '--test_file', type=str, default="./data/sample.tsv", help="test file csv/tsv")
    parser.add_argument('-a', '--test_feature_array', default="./data/feature_arrays/sample_features.npy",
                        help="feature array for test_file")

    args = parser.parse_args()
    train_file = args.train_file
    feature_array = args.train_feature_array
    test_file = args.test_file
    test_features = args.test_feature_array

    # Loads data dicts
    reviewer_data = ReviewerData(data_file=train_file, delimiter="\t")
    train_data_dict = reviewer_data.data_dict
    test_data = ReviewerData(data_file=test_file, delimiter='\t')
    test_data_dict = test_data.data_dict

    # Creates Ablation sets for Train and Test
    X1, X2, X3, X4, X5, y = create_ablation_sets(feature_array=feature_array, data=train_data_dict, scale=False)
    tX1, tX2, tX3, tX4, tX5, test_y = create_ablation_sets(feature_array=test_features, data=test_data_dict,
                                                           scale=False)

    ablation = {"model1": (X1, tX1),
                "model2": (X2, tX2),
                "model3": (X3, tX3),
                "model4": (X4, tX4),
                "model5": (X5, tX5)}

    zero_rule = get_zero_rule(test_y)

    # iteratively fits, predicts, and scores each model
    for condition, features in ablation.items():
        X, test_X = features
        model = run_logreg(X, y, condition=condition, save=False)
        predictions = model.predict(test_X)
        print("Guessed all 1s") if len(predictions) == sum(predictions) else print("Didn't guess all 1s")
        score_model(condition=condition, test_y=test_y, predictions=predictions, save_out=True)

    score_model(condition="Zero Rule", test_y=test_y, predictions=zero_rule, save_out=True)


if __name__ == "__main__":
    main()
