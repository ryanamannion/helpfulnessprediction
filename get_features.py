"""
get_features.py
creates a class to get and fill a feature matrix for a logistic regression model

Ryan A. Mannion
Written for LING472 Final Project
"""
import spacy
import numpy as np


class HelpfulnessVectorizer:
    """Creates a matrix of features for helpfulness prediction"""

    def __init__(self):
        """
        Instantiates SimilarityVectorizer, loads spaCy model
        """
        nlp = spacy.load("en_core_web_lg")

    def get_simple_features(self, data):
        """
        Features in this function:
            - Number of sentences
            - Number of words
            - Average words/ sentence
            - Number of characters
        :param data:
        :return: numpy nd array of shape TODO: specify shape
        """
        pass

    # TODO: add feature function for more complex features

    # TODO: get features, combine arrays with numpy.concatenate
