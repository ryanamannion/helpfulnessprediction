"""
get_features.py
creates a class to get and fill a feature matrix for a logistic regression model

Ryan A. Mannion
Written for LING472 Final Project
"""
import spacy
import numpy as np
import argparse
from data_util import read_data
from tqdm import tqdm


class HelpfulnessVectorizer:
    """Creates a matrix of features for helpfulness prediction"""

    def __init__(self):
        """
        Instantiates SimilarityVectorizer, loads spaCy model
        """
        print("Initializing spaCy...")
        self.nlp = spacy.load("en_core_web_lg")

    def get_simple_features(self, data):
        """
        6 features in this function:
            - Star rating of five
            - Number of sentences
            - Number of tokens
            - Number of characters
            - Average tokens/ sentence
            - Average characters/ token

        data has columns: Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,
                            Score,Time,Summary,Text
        :param data: dictionary containing review information, output from util.py's function read_data
        :return: numpy nd array of shape [# reviews, 6]
        """
        # selects sample from data dictionary for length
        sample = list(data.values())[0]

        my_array = np.zeros((len(sample), 6))   # as many rows as sample is long, 6 features wide

        texts = data["Text"]
        print("Getting Simple Features...")
        for i, text in tqdm(enumerate(texts)):
            doc = self.nlp(text)

            num_sents = 0
            num_toks = 0
            num_chars = 0
            toks_per_sent = []
            char_per_tok = []

            # collects information needed for features
            for sent in doc.sents:
                num_sents += 1      # increment by one each sentence
                num_toks += sent.end        # increment by position of end token
                num_chars += sent.end_char      # increment by position of end char
                toks_per_sent.append(sent.end)      # append number of tokens
                for token in sent:
                    char_per_tok.append(len(token))

            # populate array
            my_array[i, 0] = data["Score"][i]  # score/ star rating of review
            my_array[i, 1] = num_sents
            my_array[i, 2] = num_toks
            my_array[i, 3] = num_chars
            my_array[i, 4] = sum(toks_per_sent) / len(toks_per_sent)
            my_array[i, 5] = sum(char_per_tok) / len(char_per_tok)

        return my_array

    # TODO: add feature function for more complex features

    # TODO: get features, combine arrays with numpy.concatenate (or hstack)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="reviews.csv",
                        help="csv file containing data to be read")
    args = parser.parse_args()

    print("Loading Data...")
    my_data = read_data(args.data_path)

    my_vectorizer = HelpfulnessVectorizer()
    my_vectorizer.get_simple_features(data=my_data)
