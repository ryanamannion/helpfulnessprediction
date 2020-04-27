"""
get_features.py
creates a class to get and fill a feature matrix for a logistic regression model

Ryan A. Mannion
Written for LING472 Final Project
"""
import spacy
from spacy_readability import Readability
import numpy as np
import argparse
from data_util import ReviewerData
from tqdm import tqdm


class HelpfulnessVectorizer:
    """Creates a matrix of features for helpfulness prediction"""

    def __init__(self, data):
        """
        Instantiates SimilarityVectorizer, loads spaCy model
        """
        print("Initializing spaCy...")
        self.nlp = spacy.load("en_core_web_lg")
        self.read = Readability()
        self.nlp.add_pipe(self.read, last=True)
        self.data = data

    def get_features(self):
        """
        Simple Features in this function:
            - Star rating of five
            - Number of sentences
            - Number of tokens
            - Number of characters
            - Average tokens/ sentence
            - Average characters/ token

        Readability Features in this function:
            - Flesch-Kincaid Grade Level
            - Flesch-Kincaid Reading Ease
            - Dale Chall
            - SMOG
            - Coleman-Liau Index
            - Automated Readability Index
            - FORCAST

        data has columns: Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,
                            Score,Time,Summary,Text
        :return: numpy nd array of shape [# reviews, 13]
        """
        # selects sample from data dictionary for length
        sample = list(self.data.values())[0]

        my_array = np.zeros((len(sample), 13))   # as many rows as sample is long, 6 features wide

        texts = self.data["Text"]
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

            # Simple Features
            my_array[i, 0] = self.data["Score"][i]  # score/ star rating of review
            my_array[i, 1] = num_sents
            my_array[i, 2] = num_toks
            my_array[i, 3] = num_chars
            my_array[i, 4] = sum(toks_per_sent) / len(toks_per_sent)
            my_array[i, 5] = sum(char_per_tok) / len(char_per_tok)
            # Readability Scores
            my_array[i, 6] = doc._.flesch_kincaid_grade_level
            my_array[i, 7] = doc._.flesch_kincaid_reading_ease
            my_array[i, 8] = doc._.dale_chall
            my_array[i, 9] = doc._.smog
            my_array[i, 10] = doc._.coleman_liau_index
            my_array[i, 11] = doc._.automated_readability_index
            my_array[i, 12] = doc._.forcast

        return my_array


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default="reviews.csv",
    #                     help="csv file containing data to be read")
    # args = parser.parse_args()

    data = './data/train.tsv'

    print(f"Loading Data: {data} ...")
    reviewer_data = ReviewerData(data_file=data, delimiter='\t')

    my_vectorizer = HelpfulnessVectorizer(reviewer_data.data_dict)

    feature_array = my_vectorizer.get_features()

    print(feature_array)


if __name__ == "__main__":
    main()
