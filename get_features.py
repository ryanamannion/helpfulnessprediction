"""
get_features.py
creates a class to get and fill a feature matrix for a logistic regression model

Ryan A. Mannion
Written for LING472 Final Project
"""
import argparse
import numpy as np
import spacy
from spacy_readability import Readability
from data_util import ReviewerData, remove_html
from tqdm import tqdm


class HelpfulnessVectorizer:
    """
    Creates a matrix of features for helpfulness prediction

    Attributes:
        nlp (spacy object): spacy
        read (class): Readability pipeline for spacy
        data (dict): dictionary containing reviewer data
        feature_array (numpy.ndarray): contains extracted features for logistic regression model

    Methods:
        get_features: Extracts features from review texts and fills in a numpy ndarray
    """

    def __init__(self, data, spacy_model='en_core_web_lg', feature_array=None):
        """
        Instantiates SimilarityVectorizer, loads spaCy model
        """
        print("Initializing spaCy...")
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)
        self.read = Readability()
        self.nlp.add_pipe(self.read, last=True)
        self.data = data
        self.feature_array = feature_array

    def get_features(self):
        """
        Extracts features from review reviews and fills in a numpy ndarray with shape (#ofdocs, 18)

        Summary Text Features:
            0 - Number of sentences
            1 - Number of tokens
            2 - Number of characters
            3 - Average tokens/ sentence
            4 - Average characters/ token
        Meta Features:
            5 - Star rating of five
        Review Text Features:
            6 - Number of sentences
            7 - Number of tokens
            8 - Number of characters
            9 - Average tokens/ sentence
            10 - Average characters/ token
        Readability Features (Review text Only):
            11 - Flesch-Kincaid Grade Level
            12 - Flesch-Kincaid Reading Ease
            13 - Dale Chall
            14 - SMOG
            15 - Coleman-Liau Index
            16 - Automated Readability Index
            17 - FORCAST

        data has columns: Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,
                            Score,Time,Summary,Text

        :return: numpy nd array of shape [#ofdocs, 18]
        """
        # selects sample from data dictionary for length
        sample = list(self.data.values())[0]

        my_array = np.zeros((len(sample), 18))
        # as many rows as sample is long, 13 features wide

        reviews = self.data["Text"]
        summaries = self.data["Summary"]

        print(f"Extracting Features for {len(reviews)} Reviews...\nThis might take a while")
        for i, combined in tqdm(enumerate(zip(summaries, reviews))):
            summary_text, review_text = combined
            review_text = remove_html(text=review_text)
            summary_text = remove_html(text=summary_text)
            summary_doc = self.nlp(summary_text)
            review_doc = self.nlp(review_text)

            # Summary Stats
            s_num_sents = 0
            s_num_toks = 0
            s_num_chars = 0
            s_toks_per_sent = []
            s_char_per_tok = []
            # Review Stats
            r_num_sents = 0
            r_num_toks = 0
            r_num_chars = 0
            r_toks_per_sent = []
            r_char_per_tok = []

            # collects information needed for features
            for sent in summary_doc.sents:
                s_num_sents += 1      # increment by one each sentence
                s_num_toks += sent.end        # increment by position of end token
                s_num_chars += sent.end_char      # increment by position of end char
                s_toks_per_sent.append(sent.end)      # append number of tokens
                for token in sent:
                    s_char_per_tok.append(len(token))
            for sent in review_doc.sents:
                r_num_sents += 1      # increment by one each sentence
                r_num_toks += sent.end        # increment by position of end token
                r_num_chars += sent.end_char      # increment by position of end char
                r_toks_per_sent.append(sent.end)      # append number of tokens
                for token in sent:
                    r_char_per_tok.append(len(token))

            # Summary - Simple Features
            my_array[i, 0] = s_num_sents
            my_array[i, 1] = s_num_toks
            my_array[i, 2] = s_num_chars
            my_array[i, 3] = sum(s_toks_per_sent) / len(s_toks_per_sent)
            my_array[i, 4] = sum(s_char_per_tok) / len(s_char_per_tok)
            # Review - Simple Features
            my_array[i, 5] = self.data["Score"][i]  # score/ star rating of review
            my_array[i, 6] = r_num_sents
            my_array[i, 7] = r_num_toks
            my_array[i, 8] = r_num_chars
            my_array[i, 9] = sum(r_toks_per_sent) / len(r_toks_per_sent)
            my_array[i, 10] = sum(r_char_per_tok) / len(r_char_per_tok)
            # Review - Readability Scores
            my_array[i, 11] = review_doc._.flesch_kincaid_grade_level
            my_array[i, 12] = review_doc._.flesch_kincaid_reading_ease
            my_array[i, 13] = review_doc._.dale_chall
            my_array[i, 14] = review_doc._.smog
            my_array[i, 15] = review_doc._.coleman_liau_index
            my_array[i, 16] = review_doc._.automated_readability_index
            my_array[i, 17] = review_doc._.forcast

        self.feature_array = my_array

        return my_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/sample.tsv",
                        help="tsv/csv file containing data to be read")
    parser.add_argument('--name', type=str, default="sample_features", help="what to name the output file")
    args = parser.parse_args()

    data = args.data_path
    name = args.name

    reviewer_data = ReviewerData(data_file=data, delimiter='\t')

    my_vectorizer = HelpfulnessVectorizer(reviewer_data.data_dict)

    my_feature_array = my_vectorizer.get_features()

    np.save(name, my_feature_array)
