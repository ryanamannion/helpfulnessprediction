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
from data_util import ReviewerData, preprocess_text
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

    def __init__(self, data, spacy_model='en_core_web_lg'):
        """
        Instantiates SimilarityVectorizer, loads spaCy model
        """
        print("Initializing spaCy...")
        self.nlp = spacy.load(spacy_model)
        self.read = Readability()
        self.nlp.add_pipe(self.read, last=True)
        self.data = data

        # Populated if get_features() performed
        self.feature_array = None

    def get_features(self):
        """
        Extracts features from review texts and fills in a numpy ndarray

        Meta Features:
            1 - Star rating of five

        Simple Text Features:
            2 - Number of sentences
            3 - Number of tokens
            4 - Number of characters
            5 - Average tokens/ sentence
            6 - Average characters/ token

        Readability Features:
            7 - Flesch-Kincaid Grade Level
            8 - Flesch-Kincaid Reading Ease
            9 - Dale Chall
            10 - SMOG
            11 - Coleman-Liau Index
            12 - Automated Readability Index
            13 - FORCAST

        Other Features:
            ## - TODO: TF/IDF
            ## - TODO: Humor?
            ## - TODO: Summary Features

        data has columns: Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,
                            Score,Time,Summary,Text

        :return: numpy nd array of shape [# reviews, 13]
        """
        # selects sample from data dictionary for length
        sample = list(self.data.values())[0]

        my_array = np.zeros((len(sample), 13))   # as many rows as sample is long, 13 features wide

        texts = self.data["Text"]
        print(f"Extracting Features for {len(texts)} Texts...")
        for i, text in tqdm(enumerate(texts)):
            text = preprocess_text(text=text)
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

        self.feature_array = my_array

        return my_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/sample.tsv",
                        help="tsv/csv file containing data to be read")
    args = parser.parse_args()

    data = args.data_path

    reviewer_data = ReviewerData(data_file=data, delimiter='\t')

    my_vectorizer = HelpfulnessVectorizer(reviewer_data.data_dict)

    feature_array = my_vectorizer.get_features()

    np.save("train_x13_feats", feature_array)
