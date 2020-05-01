## First Steps: Environment

- Download `reviews.csv` from
  [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews/) and
  save it to the `data` directory

- With Conda, create a new conda environment from the helpfulness.yml
  file in dependencies directory

- Edit the prefix variable in helpfulness.yml at the very bottom to the
  path for your anaconda3, if you are on MacOS you likely only need to
  change the ???? to the name of your machine

        $ conda env create -f helpfulness.yml

- Activate the Environment:

        $ conda activate helpfulness

- Download the spaCy model (if you are computing features, else this can be skipped)

        $ python spacy download en_core_web_lg

- Ensure special dependencies downloaded 

        $ pip show spacy-readability
        ... 
        $ pip show en_core_web_lg 

## Recommended Command Line Input:
        
##### Running Scripts

- Start to finish: reviews.csv -> log file (note: this will take a long
  time, and is CPU intensive)
 
        $ python data_util.py --data_path "./data/reviews.csv"
        ...
        $ python get_features.py --data_path "train.tsv" --name "all_features"
        ...
        $ python get_features.py --data_path "dev_test.tsv" --name "dev_test_features"
        ... 
        $ python ablation.py --train_file "train.tsv" --train_feature_array "all_features.npy" \
        --test_file "dev_test.tsv" --test_feature_array "dev_test_features.npy"
 
-  The log will then be saved to the cwd with the name
   log_(month)\_(day)\_(hour)_(minutes).txt as to not overwrite other
   log file when tuning hyperparameters
   
## Recommended Python Implementation

- Start to finish

        from data_util import ReviewerData
        from get_features import HelpfulnessVectorizer
        from ablation import ablation
        
        reviews = "./data/reviews.csv"
        
        reviewer_data = ReviewerData(data_file=reviews, delimiter=',')
        reviewer_data.split_data()
        
        train_data = reviewer_data.train
        train_vectorizer = HelpfulnessVectorizer(data=train_data)
        train_vectors = test_vectorizer.get_features()
        
        test_data = reviewer_data.test
        test_vectorizer = HelpfulnessVectorizer(data=test_data)
        test_vectors = test_vectorizer.get_features()
        
        ablation(train_data, train_vectors, test_data, test_vectors)
        
- Using Pre-Extracted Features from data/feature_arrays
    
        import numpy as np
        from data_util import ReviewerData
        from get_features import HelpfulnessVectorizer
        from ablation import ablation
        
        train_path = "./data/train.tsv"
        test_path = "./data/test.tsv"
        
        train = ReviewerData(data_file=train_path, delimiter='\t')
        train_data = train_data.data_dict
        
        test = ReviewerData(data_file=test_path, delimiter='\t')
        test_data = test_data.data_dict

        train_vectors = np.load("data/feature_arrays/train_features.npy")
        test_vectors = np.load("data/feature_arrays/dev_test_features.npy")
        
        ablation(train_data, train_vectors, test_data, test_vectors)
            
        
    
        
