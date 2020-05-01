## First Steps: Environment

- Download data from
  [Google Drive Link](https://drive.google.com/file/d/13ZrQZJkeFly_C5NgPr6caV4-twK4KUpm/view?usp=sharing)
  and unzip. Move `reviews.csv` and `train.tsv` to this repo's `data`
  directory

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
        $ python get_features.py --data_path "train.tsv" --name "train"
        ...
        $ python get_features.py --data_path "dev_test.tsv" --name "dev_test"
        ... 
        $ python ablation.py --train_file "train.tsv" --train_feature_array "train.npy" \
        --test_file "dev_test.tsv" --test_feature_array "dev_test.npy"
 
-  The log will then be saved to the cwd with the name
   log_(month)\_(day)\_(hour)_(minutes).txt as to not overwrite other
   log file when tuning hyperparameters
   
- To test this without the long run time, use `sample.tsv` and
  `dev_test_sample.tsv` instead of `train.tsv` and `dev_test.tsv`
   
## Recommended Python Implementation:

- Start to finish

        from data_util import ReviewerData
        from get_features import HelpfulnessVectorizer
        from ablation import ablation
        
        # To load, split, and use ~full~ datasets, use this code
        # reviews = "./data/reviews.csv"
        # reviewer_data = ReviewerData(data_file=reviews, delimiter=',')
        # reviewer_data.split_data()
        # train_data = reviewer_data.train
        # test_data = reviewer_data.dev_test
        
        # To load pre-split sample datasets, use this code
        train_data = ReviewerData(data_file="data/sample.tsv", delimiter='\t')
        train_vectorizer = HelpfulnessVectorizer(data=train_data)
        train_vectors = train_vectorizer.get_features()
        test_data = ReviewerData(data_file="data/dev_test_sample.tsv", delimiter='\t')
        test_vectorizer = HelpfulnessVectorizer(data=test_data)
        test_vectors = test_vectorizer.get_features()

        ablation(train_data, train_vectors, test_data, test_vectors)
        
- Using Pre-Extracted Features from data/feature_arrays
    
        from data_util import ReviewerData
        from ablation import ablation
        
        train_path = "./data/train.tsv"
        train_vectors = "data/train.npy"
        test_path = "./data/dev_test.tsv"
        test_vectors = "data/dev_test.npy"
        
        train_data = ReviewerData(data_file=train_path, delimiter='\t')
        
        test_data = ReviewerData(data_file=test_path, delimiter='\t')
        
        ablation(train_data, train_vectors, test_data, test_vectors)
            
        
    
        
