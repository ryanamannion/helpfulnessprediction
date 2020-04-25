# Helpfulness Prediction Data Visualization
# Ryan A. Mannion

setwd("/users/ryanmannion/LING472/helpfulnessprediction")
review_data <- read.table("review_data.tab", header=T)
names(review_data) <- c("hnum", "hdenom", "score")
head(review_data)

attach(review_data)
# Calculate Helpfulness Scores
help_score_raw <- hnum/hdenom

# Find NaN
num_nan <- sum(is.nan(help_score_raw))

# Removes the 270052 NaN reviews 
help_score_no_nan <- subset(help_score_raw, !is.nan(help_score_raw))

# Checks Length
length(help_score_raw) - num_nan == length(help_score_no_nan)

# Finds Scores out of Range
max(help_score_no_nan) # there should not be a value of 3
sum(help_score_no_nan > 1) # there are two values greater than 1
help_scores <- help_score_no_nan[!help_score_no_nan > 1] # remove those values

# Histogram of Scores
hist(help_score_no_nan, xlim=c(0,1), 
     main="Histogram of Helpfulness Scores", 
     xlab="Helpfulness Score (helpful votes/total votes)")

detach(review_data)


