# Helpfulness Prediction Data Visualization
# Ryan A. Mannion

setwd("/users/ryanmannion/LING472/helpfulnessprediction")
review_data <- read.table("review_data.tab", header=T)
names(review_data) <- c("hnum", "hdenom", "score")
head(review_data)

attach(review_data)
# Calculate Helpfulness Scores
help_score <- hnum/hdenom

# Find NaN
num_nan <- sum(is.nan(help_score))

# Removes the 270052 NaN reviews 
help_score_no_nan <- subset(help_score, !is.nan(help_score))

# Checks Length
length(help_score) - num_nan == length(help_score_no_nan)

# Histogram of Scores
hist(help_score_no_nan, xlim=c(0,1), main="Histogram of Helpfulness Scores (Without NaN Vlues)", 
     xlab="Helpfulness Score (helpful votes/total votes)")

detach(review_data)


