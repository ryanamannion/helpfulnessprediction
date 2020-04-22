# Helpfulness Prediction Data Visualization
# Ryan A. Mannion

setwd("/users/ryanmannion/LING472/helpfulnessprediction")
review_data <- read.table("review_data.tab", header=T)
names(review_data) <- c("hnum", "hdenom", "score")
head(review_data)

attach(review_data)
help_score <- hnum/hdenom
help_score[is.nan(help_score)] <- 0

hist(help_score, xlim=c(0,1), main="Histogram of Helpfulness Scores", 
     xlab="Helpfulness Score (helpful/total votes)")

detach(review_data)


