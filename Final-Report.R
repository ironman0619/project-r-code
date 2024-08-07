# firstly, load some important libraries
library(ggplot2)
library(ggplotify)
library(gridExtra)
library(knitr)
library(patchwork)
library(tidyverse)
library(dplyr)
library(lubridate)
library(tidyr)
library(Metrics)
library(caret)
library(kableExtra)



setwd("/Users/hesun/Desktop")
getwd()




# BCM model(line 42 - line 138)
# Naive model(line 148 - line 202)
# Logistic model(line 213 - line 269)
# Elo k-factor model(line 288- line 383 )
# Elo k-factor model tuning parameter(line 399 - line 493)
# Elo 538 model(line 517 - line 610)
# Elo 538 model tuning parameter(line 624 - line 695)
# Extension for Elo k-factor model(line 706 - line 789)
# Extension for Elo 538 model(line 812 - line 914)










# BCM model 

# load the dataset with betting odds
tennis_data <- read.csv("combined_tennis_data_2010_2019.csv")
tennis_data$Date <- as.Date(tennis_data$Date, format = "%Y-%m-%d")

# Determine if the higher-ranked player won
tennis_data$higher_rank_won <- tennis_data$WRank < tennis_data$LRank

# Calculate points for higher and lower-ranked players
tennis_data <- tennis_data %>%
  mutate(
    higher_rank_points = ifelse(higher_rank_won, WPts, LPts),
    lower_rank_points = ifelse(higher_rank_won, LPts, WPts),
    point_difference = higher_rank_points - lower_rank_points
  )

# Split the dataset into training, validation, and testing sets based on year
train_set <- tennis_data %>% filter(year(Date) < 2018)
validation_set <- tennis_data %>% filter(year(Date) >= 2018 & year(Date) < 2019)
test_set <- tennis_data %>% filter(year(Date) == 2019)

# Remove rows with missing values in the key columns
train_set <- train_set %>% drop_na()
validation_set <- validation_set %>% drop_na()
test_set <- test_set %>% drop_na()

# Function to calculate normalized and consensus probabilities
calculate_probabilities <- function(df) {
  df <- df %>%
    mutate(
      p1_b365 = B365L / (B365W + B365L),
      p2_b365 = B365W / (B365W + B365L),
      p1_ps = PSL / (PSW + PSL),
      p2_ps = PSW / (PSW + PSL)
    ) %>%
    mutate(
      consensus_p1 = (p1_b365 + p1_ps) / 2,
      consensus_p2 = (p2_b365 + p2_ps) / 2,
      higher_rank_prob = ifelse(WRank < LRank, consensus_p1, consensus_p2),
      lower_rank_prob = ifelse(WRank < LRank, consensus_p2, consensus_p1),
      predicted_winner = ifelse(higher_rank_prob > lower_rank_prob, "W", "L")
    )
  return(df)
}

# Apply the probability calculation function to the datasets
train_set <- calculate_probabilities(train_set)
test_set <- calculate_probabilities(test_set)
validation_set <- calculate_probabilities(validation_set)


# to construct the metrics for accuracy, 
evaluate_performance <- function(df) {
  df <- df %>%
    mutate(
      actual_winner = ifelse(WRank < LRank, 1, 0)
    )
  
  correct_preds <- sum(df$predicted_winner == ifelse(df$actual_winner == 1, "W", "L"))
  total_preds <- nrow(df)
  accuracy <- correct_preds / total_preds
  
  log_loss <- -mean(df$actual_winner * log(df$higher_rank_prob) + 
                      (1 - df$actual_winner) * log(1 - df$higher_rank_prob), na.rm = TRUE)
  
  calibration <- sum(df$higher_rank_prob, na.rm = TRUE) / sum(df$actual_winner, na.rm = TRUE)
  
  list(
    accuracy = accuracy,
    log_loss = log_loss,
    calibration = calibration
  )
}

# then we perform the model 
train_perf <- evaluate_performance(train_set)
test_perf <- evaluate_performance(test_set)
validation_perf <- evaluate_performance(validation_set)





# Compile performance metrics
performance_metrics <- data.frame(
  dataset = c("Training", "Validation", "Testing"),
  accuracy = c(train_perf$accuracy, validation_perf$accuracy, test_perf$accuracy),
  log_loss = c(train_perf$log_loss, validation_perf$log_loss, test_perf$log_loss),
  calibration = c(train_perf$calibration, validation_perf$calibration, test_perf$calibration)
)

# Print performance metrics.
kable(performance_metrics, caption = "Model Performance Metrics for BCM") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width = F, 
                position = "center")









# naive model 
# Extract 2018 data from the validation set for training the naive model
train_data_naive <- validation_set %>% filter(year(Date) == 2018)

# Compute the average win probability of the higher-ranked player in 2018
win_prob_2018 <- mean(train_data_naive$higher_rank_won, na.rm = TRUE)

# Create a vector of predictions for 2019, with each entry equal to the 2018 win probability
preds_2019 <- rep(win_prob_2018, nrow(test_set))

# Get the actual outcomes for the 2019 test set
actual_outcomes_2019 <- test_set$higher_rank_won

# Determine the accuracy of the naive model by comparing predictions to actual outcomes
correct_predictions_2019 <- ifelse(preds_2019 > 0.5, TRUE, FALSE)
naive_accuracy <- mean(correct_predictions_2019 == actual_outcomes_2019)

# Calculate the calibration metric by dividing the sum of predicted probabilities by the sum of actual outcomes
sum_predictions <- sum(preds_2019)
sum_actual_outcomes <- sum(actual_outcomes_2019)
calibration_naive <- sum_predictions / sum_actual_outcomes

# Compute the log loss to evaluate the naive model
log_loss_values <- actual_outcomes_2019 * log(win_prob_2018) + 
  (1 - actual_outcomes_2019) * log(1 - win_prob_2018)
log_loss_naive <- -mean(log_loss_values, na.rm = TRUE)

# Store the naive model's performance metrics in a data frame
naive_metrics <- data.frame(
  dataset = "Naive Testing",   
  accuracy = naive_accuracy,   
  log_loss = log_loss_naive,   
  calibration = calibration_naive  
)

# Add the naive model's metrics to the existing performance metrics data frame
performance_metrics <- rbind(performance_metrics, naive_metrics)

# Create a separate data frame for displaying the naive model's metrics
display_naive_metrics <- data.frame(
  dataset = "Naive Testing",   # Indicate the dataset type
  accuracy = naive_accuracy,   # Store the accuracy value
  log_loss = log_loss_naive,   # Store the log loss value
  calibration = calibration_naive  # Store the calibration metric
)

# Use the kable function to format the table for displaying the naive model's performance metrics
print(
  kable(display_naive_metrics, caption = "Naive Model Performance Metrics for naive") %>%
    kable_styling(
      bootstrap_options = c("striped", "hover", "condensed", "responsive"),  # Apply table styling
      full_width = FALSE,  # Set table width to not full width
      position = "center"  # Center align the table
    )
)










# logistic model : train a logistic regression model
logistic_regression_model <- glm(
  higher_rank_won ~ point_difference + 0,
  data = train_set,
  family = binomial(link = 'logit')
)

# Predict probabilities on the test set
predicted_probabilities_logistic <- predict(logistic_regression_model, newdata = test_set, type = "response")

# Generate binary predictions based on the predicted probabilities
predicted_outcomes_logistic <- ifelse(predicted_probabilities_logistic > 0.5, 1, 0)

# Calculate the accuracy of the logistic regression model
accuracy_of_logistic_model <- mean(predicted_outcomes_logistic == test_set$higher_rank_won)

# Compute the log loss for the logistic regression model
logarithmic_loss_logistic <- -mean(
  test_set$higher_rank_won * log(predicted_probabilities_logistic) + 
    (1 - test_set$higher_rank_won) * log(1 - predicted_probabilities_logistic), 
  na.rm = TRUE
)

# Compute the calibration metric for the logistic regression model
calibration_metric_logistic <- sum(predicted_probabilities_logistic) / sum(test_set$higher_rank_won)

# Store the performance metrics of the logistic regression model in a data frame
logistic_model_metrics <- data.frame(
  dataset = "Logistic Testing",
  accuracy = accuracy_of_logistic_model,
  log_loss = logarithmic_loss_logistic,
  calibration = calibration_metric_logistic
)

# Append the logistic regression model's metrics to the performance metrics data frame
performance_metrics <- rbind(performance_metrics, logistic_model_metrics)

# Print the updated performance metrics
print(performance_metrics)

# Create a data frame for displaying the logistic regression model's metrics
logistic_model_metrics_display <- data.frame(
  dataset = "Logistic Testing",
  accuracy = accuracy_of_logistic_model,
  log_loss = logarithmic_loss_logistic,
  calibration = calibration_metric_logistic
)

# Use the kable function to format and print the table of logistic model performance metrics
print(
  kable(logistic_model_metrics_display, caption = "Logistic Model Performance Metrics for logistic model") %>%
    kable_styling(
      bootstrap_options = c("striped", "hover", "condensed", "responsive"),
      full_width = FALSE,
      position = "center"
    )
)


















# elo k-factor model : function to calculate the expected Elo score
calculate_expected_elo <- function(rating1, rating2) {
  # Calculate expected score using the Elo formula
  return(1 / (1 + 10^((rating2 - rating1) / 400)))
}

# Function to update Elo ratings after a match
update_ratings <- function(rating_winner, rating_loser, k_factor = 25) {
  # Calculate the expected score for the winner
  expected_win <- calculate_expected_elo(rating_winner, rating_loser)
  # Update the winner's rating
  updated_winner_rating <- rating_winner + k_factor * (1 - expected_win)
  # Update the loser's rating
  updated_loser_rating <- rating_loser + k_factor * (0 - (1 - expected_win))
  # Return the new ratings
  return(c(updated_winner_rating, updated_loser_rating))
}

# Initialize Elo ratings for all players
unique_players <- unique(c(train_set$Winner, train_set$Loser, validation_set$Winner, validation_set$Loser, test_set$Winner, test_set$Loser))
# Assign an initial rating of 1500 to each player
initial_elo_ratings <- setNames(rep(1500, length(unique_players)), unique_players)

# Update Elo ratings based on the training data
for (match_index in 1:nrow(train_set)) {
  # Get the winner and loser of the match
  match_winner <- train_set$Winner[match_index]
  match_loser <- train_set$Loser[match_index]
  # Get the current ratings of the winner and loser
  rating_winner <- initial_elo_ratings[as.character(match_winner)]
  rating_loser <- initial_elo_ratings[as.character(match_loser)]
  # Update the ratings based on the match outcome
  new_ratings <- update_ratings(rating_winner, rating_loser)
  # Assign the new ratings back to the players
  initial_elo_ratings[as.character(match_winner)] <- new_ratings[1]
  initial_elo_ratings[as.character(match_loser)] <- new_ratings[2]
}

# Function to calculate expected probabilities using Elo ratings
calculate_probabilities <- function(dataset, elo_ratings) {
  probabilities <- numeric(nrow(dataset))
  for (index in 1:nrow(dataset)) {
    # Get the winner and loser of the match
    player_winner <- dataset$Winner[index]
    player_loser <- dataset$Loser[index]
    # Get the current ratings of the winner and loser
    rating_winner <- elo_ratings[as.character(player_winner)]
    rating_loser <- elo_ratings[as.character(player_loser)]
    # Calculate the probability of the higher-ranked player winning
    probability <- calculate_expected_elo(rating_winner, rating_loser)
    # Adjust probability based on actual outcome
    probabilities[index] <- ifelse(dataset$higher_rank_won[index], probability, 1 - probability)
  }
  return(probabilities)
}

# Calculate probabilities for train, validation, and test sets
probabilities_train_elo <- calculate_probabilities(train_set, initial_elo_ratings)
probabilities_validation_elo <- calculate_probabilities(validation_set, initial_elo_ratings)
probabilities_test_elo <- calculate_probabilities(test_set, initial_elo_ratings)

# Function to calculate performance metrics for the Elo model
calculate_performance_metrics <- function(probabilities, actual_results) {
  # Generate binary predictions based on probabilities
  predicted_outcomes <- ifelse(probabilities > 0.5, 1, 0)
  # Calculate accuracy of the model
  accuracy_metric <- mean(predicted_outcomes == actual_results)
  # Calculate log loss of the model
  log_loss_metric <- -mean(actual_results * log(probabilities) + (1 - actual_results) * log(1 - probabilities), na.rm = TRUE)
  # Calculate calibration metric
  calibration_metric <- sum(probabilities) / sum(actual_results)
  return(list(accuracy = accuracy_metric, log_loss = log_loss_metric, calibration = calibration_metric))
}

# Evaluate Elo model on train, validation, and test sets
metrics_train_elo <- calculate_performance_metrics(probabilities_train_elo, train_set$higher_rank_won)
metrics_validation_elo <- calculate_performance_metrics(probabilities_validation_elo, validation_set$higher_rank_won)
metrics_test_elo <- calculate_performance_metrics(probabilities_test_elo, test_set$higher_rank_won)

# Compile Elo model metrics into a data frame
compiled_elo_metrics <- data.frame(
  dataset_type = c("ELO k-factor Training", "ELO k-factor Validation", "ELO k-factor Testing"),
  accuracy_value = c(metrics_train_elo$accuracy, metrics_validation_elo$accuracy, metrics_test_elo$accuracy),
  log_loss_value = c(metrics_train_elo$log_loss, metrics_validation_elo$log_loss, metrics_test_elo$log_loss),
  calibration_value = c(metrics_train_elo$calibration, metrics_validation_elo$calibration, metrics_test_elo$calibration)
)

# Print Elo model performance metrics using kable
print(
  kable(compiled_elo_metrics, caption = "ELO Model Performance Metrics for elo k-factor") %>%
    kable_styling(
      bootstrap_options = c("striped", "hover", "condensed", "responsive"),  # Apply table styling
      full_width = FALSE,  # Set table width to not full width
      position = "center"  # Center align the table
    )
)















# tune parameters for elo k-factor, get the list of all players
all_players <- unique(c(train_set$Winner, train_set$Loser, validation_set$Winner, validation_set$Loser, test_set$Winner, test_set$Loser))

# Store performance metrics for each k-value
results <- data.frame(k_value = numeric(), 
                      train_accuracy = numeric(), 
                      train_log_loss = numeric(), 
                      train_calibration = numeric(),
                      test_accuracy = numeric(), 
                      test_log_loss = numeric(), 
                      test_calibration = numeric())

# Define the range of k-values
k_values <- seq(10, 50, by = 5)

# Evaluate performance for each k-value
for (k in k_values) {
  # Initialize Elo ratings
  elo_ratings <- setNames(rep(1500, length(all_players)), all_players)
  
  # Update Elo ratings based on training data
  for (i in 1:nrow(train_set)) {
    winner <- train_set$Winner[i]
    loser <- train_set$Loser[i]
    winner_rating <- elo_ratings[as.character(winner)]
    loser_rating <- elo_ratings[as.character(loser)]
    updated_ratings <- update_elo_ratings(winner_rating, loser_rating, k)
    elo_ratings[as.character(winner)] <- updated_ratings[1]
    elo_ratings[as.character(loser)] <- updated_ratings[2]
  }
  
  # Calculate expected probabilities for training data
  train_probs_elo <- calculate_elo_probs(train_set, elo_ratings)
  # Calculate expected probabilities for test data
  test_probs_elo <- calculate_elo_probs(test_set, elo_ratings)
  
  # Calculate performance metrics
  train_metrics <- calculate_metrics(train_probs_elo, train_set$higher_rank_won)
  test_metrics <- calculate_metrics(test_probs_elo, test_set$higher_rank_won)
  
  # Store results in the data frame
  results <- rbind(results, data.frame(
    k_value = k, 
    train_accuracy = train_metrics$accuracy, 
    train_log_loss = train_metrics$log_loss, 
    train_calibration = train_metrics$calibration,
    test_accuracy = test_metrics$accuracy, 
    test_log_loss = test_metrics$log_loss, 
    test_calibration = test_metrics$calibration
  ))
}

# Print results in a table format
print(results)





# Load necessary libraries
library(dplyr)

# Define the best k-value based on analysis
best_k <- 10

# Initialize Elo ratings
elo_ratings <- setNames(rep(1500, length(all_players)), all_players)

# Update Elo ratings based on training data with the best k-value
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  updated_ratings <- update_elo_ratings(winner_rating, loser_rating, best_k)
  elo_ratings[as.character(winner)] <- updated_ratings[1]
  elo_ratings[as.character(loser)] <- updated_ratings[2]
}

# Calculate expected probabilities for test data with the best k-value
final_test_probs_elo <- calculate_elo_probs(test_set, elo_ratings)

# Calculate final performance metrics
final_elo_test_metrics <- calculate_metrics(final_test_probs_elo, test_set$higher_rank_won)

# Compile final Elo model metrics
final_elo_model_metrics <- data.frame(
  dataset = "Final ELO Testing",
  accuracy = final_elo_test_metrics$accuracy,
  log_loss = final_elo_test_metrics$log_loss,
  calibration = final_elo_test_metrics$calibration
)

# Print the final Elo model metrics
print(final_elo_model_metrics)























# elo 538 model
# firstly, we define the initial parameters for the elo 538 model
delta <- 100
nu <- 5
sigma <- 0.1

# Initialize Elo ratings and match counts for all players
elo_ratings_538 <- setNames(rep(1500, length(all_players)), all_players)
match_counts_538 <- setNames(rep(0, length(all_players)), all_players)

# Function to calculate the K-factor based on the number of matches played
compute_k_factor <- function(num_matches) {
  return(delta * (num_matches + nu) * sigma^2)
}

# Function to update Elo ratings using the Elo 538 model
update_elo_538 <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
  # Calculate the expected score for the winner
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  # Calculate the K-factor for the winner and loser
  k_winner <- compute_k_factor(winner_matches)
  k_loser <- compute_k_factor(loser_matches)
  # Update the ratings based on the match outcome
  new_winner_rating <- winner_rating + k_winner * (1 - expected_winner)
  new_loser_rating <- loser_rating + k_loser * (0 - (1 - expected_winner))
  # Increment the match counts for the winner and loser
  new_winner_matches <- winner_matches + 1
  new_loser_matches <- loser_matches + 1
  return(c(new_winner_rating, new_loser_rating, new_winner_matches, new_loser_matches))
}

# Update Elo 538 ratings based on the training data
for (i in 1:nrow(train_set)) {
  # Get the winner and loser of the match
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  # Get the current ratings and match counts for the winner and loser
  winner_rating <- elo_ratings_538[as.character(winner)]
  loser_rating <- elo_ratings_538[as.character(loser)]
  winner_matches <- match_counts_538[as.character(winner)]
  loser_matches <- match_counts_538[as.character(loser)]
  # Update the ratings and match counts
  updated_values <- update_elo_538(winner_rating, loser_rating, winner_matches, loser_matches)
  elo_ratings_538[as.character(winner)] <- updated_values[1]
  elo_ratings_538[as.character(loser)] <- updated_values[2]
  match_counts_538[as.character(winner)] <- updated_values[3]
  match_counts_538[as.character(loser)] <- updated_values[4]
}

# Function to calculate expected probabilities using Elo 538 ratings
calculate_elo_probs_538 <- function(data, ratings) {
  probabilities <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    # Get the winner and loser of the match
    winner <- data$Winner[i]
    loser <- data$Loser[i]
    # Get the current ratings for the winner and loser
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    # Calculate the probability of the higher-ranked player winning
    probability <- expected_elo_score(winner_rating, loser_rating)
    # Adjust probability based on actual outcome
    probabilities[i] <- ifelse(data$higher_rank_won[i], probability, 1 - probability)
  }
  return(probabilities)
}

# Calculate probabilities for train, validation, and test sets using Elo 538 model
train_probs_elo_538 <- calculate_elo_probs_538(train_set, elo_ratings_538)
validation_probs_elo_538 <- calculate_elo_probs_538(validation_set, elo_ratings_538)
test_probs_elo_538 <- calculate_elo_probs_538(test_set, elo_ratings_538)

# Evaluate Elo 538 model on train, validation, and test sets
elo_538_train_metrics <- calculate_metrics(train_probs_elo_538, train_set$higher_rank_won)
elo_538_validation_metrics <- calculate_metrics(validation_probs_elo_538, validation_set$higher_rank_won)
elo_538_test_metrics <- calculate_metrics(test_probs_elo_538, test_set$higher_rank_won)

# Compile Elo 538 model metrics into a data frame
elo_538_metrics <- data.frame(
  dataset = c("ELO 538 Training", "ELO 538 Validation", "ELO 538 Testing"),
  accuracy = c(elo_538_train_metrics$accuracy, elo_538_validation_metrics$accuracy, elo_538_test_metrics$accuracy),
  log_loss = c(elo_538_train_metrics$log_loss, elo_538_validation_metrics$log_loss, elo_538_test_metrics$log_loss),
  calibration = c(elo_538_train_metrics$calibration, elo_538_validation_metrics$calibration, elo_538_test_metrics$calibration)
)

# Print Elo 538 model performance metrics using kable
print(
  kable(elo_538_metrics, caption = "538 ELO Model Performance Metrics for elo 538") %>%
    kable_styling(
      bootstrap_options = c("striped", "hover", "condensed", "responsive"),
      full_width = FALSE,
      position = "center"
    )
)













# tune parameter for elo 538
# Define the range of values for each parameter
delta_values <- seq(50, 200, by = 50)
nu_values <- seq(1, 10, by = 2)
sigma_values <- seq(0.05, 0.2, by = 0.05)

# Store performance metrics for each combination of parameters
results_538 <- data.frame(delta = numeric(), nu = numeric(), sigma = numeric(), 
                          train_accuracy = numeric(), train_log_loss = numeric(), train_calibration = numeric(),
                          validation_accuracy = numeric(), validation_log_loss = numeric(), validation_calibration = numeric(),
                          test_accuracy = numeric(), test_log_loss = numeric(), test_calibration = numeric())

# Evaluate performance for each combination of parameters
for (delta in delta_values) {
  for (nu in nu_values) {
    for (sigma in sigma_values) {
      
      # Initialize Elo ratings and match counts
      elo_ratings_538 <- setNames(rep(1500, length(all_players)), all_players)
      match_counts <- setNames(rep(0, length(all_players)), all_players)
      
      # Update advanced Elo ratings based on training data
      for (i in 1:nrow(train_set)) {
        winner <- train_set$Winner[i]
        loser <- train_set$Loser[i]
        winner_rating <- elo_ratings_538[as.character(winner)]
        loser_rating <- elo_ratings_538[as.character(loser)]
        winner_matches <- match_counts[as.character(winner)]
        loser_matches <- match_counts[as.character(loser)]
        updated_ratings <- update_elo_538(winner_rating, loser_rating, winner_matches, loser_matches)
        elo_ratings_538[as.character(winner)] <- updated_ratings[1]
        elo_ratings_538[as.character(loser)] <- updated_ratings[2]
        match_counts[as.character(winner)] <- updated_ratings[3]
        match_counts[as.character(loser)] <- updated_ratings[4]
      }
      
      # Calculate probabilities for train, validation, and test sets using advanced Elo model
      train_probs_elo_538 <- calculate_elo_probs_538(train_set, elo_ratings_538)
      validation_probs_elo_538 <- calculate_elo_probs_538(validation_set, elo_ratings_538)
      test_probs_elo_538 <- calculate_elo_probs_538(test_set, elo_ratings_538)
      
      # Evaluate advanced Elo model on train, validation, and test sets
      elo_538_train_metrics <- calculate_metrics(train_probs_elo_538, train_set$higher_rank_won)
      elo_538_validation_metrics <- calculate_metrics(validation_probs_elo_538, validation_set$higher_rank_won)
      elo_538_test_metrics <- calculate_metrics(test_probs_elo_538, test_set$higher_rank_won)
      
      # Store results in the data frame
      results_538 <- rbind(results_538, data.frame(
        delta = delta, nu = nu, sigma = sigma, 
        train_accuracy = elo_538_train_metrics$accuracy, 
        train_log_loss = elo_538_train_metrics$log_loss, 
        train_calibration = elo_538_train_metrics$calibration,
        validation_accuracy = elo_538_validation_metrics$accuracy, 
        validation_log_loss = elo_538_validation_metrics$log_loss, 
        validation_calibration = elo_538_validation_metrics$calibration,
        test_accuracy = elo_538_test_metrics$accuracy, 
        test_log_loss = elo_538_test_metrics$log_loss, 
        test_calibration = elo_538_test_metrics$calibration
      ))
    }
  }
}

# Find the best combination of parameters (based on validation log loss)
best_params <- results_538[which.min(results_538$validation_log_loss), ]

# Print the best combination of parameters
print(paste("The best combination of parameters is: delta =", best_params$delta, 
            ", nu =", best_params$nu, ", sigma =", best_params$sigma))

#save all results to a CSV file for further analysis
write.csv(results_538, "results_538.csv", row.names = FALSE)










# top 50 and top 100 for elo k-factor model
# Define function to update Elo ratings with k = 10
update_elo_10 <- function(winner_rating, loser_rating, winner_matches, loser_matches) {
  expected_winner <- expected_elo_score(winner_rating, loser_rating)
  k_factor <- 10
  new_winner_rating <- winner_rating + k_factor * (1 - expected_winner)
  new_loser_rating <- loser_rating + k_factor * (0 - (1 - expected_winner))
  return(c(new_winner_rating, new_loser_rating, winner_matches + 1, loser_matches + 1))
}

# Function to evaluate k = 10 Elo model on a subset of data
evaluate_elo_10 <- function(train_data, validation_data, test_data, update_func, prob_func) {
  # Initialize Elo ratings and match counts
  ratings <- setNames(rep(1500, length(all_players)), all_players)
  match_counts <- setNames(rep(0, length(all_players)), all_players)
  
  # Update Elo ratings based on training data
  for (i in 1:nrow(train_data)) {
    winner <- train_data$Winner[i]
    loser <- train_data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    winner_matches <- match_counts[as.character(winner)]
    loser_matches <- match_counts[as.character(loser)]
    updated_ratings <- update_func(winner_rating, loser_rating, winner_matches, loser_matches)
    ratings[as.character(winner)] <- updated_ratings[1]
    ratings[as.character(loser)] <- updated_ratings[2]
    match_counts[as.character(winner)] <- updated_ratings[3]
    match_counts[as.character(loser)] <- updated_ratings[4]
  }
  
  # Calculate probabilities for train, validation, and test sets
  train_probs <- prob_func(train_data, ratings)
  validation_probs <- prob_func(validation_data, ratings)
  test_probs <- prob_func(test_data, ratings)
  
  # Evaluate performance
  train_metrics <- calculate_metrics(train_probs, train_data$higher_rank_won)
  validation_metrics <- calculate_metrics(validation_probs, validation_data$higher_rank_won)
  test_metrics <- calculate_metrics(test_probs, test_data$higher_rank_won)
  
  return(list(train = train_metrics, validation = validation_metrics, test = test_metrics))
}

# Get top 50 and top 100 players based on current Elo ratings
top_50_players <- get_top_n_players(elo_ratings, 50)
top_100_players <- get_top_n_players(elo_ratings, 100)

# Filter matches involving top 50 and top 100 players
train_set_top_50 <- filter_matches_by_top_players(train_set, top_50_players)
validation_set_top_50 <- filter_matches_by_top_players(validation_set, top_50_players)
test_set_top_50 <- filter_matches_by_top_players(test_set, top_50_players)

train_set_top_100 <- filter_matches_by_top_players(train_set, top_100_players)
validation_set_top_100 <- filter_matches_by_top_players(validation_set, top_100_players)
test_set_top_100 <- filter_matches_by_top_players(test_set, top_100_players)

# Evaluate the k = 10 Elo model on top 50 and top 100 players
elo_10_metrics_top_50 <- evaluate_elo_10(train_set_top_50, validation_set_top_50, test_set_top_50, update_elo_10, calculate_elo_probs_538)
elo_10_metrics_top_100 <- evaluate_elo_10(train_set_top_100, validation_set_top_100, test_set_top_100, update_elo_10, calculate_elo_probs_538)

# Compile results for top 50 players
results_elo_10_top_50 <- data.frame(
  dataset = c("ELO 10 Top 50 Training", "ELO 10 Top 50 Validation", "ELO 10 Top 50 Testing"),
  accuracy = c(elo_10_metrics_top_50$train$accuracy, elo_10_metrics_top_50$validation$accuracy, elo_10_metrics_top_50$test$accuracy),
  log_loss = c(elo_10_metrics_top_50$train$log_loss, elo_10_metrics_top_50$validation$log_loss, elo_10_metrics_top_50$test$log_loss),
  calibration = c(elo_10_metrics_top_50$train$calibration, elo_10_metrics_top_50$validation$calibration, elo_10_metrics_top_50$test$calibration)
)

# Compile results for top 100 players
results_elo_10_top_100 <- data.frame(
  dataset = c("ELO 10 Top 100 Training", "ELO 10 Top 100 Validation", "ELO 10 Top 100 Testing"),
  accuracy = c(elo_10_metrics_top_100$train$accuracy, elo_10_metrics_top_100$validation$accuracy, elo_10_metrics_top_100$test$accuracy),
  log_loss = c(elo_10_metrics_top_100$train$log_loss, elo_10_metrics_top_100$validation$log_loss, elo_10_metrics_top_100$test$log_loss),
  calibration = c(elo_10_metrics_top_100$train$calibration, elo_10_metrics_top_100$validation$calibration, elo_10_metrics_top_100$test$calibration)
)

# Print the results for top 50 players
print("Performance metrics for ELO 10 Top 50 players:")
print(results_elo_10_top_50)

# Print the results for top 100 players
print("Performance metrics for ELO 10 Top 100 players:")
print(results_elo_10_top_100)






















# top 50 and top 100 for elo 538 model
# Function to get top N players based on Elo ratings at a given time
get_top_n_players <- function(ratings, n) {
  top_players <- names(sort(ratings, decreasing = TRUE)[1:n])
  return(top_players)
}

# Function to filter matches involving top N players
filter_matches_by_top_players <- function(data, top_players) {
  filtered_data <- data %>%
    filter(Winner %in% top_players | Loser %in% top_players)
  return(filtered_data)
}

# Initialize Elo ratings and match counts
elo_ratings <- setNames(rep(1500, length(all_players)), all_players)
match_counts <- setNames(rep(0, length(all_players)), all_players)

# Update Elo ratings based on the initial training data
for (i in 1:nrow(train_set)) {
  winner <- train_set$Winner[i]
  loser <- train_set$Loser[i]
  winner_rating <- elo_ratings[as.character(winner)]
  loser_rating <- elo_ratings[as.character(loser)]
  winner_matches <- match_counts[as.character(winner)]
  loser_matches <- match_counts[as.character(loser)]
  updated_ratings <- update_elo_538_best(winner_rating, loser_rating, winner_matches, loser_matches)
  elo_ratings[as.character(winner)] <- updated_ratings[1]
  elo_ratings[as.character(loser)] <- updated_ratings[2]
  match_counts[as.character(winner)] <- updated_ratings[3]
  match_counts[as.character(loser)] <- updated_ratings[4]
}

# Get top 50 and top 100 players based on current Elo ratings
top_50_players <- get_top_n_players(elo_ratings, 50)
top_100_players <- get_top_n_players(elo_ratings, 100)

# Filter matches involving top 50 and top 100 players
train_set_top_50 <- filter_matches_by_top_players(train_set, top_50_players)
validation_set_top_50 <- filter_matches_by_top_players(validation_set, top_50_players)
test_set_top_50 <- filter_matches_by_top_players(test_set, top_50_players)

train_set_top_100 <- filter_matches_by_top_players(train_set, top_100_players)
validation_set_top_100 <- filter_matches_by_top_players(validation_set, top_100_players)
test_set_top_100 <- filter_matches_by_top_players(test_set, top_100_players)

# Function to evaluate algorithm on a subset of data
evaluate_algorithm <- function(train_data, validation_data, test_data, ratings, update_func, prob_func) {
  # Update Elo ratings based on training data
  for (i in 1:nrow(train_data)) {
    winner <- train_data$Winner[i]
    loser <- train_data$Loser[i]
    winner_rating <- ratings[as.character(winner)]
    loser_rating <- ratings[as.character(loser)]
    winner_matches <- match_counts[as.character(winner)]
    loser_matches <- match_counts[as.character(loser)]
    updated_ratings <- update_func(winner_rating, loser_rating, winner_matches, loser_matches)
    ratings[as.character(winner)] <- updated_ratings[1]
    ratings[as.character(loser)] <- updated_ratings[2]
    match_counts[as.character(winner)] <- updated_ratings[3]
    match_counts[as.character(loser)] <- updated_ratings[4]
  }
  
  # Calculate probabilities for train, validation, and test sets
  train_probs <- prob_func(train_data, ratings)
  validation_probs <- prob_func(validation_data, ratings)
  test_probs <- prob_func(test_data, ratings)
  
  # Evaluate performance
  train_metrics <- calculate_metrics(train_probs, train_data$higher_rank_won)
  validation_metrics <- calculate_metrics(validation_probs, validation_data$higher_rank_won)
  test_metrics <- calculate_metrics(test_probs, test_data$higher_rank_won)
  
  return(list(train = train_metrics, validation = validation_metrics, test = test_metrics))
}

# Evaluate the advanced Elo model on top 50 and top 100 players
elo_538_metrics_top_50 <- evaluate_algorithm(train_set_top_50, validation_set_top_50, test_set_top_50, elo_ratings, update_elo_538_best, calculate_elo_probs_538)
elo_538_metrics_top_100 <- evaluate_algorithm(train_set_top_100, validation_set_top_100, test_set_top_100, elo_ratings, update_elo_538_best, calculate_elo_probs_538)

# Compile results for top 50 players
results_top_50 <- data.frame(
  dataset = c("ELO 538 Top 50 Training", "ELO 538 Top 50 Validation", "ELO 538 Top 50 Testing"),
  accuracy = c(elo_538_metrics_top_50$train$accuracy, elo_538_metrics_top_50$validation$accuracy, elo_538_metrics_top_50$test$accuracy),
  log_loss = c(elo_538_metrics_top_50$train$log_loss, elo_538_metrics_top_50$validation$log_loss, elo_538_metrics_top_50$test$log_loss),
  calibration = c(elo_538_metrics_top_50$train$calibration, elo_538_metrics_top_50$validation$calibration, elo_538_metrics_top_50$test$calibration)
)

# Compile results for top 100 players
results_top_100 <- data.frame(
  dataset = c("ELO 538 Top 100 Training", "ELO 538 Top 100 Validation", "ELO 538 Top 100 Testing"),
  accuracy = c(elo_538_metrics_top_100$train$accuracy, elo_538_metrics_top_100$validation$accuracy, elo_538_metrics_top_100$test$accuracy),
  log_loss = c(elo_538_metrics_top_100$train$log_loss, elo_538_metrics_top_100$validation$log_loss, elo_538_metrics_top_100$test$log_loss),
  calibration = c(elo_538_metrics_top_100$train$calibration, elo_538_metrics_top_100$validation$calibration, elo_538_metrics_top_100$test$calibration)
)

# Print the results for top 50 players
print("Performance metrics for ELO 538 Top 50 players:")
print(results_top_50)

# Print the results for top 100 players
print("Performance metrics for ELO 538 Top 100 players:")
print(results_top_100)


































