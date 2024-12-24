# ==========================
# E-COMMERCE CUSTOMER ANALYSIS
# ==========================

# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)
library(randomForest)
library(cluster)
library(ggcorrplot)

# ==========================
# STEP 1: DATA SIMULATION
# ==========================
set.seed(123)  # Set seed for reproducibility

# Simulate customer data
num_customers <- 1000
customer_data <- data.frame(
  CustomerID = 1:num_customers,
  Age = sample(18:70, num_customers, replace = TRUE),
  Gender = sample(c("Male", "Female"), num_customers, replace = TRUE, prob = c(0.45, 0.55)),
  AnnualIncome = round(rnorm(num_customers, mean = 60000, sd = 15000), 0),
  SpendingScore = round(runif(num_customers, min = 1, max = 100), 0),
  Tenure = sample(1:10, num_customers, replace = TRUE),
  PurchaseFrequency = sample(1:50, num_customers, replace = TRUE),
  Returns = sample(0:10, num_customers, replace = TRUE),
  ProductCategory = sample(c("Electronics", "Clothing", "Home", "Books", "Beauty"), num_customers, replace = TRUE),
  IsPremiumMember = sample(c(0, 1), num_customers, replace = TRUE, prob = c(0.7, 0.3))
)

# ==========================
# STEP 2: DATA CLEANING
# ==========================
# Check for missing values
missing_values <- colSums(is.na(customer_data))
print("Missing values in each column:")
print(missing_values)

# Remove any negative or unrealistic values (if any)
customer_data <- customer_data %>%
  mutate(
    AnnualIncome = ifelse(AnnualIncome < 0, NA, AnnualIncome),
    SpendingScore = ifelse(SpendingScore < 0 | SpendingScore > 100, NA, SpendingScore)
  ) %>%
  drop_na()

# ==========================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================
# Summary Statistics
print("Summary of customer data:")
print(summary(customer_data))

# Gender distribution
gender_plot <- ggplot(customer_data, aes(x = Gender)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Gender Distribution") +
  theme_minimal()
print(gender_plot)

# Age vs Spending Score
age_spending_plot <- ggplot(customer_data, aes(x = Age, y = SpendingScore, color = Gender)) +
  geom_point(alpha = 0.7) +
  ggtitle("Age vs Spending Score") +
  theme_minimal()
print(age_spending_plot)

# Correlation Matrix
numeric_data <- customer_data %>%
  select(Age, AnnualIncome, SpendingScore, Tenure, PurchaseFrequency, Returns)
cor_matrix <- cor(numeric_data)
print("Correlation matrix:")
print(cor_matrix)
cor_plot <- ggcorrplot(cor_matrix, lab = TRUE)
print(cor_plot)

# ==========================
# STEP 4: CLUSTERING
# ==========================
# Scale numeric variables
scaled_data <- scale(numeric_data)

# K-means clustering
set.seed(123)
kmeans_model <- kmeans(scaled_data, centers = 4)
customer_data$Cluster <- as.factor(kmeans_model$cluster)

# Visualize clusters
cluster_plot <- ggplot(customer_data, aes(x = AnnualIncome, y = SpendingScore, color = Cluster)) +
  geom_point(alpha = 0.7) +
  ggtitle("Customer Clusters") +
  theme_minimal()
print(cluster_plot)

# ==========================
# STEP 5: MACHINE LEARNING MODEL (CLASSIFICATION)
# ==========================
# Task: Predict if a customer is a premium member based on their features

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(customer_data$IsPremiumMember, p = 0.8, list = FALSE)
train_data <- customer_data[train_index, ]
test_data <- customer_data[-train_index, ]

# Train a Random Forest model
rf_model <- randomForest(
  IsPremiumMember ~ Age + AnnualIncome + SpendingScore + Tenure + PurchaseFrequency + Returns,
  data = train_data,
  ntree = 100
)
print("Random Forest Model Summary:")
print(rf_model)

# Predict on test data
rf_predictions <- predict(rf_model, test_data)

# Evaluate model performance
conf_matrix <- confusionMatrix(as.factor(rf_predictions), as.factor(test_data$IsPremiumMember))
print("Confusion Matrix:")
print(conf_matrix)

# ==========================
# STEP 6: VISUALIZATION OF RESULTS
# ==========================
# Feature importance
importance <- importance(rf_model)
var_importance <- data.frame(Feature = rownames(importance), Importance = importance[, 1])
var_importance_plot <- ggplot(var_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  ggtitle("Feature Importance") +
  theme_minimal()
print(var_importance_plot)

# ==========================
# STEP 7: SAVING THE DATA AND MODEL
# ==========================
# Save the clean data to a CSV file
write.csv(customer_data, "clean_customer_data.csv", row.names = FALSE)

# Save the trained model
saveRDS(rf_model, "rf_model.rds")

# ==========================
# END OF PROGRAM
# ==========================
print("Program completed successfully!")
