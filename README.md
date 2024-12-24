# E-Commerce Customer Analysis Program

## **Overview**

This program simulates an e-commerce customer dataset and performs a comprehensive analysis, including:
- Data cleaning
- Exploratory Data Analysis (EDA)
- Customer segmentation using clustering
- Predictive modeling using machine learning
- Visualization of results
- Saving clean data and models for future use

The program is structured into several steps, with each section focusing on a specific task.

## **Requirements**

Before running the program, make sure the following R libraries are installed:

```r
install.packages(c("ggplot2", "dplyr", "tidyr", "caret", "randomForest", "cluster", "ggcorrplot"))
```

## How to Run the Program

1. Copy the entire script into an R script file (e.g., `customer_analysis.R`).
2. Install the required libraries if not already installed.
3. Run the script in RStudio or another R environment.
4. View the outputs in the R console and plots in the plotting window.


## Extending the Program

Here are some ways to enhance the program:
1. **Add more features**:
   - Include additional customer attributes like geographic location or channel preferences.
2. **Optimize Clustering**:
   - Use the Elbow Method or Silhouette Analysis to find the optimal number of clusters.
3. **Hyperparameter Tuning**:
   - Use `caret` or `gridSearch` to optimize the Random Forest model.
4. **Add Additional Models**:
   - Compare Random Forest with other models like Logistic Regression, SVM, or Gradient Boosting.
5. **Create a Dashboard**:
   - Use `shiny` to create an interactive dashboard for visualizing customer insights.
