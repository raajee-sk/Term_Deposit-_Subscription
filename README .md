# Term Deposit Subscription Prediction

The Term Deposit Subscription Prediction project aims to analyze a bank's marketing campaign data to predict customer subscriptions to a term deposit. By developing a machine learning model, the goal is to accurately identify potential customers who are likely to subscribe to the term deposit, enabling targeted marketing efforts.

## Data Exploration and Preprocessing

In this phase, we perform an initial exploration of the dataset and preprocess the data to prepare it for model training. The following steps are performed:

1. Importing libraries and setting up the notebook: We import the necessary libraries for data analysis and set up the Jupyter notebook.

2. Reading data and knowing the basics: We load the dataset and examine its shape and basic information using the Pandas library.

3. Checking for duplicated rows and null values: We check if there are any duplicated rows or null values in the dataset and handle them accordingly.

4. Exploring unique values: We explore the unique values in each column of the dataset to gain insights into the data distribution and identify any abnormalities.

5. Handling numerical and categorical data separately: We separate the dataset into numerical and categorical variables for further analysis and preprocessing.

6. Handling discrete variables: We analyze and visualize the discrete variables in the dataset, such as age and contact frequency, to understand their distributions and relationships with the target variable.

7. Handling continuous variables: We examine and visualize the continuous variables in the dataset, such as last contact duration and consumer confidence index, and apply outlier treatment to ensure the data is suitable for modeling.

8. Handling categorical variables: We preprocess the categorical variables in the dataset, including handling unknown values and encoding them for model compatibility.

## Model Building and Evaluation
1. Train-Validation-Test Split:
   - Splitting the dataset into train, validation, and test sets

2. Scaling the Data:
   - Standard scaling of numerical columns to ensure consistency

3. Handling Class Imbalance:
   - Checking the class distribution in the training set
   - Addressing class imbalance using Synthetic Minority Over-sampling Technique (SMOTE)

4. Model :
   Given  the dataset,use SVM Classifier for Term Deposit Subscription Prediction 

5. Hyperparameter Tuning:
   - Using GridSearchCV to find the best hyperparameters for the SVM Classifier

## Model Performance on Test Set
1. Evaluation Metrics:
   - Calculating accuracy, precision, recall, and F1-score on the test set
   - Focusing on positive outcomes (customer subscriptions) for evaluation

2. Results Interpretation:
   - Analyzing the performance metrics to assess the model's ability to predict customer subscriptions
   

## Results and Metrics

After training and evaluating the models, we analyze the results and metrics obtained from the best performing model. The following metrics are considered:

- Accuracy Score: The accuracy of the model in predicting the term deposit subscription.

- Confusion Matrix: A matrix showing the number of true positives, true negatives, false positives, and false negatives.

- Classification Report: A report showing precision, recall, F1 score, and support for each class.

- ROC AUC Score: The area under the receiver operating characteristic curve, which indicates the model's ability to distinguish between positive and negative classes.

- Average Precision Score: The average precision achieved across all recall levels, which provides a single-value summary of the precision-recall curve.

    We achieved,
  
          Accuracy ≃ 89%
          ROC-AUC Score ≃ 0.82
          False Negatives ≃ 2.8%
          False Positives ≃ 8%



## Conclusion
This project showcases the application of machine learning techniques to analyze a bank's marketing campaign data and predict customer behavior. The trained Gradient Boosting Classifier shows good performance on the test set, but further optimization can be done to improve its precision and recall for positive outcomes. The insights gained from this analysis can help the bank in targeting potential customers more effectively and improving the success rate of their marketing campaigns.
