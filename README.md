# End-to-end Multi-class Crime Text Classification using Pyspark

Apache Spark's ability to process streaming data efficiently makes it an ideal choice for developing our Crime Classification System. Leveraging Spark's Machine Learning Library (MLlib) and its Pipelines API, we will build a robust end-to-end solution for multi-class text classification.

## 1. Problem

Law enforcement agencies often face challenges in efficiently categorizing crimes based on descriptions, which can be time-consuming and prone to errors. To address this issue and improve resource allocation, we aim to develop an automated Crime Classification System. This system will accurately classify crime descriptions into predefined categories using machine learning techniques tailored to multi-class text classification.

## 2. Data

We will utilize a dataset sourced from Kaggle on San Francisco Crime, containing information such as crime descriptions, dates, day of the week, police district, and more. The dataset will serve as the foundation for training and testing our Crime Classification System.

https://www.kaggle.com/c/sf-crime/data

## 3. Features

The dataset used for developing the Crime Classification System contains the following features:

* Dates: Date and time of the crime incident.
* Category: Predefined category of the crime. This is our **target variable** and is only available in the training set.
* Descript: Description of the crime incident.
* DayOfWeek: Day of the week when the crime occurred.
* PdDistrict: Police district where the crime occurred.
* Resolution: How the crime incident was resolved.
* Address: Address or location of the crime incident.
* X: Longitude coordinate of the crime location.
* Y: Latitude coordinate of the crime location.

These features provide essential information about each crime incident, including its timing, location, category, and description. By analyzing these features, we can train machine learning models to accurately classify crime descriptions into predefined categories, aiding law enforcement agencies in resource allocation and investigation prioritization.

## Data Exploration (Exploratory Data Analysis or EDA)
The objective of this EDA is to gain insights into the San Francisco crime dataset, including the nature and frequency of different types of crimes. By analyzing the data, we aim to identify trends and patterns that can inform further analysis and modeling efforts.

* **Counting Occurrences:** We start by counting the occurrences of each crime category and description in the dataset. This helps us understand the frequency distribution of different types of crimes recorded in San Francisco.

* **Analysis of Categories:** We analyze the distribution of crime categories by counting the occurrences of each category. This step provides insights into which types of crimes are most prevalent in the dataset.

* **Analysis of Descriptions:** Similarly, we analyze the distribution of crime descriptions by counting their occurrences. This allows us to identify the most common descriptions associated with reported crimes.

* **Visualizations:** We use visualizations such as countplots to illustrate the distribution of crime categories and the top 20 crime descriptions. These visual representations make it easier to interpret and understand the patterns in the data.

## Model Pipeline:
Our model pipeline consists of several steps:

* **regexTokenizer:** Tokenization using Regular Expression.
* **stopwordsRemover:** Removing stop words.
* **countVectors:** Generating count vectors for text data.
* **StringIndexer:** Encoding labels to label indices. In our case, the label column (Category) will be encoded to label indices.

## Model Training and Evaluation:

We employ various supervised machine learning algorithms in Spark, including:

* Logistic Regression using Count Vector Features.
* Logistic Regression using TF-IDF features.
* Random Forest
* Decision Tree Classifier

In addition, we also utilize techniques such as cross-validation to tune hyperparameters and improve model performance.

## Logistic Regression using Count Vector Features

We train a logistic regression model using count vector features extracted from the crime descriptions. Count vectorization converts text data into numerical features representing the frequency of each word in the corpus. The logistic regression algorithm is applied to these features to classify the crimes into different categories.

## Logistic Regression using TF-IDF Features

Following the count vectorization, we also train another logistic regression model using TF-IDF (Term Frequency-Inverse Document Frequency) features. TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. This approach aims to capture the significance of words in distinguishing between different crime categories.

* A Logistic Regression accuracy of **97.20%** suggests that the model is performing exceptionally well in predicting the crime categories based on the given features. In the context of this problem, where accurately classifying crime descriptions into predefined categories is crucial for law enforcement resource allocation, such high accuracy indicates that the model can effectively assist in prioritizing and addressing various types of crimes. This level of accuracy implies that the model is successfully capturing the patterns and relationships within the dataset, making it a valuable tool for supporting law enforcement agencies in their decision-making processes.

* Furthermore, the logistic regression model utilizing TF-IDF features exhibits a similarly high accuracy rate of **97.17%**. These results affirm that logistic regression, when coupled with appropriate feature representations like TF-IDF, excels in categorizing crime descriptions accurately. The robust performance of logistic regression underscores its suitability for this text classification problem, emphasizing the significance of **effective feature engineering** in enhancing model performance.

## Cross Validation

To ensure the robustness of our models and optimize their hyperparameters, we use cross-validation. Cross-validation involves splitting the dataset into multiple subsets, training the model on different combinations of these subsets, and evaluating its performance. By tuning hyperparameters through cross-validation, we aim to improve the generalization ability of our models and prevent overfitting to the training data.

We will only tune the **Count Vectors Logistic Regression.**

Achieving a cross-validated Logistic Regression accuracy of **99.13%** shows how effective cross-validation is in making our model better. Through cross-validation, we iteratively fine-tuned the model's hyperparameters, optimizing its ability to generalize to new, unseen data. This process allowed us to systematically evaluate various parameter configurations and select the ones that yielded the best performance. As a result, the model demonstrates exceptional accuracy in predicting crime categories, showcasing the significant improvement achieved through cross-validation. By leveraging this technique, we were able to enhance the robustness and reliability of the model, ensuring its efficacy in real-world applications such as law enforcement resource allocation.

## Decision Tree classifier

Incorporating a Decision Tree classifier into our crime classification task allows us to create a hierarchical structure that learns from the features provided in the dataset to predict the corresponding crime categories. By setting parameters like the maximum depth of the tree, we control the complexity of the model and its ability to capture intricate patterns in the data. The Decision Tree algorithm splits the dataset into subsets based on the most significant features, creating a tree-like structure where each internal node represents a feature, each branch corresponds to a decision based on that feature, and each leaf node represents a predicted category. This method enables interpretable predictions, as we can trace the path from the root to the leaf node to understand how the model arrives at its decision.

The relatively low accuracy of the Decision Tree Model suggests that the decision tree model is not performing well in accurately predicting crime categories based on the given features.

The reason for this low accuracy could be the inherent limitations of decision trees, especially when dealing with high-dimensional and sparse data, as is often the case in text classification problems like crime categorization. Decision trees tend to create overly complex models, which can lead to overfitting, where the model learns to memorize the training data rather than generalize to new, unseen data. Additionally, decision trees may struggle to capture the intricate relationships and patterns present in textual data, resulting in suboptimal performance.

## Random Forest

Lastly, we employ the Random Forest algorithm, a popular ensemble learning technique that constructs multiple decision trees during training and outputs the mode of the classes as the prediction.

Random Forest is generally robust, but it's not ideal for high-dimensional sparse data. In such cases, where there are many features with low predictive power, Random Forest's performance may suffer. This is because it relies on decision trees, which can struggle with inefficient splitting decisions in such datasets. Linear models like Logistic Regression or Naive Bayes are often more suitable for handling high-dimensional sparse data efficiently. Therefore, while Random Forest has its strengths, it's important to consider the dataset's characteristics and explore alternative algorithms for better performance.

## Evaluation Results

Based on the evaluation results, it's clearly evident that logistic regression, particularly when optimized through cross-validation, is the most suitable model for the task of crime categorization based on textual data, outperforming both the Decision Tree and Random Forest models. The high accuracy of the logistic regression model underscores its efficacy in assisting law enforcement agencies in allocating resources and prioritizing investigations.

Overall, the project also highlights the importance of employing appropriate machine learning algorithms and techniques for text classification tasks, emphasizing the superior performance of logistic regression with cross-validation in this context.

## Conclusion

The Crime Classification System developed using PySpark offers a valuable tool for law enforcement agencies to accurately categorize crime descriptions and allocate resources effectively. By automating the classification process, this system contributes to enhancing public safety and streamlining investigation prioritization.
