## <span style="color:black"> PHASE 3 PROJECT SUBMISSION </span>
| Student Pace             | Project Review Date  | Instructor Name     | Blog Post URL  |
|--------------------------|--------------------------|---------------------|----------------|
| `Part Time`                | `September 01st, 2024`      | `Josphat Njuguna Wanjiru`      | _https://github.com/Giwanji/dsc-phase3-project.git_               |

**STUDENT NAME: JOSPHAT NJUGUNA WANJIRU**
# <span style="color:black"> SyriaTel CUSTOMER CHURN PREDICTION MODEL </span>
## <span style="color:black"> PROJECT OVERVIEW </span> <a class="anchor" id="first-bullet"></a>
This project seeks to build a Machine Learning classifier algorithm that can predict the probability of customers churning SyriaTel company using the companies data
## <span style="color:black"> BUISNESS UNDERSTANDING </span> <a class="anchor" id="first-bullet"></a>

Syriatel is a mobile network provider in Syria.[1] It is one of the only two providers in Syria, the other being MTN Syria. In 2022 the Syrian telecommunications authority awarded the third telecom license to Wafa Telecom.[3] It offers LTE with 150 Mb/s speeds, under the brand name Super Sur
## <span style="color:black"> BUSINESS PROBLEM </span> <a class="anchor" id="fourth-bullet"></a>

The ability to predict that a particular customer is at a high risk of churning, while there is still time to do something about it, represents a huge additional potential revenue source for every online business. Besides the direct loss of revenue that results from a customer abandoning the business, the costs of initially acquiring that customer may not have already been covered by the customer’s spending to date. (In other words, acquiring that customer may have actually been a losing investment.) Furthermore, it is always more difficult and expensive to acquire a new customer than it is to retain a current paying customer
Customer churn is the loss of clients or customers. Predicting churn can help the Telecom company, so it can effectively focus a customer retention marketing program (e.g. a special offer) to the subset of clients which are most likely to change their carrier. Therefore, the “churn” column is chosen as target and the following predictive analysis is a supervised classification problem.
## <span style="color:black"> BUSINESS AIM AND OBJECTIVES </span> <a class="anchor" id="fifth-bullet"></a>

The aim of this project is to predict customer churn and retention in SyriaTel company

The objectives of the are to:

* Develop a predictive model that accurately identifies customers at high risk of churn.

* Determine key factors influencing customer churn and retention.

* Optimize customer retention strategies based on predictive insights.
## <span style="color:black"> Loading relevant Libraries </span>

### <span style="color:black"> 
# scientific computing libaries
import pandas as pd
import numpy as np
import math
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# data mining libaries
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, learning_curve

# Class imbalance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

## <span style="color:black"> DATA UNDERSTANDING </span>

### <span style="color:red">

The dataset contains data on the customers of a SyriaTel Telecom company. Each row represents a customer and the columns contain customer’s attributes which are described in the following:
# Import the SyriaTel customer data
df = pd.read_csv('./bigml_59c28831336c6604c800002a.csv')
df.head()

Rows 3333
Columns 21
Each row is a customer



| Field Name                 | Description                                                                                   |
|----------------------------|-----------------------------------------------------------------------------------------------|
| **state**                  | The state the user lives in                                                                   |
| **account length**         | The number of days the user has had this account                                              |
| **area code**              | The code of the area the user lives in                                                        |
| **phone number**           | The phone number of the user                                                                  |
| **international plan**     | `true` if the user has the international plan, otherwise `false`                              |
| **voice mail plan**        | `true` if the user has the voice mail plan, otherwise `false`                                 |
| **number vmail messages**  | The number of voice mail messages the user has sent                                           |
| **total day minutes**      | Total number of minutes the user has been in calls during the day                             |
| **total day calls**        | Total number of calls the user has made during the day                                        |
| **total day charge**       | Total amount of money the user was charged by the Telecom company for calls during the day    |
| **total eve minutes**      | Total number of minutes the user has been in calls during the evening                         |
| **total eve calls**        | Total number of calls the user has made during the evening                                    |
| **total eve charge**       | Total amount of money the user was charged by the Telecom company for calls during the evening|
| **total night minutes**    | Total number of minutes the user has been in calls during the night                           |
| **total night calls**      | Total number of calls the user has made during the night                                      |
| **total night charge**     | Total amount of money the user was charged by the Telecom company for calls during the night  |
| **total intl minutes**     | Total number of minutes the user has been in international calls                              |
| **total intl calls**       | Total number of international calls the user has made                                         |
| **total intl charge**      | Total amount of money the user was charged by the Telecom company for international calls     |
| **customer service calls** | Number of customer service calls the user has made                                            |
| **churn**                  | `true` if the user terminated the contract, otherwise `false`                                 |




## <span style="color:black">EXPLORATORY DATA ANALYSIS (EDA)</span> <a class="anchor" id="eighth-bullet"></a>
In the EDA section, we explored the various features in the data, clean and tranform some of the features.
#checking data  shape
df.shape
print('Number of rows =', df.shape[0])
print('Number of columns =', df.shape[1])
# Check the data type
df.info()
### Splitting Categorical from Numerical
# Selecting categorical variables
categorical = df.select_dtypes(include=['object'])
#Selecting numerical variables
numerical = df.select_dtypes(include=['number'])
### Number of labels: cardinality
# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
The phone number has a  high cadinality and may pose problems in the classification model. Therefore it was dropped.
# Dropping phone number
df.drop(['phone number'], axis=1, inplace=True)
### Feature Engineering of Area Code and Churn Variable

The target variable 'churn' was converted from boolean to object and area code was converted from numeric to categorical 
# Feature Enginnering of Area code from mumerical to categorical
df['area code'] = df['area code'].astype(str)
# Feature Engineering of churn Variable from boolean datatype to object
df['churn'] = df['churn'].map({True: 1, False: 0}).astype('int')
df['churn'].head()
# checking for duplicates
df.duplicated().sum()
There are no duplicate values
# Checking for missing values 
df.isna().sum()
The dataset does not contain any significant number of NaN values
### Descriptive Statistics
# Computing the descriptive statistics
df.describe()
To aid in the interpretation of the above statistics we created graphs which visualize them in a better way. Firstly, we look at the distribution of the our target variable
### Churn Distribution
# Churn dictionary for labeling
churn_dict = {0: "No Churn", 1: "Churn"}
df['churn_label'] = df['churn'].map(churn_dict)

# Setting color palette
colors = sns.color_palette("husl", len(df['churn_label'].unique()))

# Creating the bar plot
sns.countplot(x='churn_label', data=df, palette=colors)

# Set plot title and labels
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('# Samples')

# Display the plot
plt.show()
churn_perc = df["churn"].sum() * 100 / df["churn"].shape[0]
print("Churn percentage is %.3f%%." % churn_perc)
We observe a noticeable imbalance in our data, with a higher number of samples for customers who did not churn compared to those who did. This class imbalance in the target variable might cause our predictive models to be skewed towards the majority class, which in this case is customers who did not churn. To mitigate this potential bias, we will consider implementing oversampling techniques when developing our models.

Following this, we will analyze how churn rates vary by state to assess the impact of geographic location on our target variable.
### Churn by State

df.groupby(["state", "churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30,10)) 

We notice that certain states, such as AK, HI, and IA, have a lower proportion of customers who churn, while others, like WA, MD, and TX, have a higher proportion. This indicates that the state a customer is in could be an important factor in predicting churn. Therefore, it’s essential to include the state variable in our further analysis.

The interactive graph below illustrates the distribution of each feature for customers who churned and those who didn’t. You can use the slider to switch between the various features and observe their distributions.
# Exclude non-numerical features and target variable
features_not_for_hist = ["state", "phone_number", "churn"]
features_for_hist = [x for x in df.columns if x not in features_not_for_hist]

# Separate churned and non-churned customers
churn = df[df['churn'] == 1]
no_churn = df[df['churn'] == 0]

# Number of rows and columns for the grid
n_rows = len(features_for_hist) // 2 + len(features_for_hist) % 2
n_cols = 3

plt.figure(figsize=(15, n_rows * 5))

for i, feature in enumerate(features_for_hist):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(churn[feature], kde=False, color='orange', label='Churn', alpha=0.6)
    sns.histplot(no_churn[feature], kde=False, color='blue', label='No Churn', alpha=0.6)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('# Samples')
    plt.legend()

plt.tight_layout()
plt.show()

One notable observation comes from the histogram of the "international_plan" feature. Customers who have an international plan exhibit a much higher proportion of churn compared to those without it.

The histograms for "total_day_minutes" and "total_day_charge" reveal a similar trend, where customers with higher values for these features are more likely to churn. Interestingly, this pattern does not hold for the number of day calls, suggesting that customers who churn tend to make longer calls rather than more frequent ones. In contrast, the distributions for minutes, charges, and the number of calls during other times of the day (i.e., evening, night) do not differ significantly between customers who churn and those who do not.

Another intriguing pattern emerges in the "total_intl_calls" feature. The data for customers who churn is more left-skewed compared to those who do not churn, indicating a distinct difference in their international calling behavior.
### Outlier Analysis using Boxplot
# Separate churned and non-churned customers
churn = df[df['churn'] == 1]
no_churn = df[df['churn'] == 0]

# Exclude non-numerical features and target variable
features_not_for_hist = ["state", "phone number", "churn"]
features_for_hist = [x for x in df.columns if x not in features_not_for_hist]

# Remove features with too few distinct values (e.g., binary features)
features_for_box = [col for col in features_for_hist if len(df[col].unique()) > 5]

# Number of rows and columns for the grid
n_rows = len(features_for_box) // 2 + len(features_for_box) % 2
n_cols = 2

# Set up the figure for multiple subplots
plt.figure(figsize=(15, n_rows * 5))

for i, feature in enumerate(features_for_box):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x='churn', y=feature, data=df, palette=["blue", "orange"])
    plt.title(f'Box Plot of {feature} by Churn Status')
    plt.xlabel('Churn')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()
When examining the box plot for the number of voice mail messages ("number_vmail_messages"), it becomes apparent that there are some outliers among customers who churned, though the majority of them sent zero voice mail messages. In contrast, customers who did not churn tend to send more voice mail messages.

Consistent with our observations from the histograms, the box plot also reveals that the median values for total day minutes and total day charge are higher for customers who churned compared to those who didn’t.

For the total international calls ("total_intl_calls"), the box plot indicates that both churned and non-churned customers make a similar number of international calls. However, churned customers tend to make longer calls, as their median total international minutes is higher than that of non-churned customers.

Lastly, the box plot for the number of customer service calls shows that customers who churned have a higher median and greater variance in the number of customer service calls compared to those who did not churn.
### Investigate Pairwise Correlation
# Calculate the correlation matrix
corr = numerical.corr()

# Set up the figure for the heatmap
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)

# Set the title and show the plot
plt.title('Heatmap of Pairwise Correlation of the Columns')
plt.xticks(rotation=40)
plt.yticks(rotation=0)
plt.show()
The heatmap reveals several key insights regarding the correlations among the features in the dataset. Notably, there is a perfect correlation between several pairs of features, particularly those related to call minutes and charges. For example, total day minutes and total day charge, as well as total evening minutes and total evening charge, total night minutes and total night charge, and total international minutes and total international charge, all show a correlation of 1.00. This strong relationship suggests that the charges are directly proportional to the number of minutes used, indicating that the Telecom company likely charges customers based on the duration of their calls.

In contrast, the correlation between the churn variable and other features is relatively weak, with the highest observed correlation being around 0.21. This weak association implies that while certain factors, such as having an international plan or the number of customer service calls, may have some influence on customer churn, these factors alone do not strongly predict churn.

Additionally, most other features exhibit weak correlations with each other, generally below 0.05, indicating that they are largely independent of one another. Given these observations, there is an opportunity to reduce the dimensionality of the dataset by removing redundant features. Specifically, since features like total day minutes and total day charge are perfectly correlated, one of these can be removed without losing significant information. Overall, while the weak correlations with churn suggest a need for further investigation, particularly in exploring more complex interactions or additional variables that might better explain customer behavior, the analysis provides a clear direction for simplifying the dataset by eliminating redundant features.
### Reducing Dimensionality of the Dataset
# Calculate the correlation matrix
corr_matrix = numerical.corr().abs()

# Create a mask to identify the upper triangle of the correlation matrix
upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

# Filter the correlation matrix to only keep the upper triangle
upper_corr_matrix = corr_matrix.where(upper_tri)

# Define a threshold for high correlation
threshold = 0.95

# Find index of features with correlation greater than the threshold
to_drop = [column for column in upper_corr_matrix.columns if any(upper_corr_matrix[column] > threshold)]

# Drop the features that have high correlation
reduced_df = df.drop(columns=to_drop)

# Output the names of the dropped features
print(f"Dropped columns due to high correlation: {to_drop}")

# Show the reduced DataFrame
print("Shape of reduced_df ",reduced_df.shape,"shape of original_df", df.shape)
reduced_df.head()

---
## PREPROCESSING

To prepare the dataset for further analysis, we first split it into the target column and the other predictor variables. Additionally, we standardize all features to ensure that features with higher absolute values do not disproportionately influence classifiers that rely on distance metrics.
# splitting the dataset into feature vectors and the target variable
y = reduced_df["churn"]
X = reduced_df.drop(["churn","churn_label"], axis=1)
# List of Categorical Columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to dummy variables
X = pd.get_dummies(X, columns=categorical_columns,drop_first=True)

categorical_columns
# Convert any boolean columns to integers (1 and 0)
X = X.astype(int)
X
# Split Train and Validation Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
We now have the training and testing sets prepared for model building. Before proceeding, it's essential to map all the feature variables onto the same scale, a process known as `feature scaling`. Here's how I performed it:
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# Scale the training data
X_train = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns) 

# Scale the test data
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_train.columns 
)

# Display the first few rows of the scaled training data
X_train.head()

# Getting statistics of X_train
X_train.describe()
---
---
## MODELLING

#### Baseline Metrics
Before building the actual model, I calculated the Null accuracy which represents the accuracy that could be achieved by always predicting the most frequent class.
It serves as a baseline to assess the performance of my classification models. Calculating null accuracy is straightforward and provides a valuable benchmark, especially in the context of imbalanced datasets.
When evaluating your model, it’s important to ensure that its accuracy significantly exceeds the null accuracy, indicating that the model is actually learning patterns from the data rather than simply guessing the most common outcome.
# check class distribution in test set
y_test.value_counts()

# Confusion matrix for the baseline
ConfusionMatrixDisplay.from_estimator(estimator=DummyClassifier(strategy='constant',constant=1).fit(X_train,y_train),X=X_test,y=y_test);

# Calculate the null accuracy
most_frequent_class_count = Counter(y_test).most_common(1)[0][1]
null_accuracy = most_frequent_class_count / len(y_test)

print(f'The Null Accuracy is: {null_accuracy:.4f}')
Given the code block above, it's clear that the target variable is imbalanced. To accurately evaluate the model's performance for both True (1) and False (0) labels, I'll calculate various classification metrics. These metrics will provide a comprehensive assessment of the model's performance.
---
def modelling(models, X_train, y_train, X_test, y_test):
    # Fit the models and plot the ROC curves
    plt.figure(figsize=(10, 8))
    # Call the null accuracy
    print(f'The baseline accuracy is {null_accuracy:.4f} which compares with the \n'
            f'the fitted models as follow:')

    print(" ")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Get the predicted probabilities for the positive class (class 1)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute the ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        #Validation
        # Accuracy Score
        accuracy_train_data = accuracy_score(y_train, y_train_pred)
        accuracy_test_data = accuracy_score(y_test, y_test_pred)
        print(f'Training set accuracy score for {model_name} is {accuracy_train_data:.4f}')
        print(f'Test set score for {model_name} is {accuracy_test_data:.4f}')


        print(" ")

        #Accuracy score
        print(f'Accuracy score for {model_name} is {accuracy_test_data:.4f}')

        # Precision Score
        print(f'Precision score for {model_name} is {precision_score(y_test,y_test_pred):.4f}')

        # Recall
        print(f'Recall score for {model_name} is {recall_score(y_test,y_test_pred):.4f}')

        # F1 Score
        print(f'F1 Score score for {model_name} is {f1_score(y_test,y_test_pred):.4f}')
        
        # Cross Validation 
        cv_scores = cross_val_score(model,X_train,y_train)
        print(f'Cross Validation Score for {model_name} is {cv_scores.mean():.4f}')
        
        print(" ")
     
        # Plot the ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    # Plot the diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Chance')

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves for Multiple Models')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Show the plot
    plt.show() 
# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Logistic Regression100': LogisticRegression(C=100, solver='liblinear', random_state=0),
    'Logistic Regression001)': LogisticRegression(C=0.001, solver='liblinear', random_state=0),
    'Decision Tree': DecisionTreeClassifier()
}

# calling modelling function
modelling(models, X_train, y_train, X_test, y_test)
### Observations

* The baseline accuracy for this dataset is 0.8546, serving as a benchmark to evaluate the effectiveness of various models. Here's an analysis of the performance metrics for Logistic Regression variants and a Decision Tree model compared to this baseline:

* Logistic Regression Models: The three Logistic Regression models (Logistic Regression, Logistic Regression100, and Logistic Regression001) demonstrate similar performance with training and test accuracies slightly above the baseline. The models exhibit precision scores around 0.5532 to 0.5769, indicating moderate success in predicting the positive class correctly. However, the recall values are notably low, especially for Logistic Regression001, which has a recall of only 0.1546. This suggests that the models are not very effective at capturing true positive cases, leading to relatively low F1 scores (ranging from 0.2439 to 0.3611). The cross-validation scores are consistent across the models, indicating stable but suboptimal generalization capabilities.
* Decision Tree Model: The Decision Tree model achieves perfect accuracy on the training set (1.0000), which suggests significant overfitting—it is likely memorizing the training data rather than generalizing well. However, its test accuracy of 0.8981 surpasses the baseline and all Logistic Regression models, indicating that despite overfitting, it performs better on unseen data. The Decision Tree also excels in precision (0.6629) and recall (0.6082), leading to a much higher F1 score (0.6344) compared to the Logistic Regression models. The cross-validation score of 0.9115 supports its consistent performance across different data subsets.

__ROC CURVE AUC SCORE__

* The ROC curve and AUC scores indicate that all models are performing better than random guessing, with AUC values close to or slightly above 0.79.
* The Decision Tree and Logistic Regression001 models have the highest AUC (0.80), suggesting they might have a slight edge in classification performance compared to the other Logistic Regression models.
* The similarity in AUC values across the models indicates that, while there are some differences, the overall performance of these models is comparable. * The choice between them depended on the risk of overfitting (as noted earlier with the Decision Tree.
### Confusion Matrix
# Print the Confusion Matrix and slice it into four pieces

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    
The confusion matrix shows `551 + 21 = 572 correct predictions` and `15 + 80 = 95 incorrect predictions`
---
### Handling Class Imbalance
The classes are highly imbalanced use of SMOTE with a sampling staregy of 0.5

# Instantiate SMOTE with random_state=42 and sampling_strategy=0.5
sm = SMOTE(random_state=42, sampling_strategy=0.2)
# Fit and transform X_train_scaled and y_train using sm
X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("Original class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Resampled class distribution:", dict(zip(*np.unique(y_train_oversampled, return_counts=True))))
modelling(models, X_train_oversampled,y_train_oversampled, X_test, y_test)
### __Observations__
 The baseline accuracy for this dataset is 0.8546, providing a benchmark against which to compare the performance of various models. Below is an analysis of the Logistic Regression variants and the Decision Tree model:

__Logistic Regression Models:__
* Logistic Regression: This model has a training accuracy of 0.8578 and a test accuracy of 0.8561, which is slightly above the baseline. The precision score is 0.5082, indicating a moderate ability to correctly predict positive cases, but the recall score is lower at 0.3196, meaning the model misses a significant number of actual positives. The F1 score, which balances precision and recall, is 0.3924, reflecting the model’s moderate performance. The cross-validation score of 0.8461 suggests that the model generalizes reasonably well, though it slightly underperforms compared to the baseline.
* Logistic Regression100: This model shows a training accuracy of 0.8589 and a test accuracy exactly at the baseline of 0.8546. The precision (0.5000) and recall (0.3196) are similar to the original Logistic Regression model, leading to an F1 score of 0.3899. The cross-validation score of 0.8465 is consistent with the training and test accuracies, indicating stable but unremarkable performance.
* Logistic Regression001: With a training accuracy of 0.8501 and a test accuracy of 0.8606, this model slightly outperforms the other Logistic Regression models on the test set. The precision (0.5526) is higher, but the recall is significantly lower at 0.2165, which results in a lower F1 score of 0.3111. The cross-validation score of 0.8421 is slightly lower than the other Logistic Regression models, suggesting slightly less stability in generalization.

__Decision Tree Model__
 The Decision Tree model, with a perfect training accuracy of 1.0000, indicates overfitting, where the model captures noise and specific details of the training data rather than general patterns. However, its test accuracy of 0.9115 significantly exceeds the baseline and all Logistic Regression models. The precision (0.7111) and recall (0.6598) are both high, leading to an F1 score of 0.6845, indicating strong overall performance. The cross-validation score of 0.9046 further suggests consistent and reliable performance across different subsets of data, despite the overfitting on the training set
__ROC CURVE AUC SCORE__

The Decision Tree model outperforms the Logistic Regression models slightly in terms of AUC, indicating better overall classification performance. The Logistic Regression models are closely matched, with Logistic Regression001 performing marginally better than the others. The ROC curves provide a visual representation of these differences in model performance.
---
---
## CONLUSION
* The Logistic Regression Models provided stable and consistent performance, with lower risk of overfitting, but at the cost of lower precision and recall, leading to modest F1 scores.
* The Decision Tree Model offers higher accuracy, precision, recall, and F1 score on the test set, but it is prone to overfitting, as evidenced by the perfect training accuracy.
* The choice between these models was based on avoidance of overfitting in decision tree and rather prefered logistic regression which is more generalized
## RECOMMENDATIONS
**Model to Selection** 

Based on the metrics, the **Decision Tree** model appears to be the best option overall, with a high AUC (0.80), and offers a higher accuracy, precision, recall, and F1 score on the test set however it is prone to overfitting
