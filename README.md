# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding
2. Load the Dataset
3. Define Features and Labels
4. Split Data into Training and Testing Sets
5. Convert Text to Numerical Data
6. Train the Classifier
7. Make Predictions
8. Evaluate Model Performance

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SANJAI U
RegisterNumber:  24004616
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Detect the file encoding
file = 'spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
    print(result)  # Output the detected encoding

# Load the CSV file with the detected encoding
data = pd.read_csv(file, encoding=result['encoding'])

# Display the first few rows of the dataset
print(data.head())

# Display dataset information
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Correct column names
# Assuming "v1" is the label column (e.g., spam/ham) and "v2" is the message text
x = data["v2"].values  # Feature: the text messages
y = data["v1"].values  # Label: spam or ham

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Convert text data to numerical data using CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train a Support Vector Classifier (SVC)
svc = SVC()
svc.fit(x_train, y_train)

# Predict the labels for the test set
y_pred = svc.predict(x_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

*/
```

## Output:
![image](https://github.com/user-attachments/assets/6fb3878c-8a66-4b5f-b250-1365e05076c3)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
