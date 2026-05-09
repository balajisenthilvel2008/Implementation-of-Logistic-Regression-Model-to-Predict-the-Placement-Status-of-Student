# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load the dataset, drop unnecessary columns, and encode categorical variables.
2. Define the features (X) and target variable (y).
3. Split the data into training and testing sets.
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BALAJI S
RegisterNumber: 212225220015
*/
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics
df=pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")
df1=df.copy()
df1=df1.drop(['sl_no','salary'],axis=1)
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df1['ssc_b']=le.fit_transform(df1['ssc_b'])
df1['hsc_b']=le.fit_transform(df1['hsc_b'])
df1['hsc_s']=le.fit_transform(df1['hsc_s'])
df1['degree_t']=le.fit_transform(df1['degree_t'])
df1['workex']=le.fit_transform(df1['workex'])
df1['specialisation']=le.fit_transform(df1['specialisation'])
df1['status']=le.fit_transform(df1['status'])
x=df1.iloc[:,:-1]
y=df1['status']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy Score:",accuracy)
print("\nConfusion Matrix:\n",confusion)
print("\nClassification Report:\n",cr)
cn_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=['true','false'])
cn_display.plot()
```

## Output:

<img width="967" height="420" alt="image" src="https://github.com/user-attachments/assets/ffcdac61-7aac-4fb9-98cd-e7e490e158a9" />
<BR>
<img width="942" height="707" alt="image" src="https://github.com/user-attachments/assets/8402361e-756e-4208-a2f1-c9caca4cd2f6" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
