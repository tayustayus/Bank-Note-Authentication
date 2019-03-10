# Bank-Note-Authentication
Leveraging on an existing data , evaluated the accuracy of a Bank note via the Random Forest Algorithm of Machine Learning
#The task here is to predict whether a bank currency note is authentic or not
# Attributes used are variance, skewness, entropy,curtosis of the image.
#import pandas 
import pandas as pd
#import seaborn module
import seaborn as sns
# set up the matplotlib environment
import matplotlib.pyplot as plt
#import numpy module
import numpy as np
databill=pd.read_csv('bill_authentication.csv')
databill.head()

X = databill.iloc[:, 0:4].values  
y = databill.iloc[:, 4].values  

from sklearn.model_selection import train_test_split
​
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
​
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

from sklearn.ensemble import RandomForestClassifier
​
regressor = RandomForestClassifier(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
​
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred)) 

[[155   2]
 [  1 117]]
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       157
           1       0.98      0.99      0.99       118

   micro avg       0.99      0.99      0.99       275
   macro avg       0.99      0.99      0.99       275
weighted avg       0.99      0.99      0.99       275

0.9890909090909091
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))

[[155   2]
 [  1 117]]
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       157
           1       0.98      0.99      0.99       118

   micro avg       0.99      0.99      0.99       275
   macro avg       0.99      0.99      0.99       275
weighted avg       0.99      0.99      0.99       275

0.9890909090909091

x=(20,50,100,150)
y=(98.9,98.9,98.9,98.9)
plt.plot(x,y)
plt.title('Accuracy')
plt.legend(['tree')
plt.show()
