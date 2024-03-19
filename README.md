# K_nearest_Neighbors-KNN27
Applying K-Nearest Neighbors classifier model.
``` python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('Social_Network_Ads.csv')
df.head()

#Observations: In the above output, you can see the first few rows of the dataset.
#There are different columns such as user ID, gender, age, estimated salary, and purchased data.

df.info()

Observation: There are no null values.

df['Purchased'].value_counts()

#Observation: The output above indicates that 143 people purchased while 257 people didn't.

Gender = pd.get_dummies(df['Gender'],drop_first=True)

df = pd.concat([df,Gender],axis=1)

df.drop(['Gender'],axis=1,inplace=True)
X = df[['Age','EstimatedSalary','Male']] # Male is the Gender column converted to contain numerical values.
y = df['Purchased']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)
scaled_features

df_feat = pd.DataFrame(scaled_features,columns=X.columns)
df_feat.head()

#Observation:The data is transformed here.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,y,
                                                    test_size=0.20)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create and train the logistic regression model
lr = LogisticRegression(multi_class='ovr', solver='liblinear')
model = lr.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# ROC Curve
proba = model.predict_proba(X_test)
proba_class1 = proba[:, 1]  # Probability of positive class
fpr, tpr, thresholds = roc_curve(y_test, proba_class1)
roc_auc = roc_auc_score(y_test, proba_class1)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

#Observations: Here, we can observe the classification report and ROC curve of the classification.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))

#Observation: In this confusion matrix, let's look at non-purchased values (Negative), the model classified 21 datapoints and misclassified 2 datapoints.

print(classification_report(y_test,pred))

#Observations: In the above output, we can see that we are able to achieve 90% accuracy.
#For the purchase, we are able to have a precision of 90 and a recall of 87 with an f1-score of 88.
