import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler , LabelEncoder


df = pd.read_csv('')

df['color'] = ''

df.loc[df['class'] == 0, 'color'] = 'green'
df.loc[df['class'] == 1, 'color'] = 'red'

df.drop('class', axis = 1)

X = df[['variance', 'skewness', 'curtosis', 'entropy']].values
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
le = LabelEncoder()
Y = le.fit_transform(df['color'].values) # label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 12)

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,Y_train)

predicted = log_reg_classifier.predict(X_test)

accuracy = accuracy_score(Y_test, predicted)

matrix_one = confusion_matrix(Y_test, predicted)

# table

TP = matrix_one[0][0] / sum(sum(matrix_one))

FP = matrix_one[0][1] / sum(sum(matrix_one))

FN = matrix_one[1][0] / sum(sum(matrix_one))

TN = matrix_one[1][1] / sum(sum(matrix_one))

ACC = accuracy

TPR = TP / (TP + FN)

TNR = TN / (TN + FP)

d = {'TP': TP,'TN': TN,'FP': FP,'FN': FN, 'ACC':ACC, 'TPR' : TPR, 'TNR' : TNR}

tble = pd.DataFrame(data = d, index = [''])

tble # accuracy is 97%, TPR is 99% and TNR is 95 %




