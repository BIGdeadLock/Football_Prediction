#import accuracy as accuracy
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import svm

def normalize(df):
    cols = df.columns.tolist()
    home_team = df["HomeTeamAPI"]
    away_team = df["AwayTeamAPI"]
    cols.remove("HomeTeamAPI")
    cols.remove("AwayTeamAPI")
    df.drop(columns=['HomeTeamAPI'], inplace=True)
    df.drop(columns=['AwayTeamAPI'], inplace=True)
    min_max = preprocessing.MinMaxScaler()
    scaled_df = min_max.fit_transform(df.values)
    df2 =  pd.DataFrame(scaled_df,columns=cols)
    df2.insert(0 , "HomeTeamAPI", home_team)
    df2.insert(0 , "AwayTeamAPI", away_team)

    return df2

df2 = pd.read_csv('dataset_no_2015_2016.csv')
df2 = normalize(df2)

df2test = pd.read_csv("testset.csv")
df2test = normalize(df2test)

# df2train, df2test= train_test_split(df2, test_size=0.35, random_state=42)

df2=np.array(df2)
df2test=np.array(df2test)
Y=df2[:,-1]
X=df2[:,0:df2.shape[1]-1]
Y1=df2test[:,-1]
X1=df2test[:,0:df2.shape[1]-1]


## Model XBClassifier
# model = XGBClassifier()
# model.fit(X,Y)
# y_pred = model.predict(X1)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(Y1, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

## KNeighbors Model
model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X,Y)
score=cross_validate(model1,X,Y,scoring="accuracy")
print("KNeighbors Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model1.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
print("KNeighbors Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
print()

## Gaussian Model
model2 = GaussianNB()
model2.fit(X,Y)
score=cross_validate(model2,X,Y,scoring="accuracy")
print("Gaussian Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model2.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
print("Gaussian Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
print()

# Discriminant Analysis
model3 = LinearDiscriminantAnalysis()
seed = 42
model3.fit(X,Y)
score=cross_validate(model3,X,Y,scoring="accuracy")

print("Discriminant Analysis Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model3.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
print("Discriminant Analysis Accuracy: %.2f%%" % (accuracy1 * 100.0))
print()

## Logistic Regression Model
model4 = LogisticRegression()
model4.fit(X,Y)
score=cross_validate(model4,X,Y,scoring="accuracy")
print("Logistic Regression Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model4.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
print("Logistic Regression Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
print()

# seed = 42
# #kfold = model_selection.KFold(n_splits=2,shuffle=True, random_state=seed)
model5=svm.SVC(kernel='linear', C=1).fit(X,Y)
score=cross_validate(model5,X,Y,scoring="accuracy")
print("SVM Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model5.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
print("SVC Model Accuracy: %.2f%%" % (accuracy1 * 100.0))

