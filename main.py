#import accuracy as accuracy
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.ensemble import RandomForestClassifier

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
df2test = pd.read_csv("testset.csv")
labels_train = df2.pop("win").tolist()
labels_test = df2test.pop("win").tolist()

df2 = normalize(df2)
df2test = normalize(df2test)

df2=np.array(df2)
df2test=np.array(df2test)
# Y=df2[:,-1]
Y=np.array(labels_train)
X=df2[:,0:df2.shape[1]-1]
# Y1=df2test[:,-1]
Y1=np.array(labels_test)
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
model1 = KNeighborsClassifier(n_neighbors=5)
model1.fit(X,Y)
score=cross_validate(model1,X,Y,scoring='accuracy')
print("KNeighbors Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model1,X,Y,scoring='precision_macro')
print("KNeighbors Cross Validation precision %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model1,X,Y,scoring='recall_macro')
print("KNeighbors Cross Validation recall %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model1,X,Y,scoring='f1_macro')
print("KNeighbors Cross Validation f1 %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model1,X,Y,scoring='jaccard_macro')
print("KNeighbors Cross Validation jaccard %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model1.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
precision1 = precision_score(Y1, predictions, average='macro')
recall1 = recall_score(Y1, predictions, average='macro')
f1 = f1_score(Y1,predictions,average='macro')
jaccard = jaccard_score(Y1, predictions, average='macro')
print("KNeighbors Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
print("KNeighbors Model Precision: %.2f%%" % (precision1 * 100.0))
print("KNeighbors Model Recall: %.2f%%" % (recall1 * 100.0))
print("KNeighbors Model F1: %.2f%%" % (f1 * 100.0))
print("KNeighbors Model jaccard: %.2f%%" % (jaccard * 100.0))
print()

## Gaussian Model
model2 = GaussianNB()
model2.fit(X,Y)
score=cross_validate(model2,X,Y,scoring='accuracy')
print("Gaussian Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model2,X,Y,scoring='precision_macro')
print("Gaussian Cross Validation precision %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model2,X,Y,scoring='recall_macro')
print("Gaussian Cross Validation recall %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model2,X,Y,scoring='f1_macro')
print("Gaussian Cross Validation f1 %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model2,X,Y,scoring='jaccard_macro')
print("Gussian Cross Validation jaccard %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model2.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
precision1 = precision_score(Y1, predictions, average='macro')
recall1 = recall_score(Y1, predictions, average='macro')
f1 = f1_score(Y1,predictions,average='macro')
jaccard = jaccard_score(Y1, predictions, average='macro')
print("Gaussian Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
print("Gaussian Model Precision: %.2f%%" % (precision1 * 100.0))
print("Gaussian Model Recall: %.2f%%" % (recall1 * 100.0))
print("Gaussian Model F1: %.2f%%" % (f1 * 100.0))
print("Guassian Model jaccard: %.2f%%" % (jaccard * 100.0))
print()

# Discriminant Analysis
model3 = LinearDiscriminantAnalysis()
seed = 42
model3.fit(X,Y)
score=cross_validate(model3,X,Y,scoring='accuracy')
print("Discriminant Analysis Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model3,X,Y,scoring='precision_macro')
print("Discriminant Analysis Cross Validation precision %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model3,X,Y,scoring='recall_macro')
print("Discriminant Analysis Cross Validation recall %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model3,X,Y,scoring='f1_macro')
print("Discriminant Analysis Cross Validation f1 %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model3,X,Y,scoring='jaccard_macro')
print("Discriminant Analysis Cross Validation jaccard %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model3.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
precision1 = precision_score(Y1, predictions, average='macro')
recall1 = recall_score(Y1, predictions, average='macro')
f1 = f1_score(Y1,predictions,average='macro')
jaccard = jaccard_score(Y1, predictions, average='micro')
print("Discriminant Analysis Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
print("Discriminant Analysis Model Precision: %.2f%%" % (precision1 * 100.0))
print("Discriminant Analysis Model Recall: %.2f%%" % (recall1 * 100.0))
print("Discriminant Analysis Model F1: %.2f%%" % (f1 * 100.0))
print("Discriminant Analysis Model jaccard: %.2f%%" % (jaccard * 100.0))
print()

## Logistic Regression Model
model4 = LogisticRegression()
model4.fit(X,Y)
score=cross_validate(model4,X,Y,scoring='accuracy')
print("Logistic Regression Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model4,X,Y,scoring='precision_macro')
print("Logistic Regression Cross Validation precision %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model4,X,Y,scoring='recall_macro')
print("Logistic Regression Cross Validation recall %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model4,X,Y,scoring='f1_macro')
print("Logistic Regression Cross Validation f1 %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model4.predict(X1)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
precision1 = precision_score(Y1, predictions, average='macro')
recall1 = recall_score(Y1, predictions, average='macro')
f1 = f1_score(Y1,predictions,average='macro')
print("Logistic Regression Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
print("Logistic Regression Model Precision: %.2f%%" % (precision1 * 100.0))
print("Logistic Regression Model Recall: %.2f%%" % (recall1 * 100.0))
print("Logistic Regression Model F1: %.2f%%" % (f1 * 100.0))
print()

# seed = 42
# # #kfold = model_selection.KFold(n_splits=2,shuffle=True, random_state=seed)
# model5=svm.SVC(gamma=0.001,kernel='linear', C=100).fit(X,Y)
# score=cross_validate(model5,X,Y,scoring='accuracy')
# print("SVC Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
# score=cross_validate(model5,X,Y,scoring='precision_macro')
# print("SVC Cross Validation precision %.2f%%" % (np.mean(score['test_score'])*100))
# score=cross_validate(model5,X,Y,scoring='recall_macro')
# print("SVC Cross Validation recall %.2f%%" % (np.mean(score['test_score'])*100))
# score=cross_validate(model5,X,Y,scoring='f1_macro')
# print("SVC Cross Validation f1 %.2f%%" % (np.mean(score['test_score'])*100))
# y_pred = model5.predict(X1)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy1 = accuracy_score(Y1, predictions)
# precision1 = precision_score(Y1, predictions, average='macro')
# recall1 = recall_score(Y1, predictions, average='macro')
# f1 = f1_score(Y1,predictions,average='macro')
# print("SVC Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
# print("SVC Model Precision: %.2f%%" % (precision1 * 100.0))
# print("SVC Model Recall: %.2f%%" % (recall1 * 100.0))
# print("SVC Model F1: %.2f%%" % (f1 * 100.0))

#Random Forest
model6 = RandomForestClassifier()
model6.fit(X,Y)
score=cross_validate(model6,X,Y,scoring='accuracy')
print("Random Forest Cross Validation accuracy %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model6,X,Y,scoring='precision_macro')
print("Random Forest Cross Validation precision %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model6,X,Y,scoring='recall_macro')
print("Random Forest Cross Validation recall %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model6,X,Y,scoring='f1_macro')
print("Random Forest Cross Validation f1 %.2f%%" % (np.mean(score['test_score'])*100))
score=cross_validate(model6,X,Y,scoring='jaccard_macro')
print("Random Forest Cross Validation jaccard %.2f%%" % (np.mean(score['test_score'])*100))
y_pred = model6.predict(X1)
predictions = [round(value) for value in y_pred]
# # evaluate predictions
accuracy1 = accuracy_score(Y1, predictions)
precision1 = precision_score(Y1, predictions, average='macro')
recall1 = recall_score(Y1, predictions, average='macro')
f1 = f1_score(Y1,predictions,average='macro')
jaccard = jaccard_score(Y1, predictions, average='macro')
print("Random Forest Model Accuracy: %.2f%%" % (accuracy1 * 100.0))
print("Random Forest Model Precision: %.2f%%" % (precision1 * 100.0))
print("Random Forest Model Recall: %.2f%%" % (recall1 * 100.0))
print("Random Forest Model F1: %.2f%%" % (f1 * 100.0))
print("Random Forest Model jaccard: %.2f%%" % (jaccard * 100.0))
