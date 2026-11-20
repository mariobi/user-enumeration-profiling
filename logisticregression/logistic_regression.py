# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn import metrics 

sns.set_style('whitegrid')

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)
dataframe1 = pd.read_excel('../Data.xlsx')
df = dataframe1.iloc[:,:]
df = df.fillna(0)
print(df.dtypes)

TargetVariable=['gender']
Predictors=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M1', 'M2', 'M3']

X=df[Predictors].values
Y=df[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=78)

X_train, Y_train = sm.fit_resample(X_train, Y_train)

# Logistic Regression
logreg = LogisticRegression(solver='lbfgs',penalty='l2',max_iter=1000)
logreg.fit(X_train, Y_train.ravel())

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("Logistic Regression: ")
print(round(acc_log,2,), "%")
predictions = cross_val_predict(logreg, X_train, Y_train, cv=10)
confusion_matrix(Y_train, predictions)
print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))
print("f1 score:",f1_score(Y_train, predictions))
print("")

# Predictions on testing data
Predictions=logreg.predict(X_test)


# Generating a data frame for analyzing the test data
TestingData=pd.DataFrame(data=X_test, columns=Predictors)
TestingData['gender']=Y_test
TestingData['PredictedGenderProb']=Predictions

# Defining the probability threshold
def probThreshold(inpProb):
    if inpProb  >  0.5:
        return(1)
    else:
        return(0)

# Generating predictions on the testing data by applying probability threshold
TestingData['PredictedGender']=TestingData['PredictedGenderProb'].apply(probThreshold)
print(TestingData)
###############################################
print('\n######### Testing Accuracy Results logistic regression #########')
print(metrics.classification_report(TestingData['gender'], TestingData['PredictedGender']))
print(metrics.confusion_matrix(TestingData['gender'], TestingData['PredictedGender']))

# getting the probabilities of our predictions
y_scores = logreg.predict_proba(X_train)
y_scores = y_scores[:,1]
precision, recall, threshold = precision_recall_curve(Y_train, y_scores)

