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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics 

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 6)
sns.set_style('whitegrid')

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)
dataframe1 = pd.read_excel('../Data.xlsx')
df = dataframe1.iloc[:,:]
df = df.fillna(0)

TargetVariable=['gender']
Predictors=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'M1', 'M2', 'M3']
X=df[Predictors].values
Y=df[TargetVariable].values

### Sandardization of data ###
### We does not standardize the Target variable for classification
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
 
 
# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

PredictorScalerFit=PredictorScaler.fit(X_train)
X_train = PredictorScalerFit.transform(X_train)
X_test = PredictorScalerFit.transform(X_test)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=78)

X_train, Y_train = sm.fit_resample(X_train, Y_train)

precision_avg_0 = 0
recall_avg_0 = 0
f1_avg_0 = 0
precision_avg_1 = 0
recall_avg_1 = 0
f1_avg_1 = 0

for i in range(0,20):
  random_forest = RandomForestClassifier(n_estimators=1400,min_samples_split=5,min_samples_leaf=1,max_features='sqrt',max_depth=80,bootstrap=False)
  random_forest.fit(X_train, Y_train.ravel())

  Y_prediction = random_forest.predict(X_test)

  random_forest.score(X_train, Y_train)
  acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
  print("Random Forest: ")
  print(round(acc_random_forest,2,), "%")

  predictions = cross_val_predict(random_forest, X_train, Y_train, cv=10)
  confusion_matrix(Y_train, predictions)
  print("Precision:", precision_score(Y_train, predictions))
  print("Recall:",recall_score(Y_train, predictions))
  print("f1 score:",f1_score(Y_train, predictions))
  print("")
  #precision_avg = precision_avg + precision_score(Y_train, predictions)
  #recall_avg = recall_avg + recall_score(Y_train, predictions)
  #f1_avg = f1_avg + f1_score(Y_train, predictions)
  # Predictions on testing data
  Predictions=random_forest.predict(X_test)

  # Scaling the test data back to original scale
  Test_Data=PredictorScalerFit.inverse_transform(X_test)

  # Generating a data frame for analyzing the test data
  TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
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
  report = metrics.classification_report(TestingData['gender'], TestingData['PredictedGender'], output_dict=True)
  ###############################################
  print('\n######### Testing Accuracy Results Random Forest  #########')
  print(metrics.classification_report(TestingData['gender'], TestingData['PredictedGender']))
  print(metrics.confusion_matrix(TestingData['gender'], TestingData['PredictedGender']))
  precision_avg_0 = precision_avg_0 + report['0']['precision']
  recall_avg_0 = recall_avg_0 + report['0']['recall']
  f1_avg_0 = f1_avg_0 + report['0']['f1-score']
  precision_avg_1 = precision_avg_1 + report['1']['precision']
  recall_avg_1 = recall_avg_1 + report['1']['recall']
  f1_avg_1 = f1_avg_1 + report['1']['f1-score']

print("avg precision_0: " + str(precision_avg_0/20))
print("avg recall_0: " + str(recall_avg_0/20))
print("avg f1_0: " + str(f1_avg_0/20))
print("avg precision_1: " + str(precision_avg_1/20))
print("avg recall_1: " + str(recall_avg_1/20))
print("avg f1_1: " + str(f1_avg_1/20))

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]
precision, recall, threshold = precision_recall_curve(Y_train, y_scores)

# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
#plt.show()
