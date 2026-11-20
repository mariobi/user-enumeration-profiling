# import pandas lib as pd
import pandas as pd
import matplotlib.pyplot as plt
import time 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# read by default 1st sheet of an excel file
dataframe1 = pd.read_excel('../Data.xlsx')

#print(dataframe1)

df_female = dataframe1.loc[dataframe1['gender'] == 1]
df_male = dataframe1.loc[dataframe1['gender'] == 0]
#print(df_female)
#print(df_male)

df_female_values = df_female.iloc[:, 1:]

stats_female = df_female_values.describe()

df_male_values = df_male.iloc[:, 1:]

stats_male = df_male_values.describe()
#print(stats_male)
#print(stats_male.iloc[1])
#print(stats_female.iloc[1])


male_minus_female_means = stats_male.iloc[1] - stats_female.iloc[1]
female_minus_male_means = stats_female.iloc[1] - stats_male.iloc[1]


factor = 0.45
female_weights = ['B4','B5','C2','D7','E1','F4','F5','G4','J5','K4','L4']
weight_factor_male = 1.12
weight_factor_female = 1.2


male_weights = ['A2','A5','A6','B1','B2','B9','C3','C4','C5','D3','E2','D9','E3','E4','E5','E7','E8','F3','F6','G2','G9','H1','H2','H3','H6','H7','H9','I1','I4','I5','I6','I8','J1','J2','J4','J7','J8','K1','K2','K3','K5','K7','K8','K9','L2','L5','L6','L9','M2']



df_female_values = df_female.iloc[:, 1:]
stats_female = df_female_values.describe()
df_male_values = df_male.iloc[:, 1:]

tmp_femalestats = df_female_values.describe()
tmp_malestats = df_male_values.describe()
#print(tmp_femalestats.iloc[1,1])

print("---- OPTMIZED ----")
print("---- male prediction: ----")
time.sleep(1)
tp = 0
fp = 0
stats_female = df_female_values.describe()

for i in range(171):
        row = df_male_values.iloc[i]
        row2 = df_female_values.iloc[i]
        df_male_values.drop(df_male_values.index[i], inplace=True)
        df_female_values.drop(df_female_values.index[i], inplace=True)
        stats_male = df_male_values.describe()
        stats_female = df_female_values.describe()
        for column in stats_female.columns[1:]:
                if stats_female[column].name in female_weights:
                        stats_female[column].iloc[1] = stats_female[column].iloc[1]*weight_factor_female
        for column in stats_male.columns[1:]:
                if stats_male[column].name in male_weights:
                        stats_male[column].iloc[1] = stats_male[column].iloc[1]*weight_factor_male
        male_minus_female_means = stats_male.iloc[1] - stats_female.iloc[1]
        df_diff_mean = male_minus_female_means*row
        df_diff_mean2 = male_minus_female_means*row2
        if df_diff_mean.sum()-factor >= 0:
                tp+=1
        if df_diff_mean2.sum()-factor >= 0:
                fp+=1
        #print(df_diff_mean2.sum()-factor)
        df_male_values.loc[i+0.5] = row
        df_male_values = df_male_values.sort_index().reset_index(drop=True)
        df_female_values.loc[i+0.5] = row2
        df_female_values = df_female_values.sort_index().reset_index(drop=True)        
print("")
print("true positive: " + str(tp))
print("true negative: " + str(fp))
print("success: " + str(100/171*tp))
#print("success2: " + str(100/171*(171-fp)))
print("")
precision=tp/(tp+fp)
recall=tp/(tp+(171-tp))
f1=2*((precision*recall)/(precision+recall))
print("precision: " + str(precision))
print("recall: " + str(recall))
print("f1 :" + str(f1))
print("")
time.sleep(4)

# reset stats
stats_female = df_female_values.describe()
stats_male = df_male_values.describe()

print("---- female prediction: ----")
time.sleep(1)
tp = 0
fp = 0
df_female_values = df_female.iloc[:, 1:]
df_male_values = df_male.iloc[:, 1:]
stats_male = df_male_values.describe()
for i in range(171):
        row = df_female_values.iloc[i]
        row2 = df_male_values.iloc[i]
        df_female_values.drop(df_female_values.index[i], inplace=True)
        stats_female = df_female_values.describe()
        df_male_values.drop(df_male_values.index[i], inplace=True)
        stats_male = df_male_values.describe()
        for column in stats_male.columns[1:]:
                if stats_male[column].name in male_weights:
                        stats_male[column].iloc[1] = stats_male[column].iloc[1]*weight_factor_male
        for column in stats_female.columns[1:]:
                if stats_female[column].name in female_weights:
                        stats_female[column].iloc[1] = stats_female[column].iloc[1]*weight_factor_female
        male_minus_female_means = stats_male.iloc[1] - stats_female.iloc[1]
        df_diff_mean = male_minus_female_means*row
        df_diff_mean2 = male_minus_female_means*row2
        if df_diff_mean.sum()-factor < 0:
                 tp+=1
        if df_diff_mean2.sum()-factor < 0:
                 fp+=1
        df_male_values.loc[i+0.5] = row2
        df_male_values = df_male_values.sort_index().reset_index(drop=True)
        df_female_values.loc[i+0.5] = row
        df_female_values = df_female_values.sort_index().reset_index(drop=True)
        #print(df_diff_mean2.sum()-factor)
print("")
print("true positive: " + str(tp))
print("true negative: " + str(fp))
print("success: " + str(100/171*tp))
#print("success2: " + str(100/171*(171-fp)))
print("")
precision=tp/(tp+fp)
recall=tp/(tp+(171-tp))
f1=2*((precision*recall)/(precision+recall))
print("precision: " + str(precision))
print("recall: " + str(recall))
print("f1 :" + str(f1))
time.sleep(10)

df_male_values = df_male.iloc[:, 1:]
stats_male = df_male_values.describe()
df_female_values = df_female.iloc[:, 1:]
stats_female = df_female_values.describe()
print("")
print("---- NORMAL ----")
print("---- male prediction: ----")
print("")
tp = 0
fp = 0
for i in range(171):
        row = df_male_values.iloc[i]
        row2 = df_female_values.iloc[i]
        df_male_values.drop(df_male_values.index[i], inplace=True)
        stats_male = df_male_values.describe()
        df_female_values.drop(df_female_values.index[i], inplace=True)
        stats_female = df_female_values.describe()
        male_minus_female_means = stats_male.iloc[1] - stats_female.iloc[1]
        df_diff_mean = male_minus_female_means*row
        df_diff_mean2 = male_minus_female_means*row2
        if df_diff_mean.sum()-factor >= 0:
                tp+=1
        if df_diff_mean2.sum()-factor >= 0:
                fp+=1	
        df_male_values.loc[i+0.5] = row
        df_male_values = df_male_values.sort_index().reset_index(drop=True)
        df_female_values.loc[i+0.5] = row2
        df_female_values = df_female_values.sort_index().reset_index(drop=True)

print("true positive: " + str(tp))
print("true negative: " + str(fp))
print("success: " + str(100/171*tp))
#print("succress2: " + str(100/171*(171-fp)))

precision=tp/(tp+fp)
recall=tp/(tp+(171-tp))
f1=2*((precision*recall)/(precision+recall))
print("")
print("precision: " + str(precision))
print("recall: " + str(recall))
print("f1 :" + str(f1))

print("")
print("---- female prediction: ----")
print("")
tp = 0
fp = 0
df_female_values = df_female.iloc[:, 1:]
df_male_values = df_male.iloc[:, 1:]
stats_male = df_male_values.describe()
df_female_values = df_female.iloc[:, 1:]
stats_female = df_female_values.describe()

for i in range(171):
        row = df_female_values.iloc[i]
        row2 = df_male_values.iloc[i]
        df_female_values.drop(df_female_values.index[i], inplace=True)
        stats_female = df_female_values.describe()
        df_male_values.drop(df_male_values.index[i], inplace=True)
        stats_male = df_male_values.describe()
        male_minus_female_means = stats_male.iloc[1] - stats_female.iloc[1]
        df_diff_mean = male_minus_female_means*row
        df_diff_mean2 = male_minus_female_means*row2
        if df_diff_mean.sum()-factor < 0:
                 tp+=1
        if df_diff_mean2.sum()-factor < 0:
                 fp+=1
        df_male_values.loc[i+0.5] = row2
        df_male_values = df_male_values.sort_index().reset_index(drop=True)
        df_female_values.loc[i+0.5] = row
        df_female_values = df_female_values.sort_index().reset_index(drop=True)

print("true positive: " + str(tp))
print("true negative: " + str(fp))
print("success: " + str(100/171*tp))
#print("succress2: " + str(100/171*(171-fp)))
print("")
precision=tp/(tp+fp)
recall=tp/(tp+(171-tp))
f1=2*((precision*recall)/(precision+recall))
print("precision: " + str(precision))
print("recall: " + str(recall))
print("f1 :" + str(f1))

print("adjustment factor: " + str(factor))

