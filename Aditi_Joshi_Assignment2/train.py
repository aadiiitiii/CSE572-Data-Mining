#!/usr/bin/env python
# coding: utf-8

# In[23]:


#import libraries
import pandas as pd 
import numpy as np
from datetime import timedelta
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pickle
import pickle_compat

pickle_compat.patch()


# In[24]:


def read_CGM_data(filename):
    CGM = pd.read_csv(filename, usecols=["Date", "Time", "Sensor Glucose (mg/dL)"],low_memory=False)
    CGM['DateTime']= pd.to_datetime(CGM['Date']+' '+CGM['Time'])
    CGM = CGM.replace('',np.nan)
    CGM = CGM.replace('NaN',np.nan)
    CGM = CGM.dropna()
    CGM.reset_index(drop=True, inplace=True)
    CGM = CGM[CGM['Sensor Glucose (mg/dL)'].notna()]
    
    return CGM


# In[25]:


CGM1 = read_CGM_data("CGMData.csv")
# print(CGM1) #55343 rows - before, 51175 rows - later


# In[26]:


CGM2 = read_CGM_data("CGM_patient2.csv")
# print(CGM1) #33248 rows - before, 31289 rows - later


# In[27]:


def read_Insulin_data(filename):
    InsulinData=pd.read_csv(filename, usecols=["Date", "Time", "BWZ Carb Input (grams)"],low_memory=False)
    InsulinData['DateTime']= pd.to_datetime(InsulinData['Date']+' '+InsulinData['Time'])
    
    InsulinData=InsulinData.dropna()
    InsulinData.drop(InsulinData[InsulinData["BWZ Carb Input (grams)"]==0].index, inplace = True)
    InsulinData=InsulinData.reset_index(drop = True) 

    return InsulinData


# In[28]:


Insulin1 = read_Insulin_data("InsulinData.csv")
# print(Insulin1) #41435 rows - before, 747 rows - later


# In[29]:


Insulin2 = read_Insulin_data("Insulin_patient2.csv")
# print(Insulin2) #23185 rows - before, 424 rows - later


# In[30]:


def create_meal_data(insulin_data,cgm_data,dateidentifier):
    insulin_list=insulin_data.copy()
    insulin_list=insulin_list.set_index('DateTime')
    timestamp_within_range=insulin_list.sort_values(by='DateTime',ascending=True).dropna().reset_index()
    timestamp_within_range['BWZ Carb Input (grams)'].replace(0.0,np.nan,inplace=True)
    timestamp_within_range=timestamp_within_range.dropna()
    timestamp_within_range=timestamp_within_range.reset_index().drop(columns='index')
    valid_timestamp_list=[]
    value=0
    for idx,i in enumerate(timestamp_within_range['DateTime']):
        try:
            value=(timestamp_within_range['DateTime'][idx+1]-i).seconds / 60.0
            if value >= 120:
                valid_timestamp_list.append(i)
        except KeyError:
            break
    
    list1=[]
    if dateidentifier==1:
        for idx,i in enumerate(valid_timestamp_list):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=90))
            get_date=i.date().strftime('%-m/%-d/%Y')
            list1.append(cgm_data.loc[cgm_data['Date']==get_date].set_index('DateTime').between_time(start_time=start.strftime('%-H:%-M:%-S'),end_time=end.strftime('%-H:%-M:%-S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(list1)
    else:
        for idx,i in enumerate(valid_timestamp_list):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=90))
            get_date=i.date().strftime('%Y-%m-%d')
            list1.append(cgm_data.loc[cgm_data['Date']==get_date].set_index('DateTime').between_time(start_time=start.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(list1)


# In[31]:


meal_data=create_meal_data(Insulin1,CGM1,1)
meal_data1=create_meal_data(Insulin2,CGM2,2)
meal_data=meal_data.iloc[:,0:24]
meal_data1=meal_data1.iloc[:,0:24]


# In[32]:


def create_no_meal_data(insulin_data,cgm_data):
    insulin_no_meal_df=insulin_data.copy()
    test1_df=insulin_no_meal_df.sort_values(by='DateTime',ascending=True).replace(0.0,np.nan).dropna().copy()
    test1_df=test1_df.reset_index().drop(columns='index')
    valid_timestamp=[]
    for idx,i in enumerate(test1_df['DateTime']):
        try:
            value=(test1_df['DateTime'][idx+1]-i).seconds//3600
            if value >=4:
                valid_timestamp.append(i)
        except KeyError:
            break
    dataset=[]
    for idx, i in enumerate(valid_timestamp):
        iteration_dataset=1
        try:
            length_of_24_dataset=len(cgm_data.loc[(cgm_data['DateTime']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_data['DateTime']<valid_timestamp[idx+1])])//24
            while (iteration_dataset<=length_of_24_dataset):
                if iteration_dataset==1:
                    dataset.append(cgm_data.loc[(cgm_data['DateTime']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_data['DateTime']<valid_timestamp[idx+1])]['Sensor Glucose (mg/dL)'][:iteration_dataset*24].values.tolist())
                    iteration_dataset+=1
                else:
                    dataset.append(cgm_data.loc[(cgm_data['DateTime']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_data['DateTime']<valid_timestamp[idx+1])]['Sensor Glucose (mg/dL)'][(iteration_dataset-1)*24:(iteration_dataset)*24].values.tolist())
                    iteration_dataset+=1
        except IndexError:
            break
    return pd.DataFrame(dataset)


# In[33]:


no_meal_data=create_no_meal_data(Insulin1,CGM1)
no_meal_data1=create_no_meal_data(Insulin2,CGM2)


# In[34]:


def handleMissingData(data):
    for row in data.itertuples():
        if pd.Series(list(row)).isnull().sum() / len(row) > 0.2:
            data = data.drop(index=row[0])
    return data


# In[35]:


meal1 = handleMissingData(meal_data)
meal2 = handleMissingData(meal_data1)
noMeal1 = handleMissingData(no_meal_data)
noMeal2 = handleMissingData(no_meal_data1)


# In[36]:


meal1["label"] = 1
meal2["label"] = 1
noMeal1["label"] = 0
noMeal2["label"] = 0


# In[37]:


meal = pd.concat([meal1, meal2], axis=0).reset_index()
noMeal = pd.concat([noMeal1, noMeal2], axis=0).reset_index()


# In[38]:


def getPeak(row):
    data = [x for x in list(row) if type(x) != str and x > 0]
    maxData = max(data)
    return maxData


def getStd(row):
    data = [x for x in list(row) if type(x) != str and x > 0]
    std = np.mat(data).std()
    return std


def getGN(row):
    meal = row[5]
    peak = getPeak(row)
    gn = (peak - meal) / meal
    return gn


def fft(row):
    data = [x for x in list(row) if type(x) != str and x > 0]
    fftArray = np.fft.fft(data)
    freqArray = np.fft.fftfreq(len(data))
    fftDict = dict(zip(fftArray, freqArray))
    fftArray = sorted(fftArray)
    fft1 = fftArray[-3]
    fft2 = fftArray[-5]
    freq1 = fftDict.get(fft1)
    freq2 = fftDict.get(fft2)
    return [abs(fft1), abs(freq1), abs(fft2), abs(freq2)]


def fft1(row):
    fft1, fft2, fft3, fft4 = fft(row)
    return fft1


def fft2(row):
    fft1, fft2, fft3, fft4 = fft(row)
    return fft2


def fft3(row):
    fft1, fft2, fft3, fft4 = fft(row)
    return fft3


def fft4(row):
    fft1, fft2, fft3, fft4 = fft(row)
    return fft4

def getGT(row):
    index = row.index[1:-1]
    data = [x for x in list(row)[1:-1] if type(x) != str and x > 0]
    Map = dict(zip(row.values[1:-1], index))
    peak = max(data)
    peakLoc = Map.get(peak)
    mealLoc = Map.get(row[5])
    if peakLoc == 5:
        return 0
    else:
        GT = (peak - row[5]) / (abs(peakLoc - 5) * 5 * 60)
        return GT

def getMaxTime(row):
    mealMaxIndex = max([x for x in list(row) if type(x) != str and x > 0])
    mealMinIndex = min([x for x in list(row) if type(x) != str and x > 0])
    return mealMaxIndex - mealMinIndex

def getDerivative(row):
    derivative = (row.diff() / 5).max()
    return derivative

def getRollingMean(row):
    rolling_mean=row.rolling(window=5).mean().mean()
    return rolling_mean


# In[39]:


def featureExtractor(data):
    peak = data.apply(lambda row: getPeak(row), axis=1)
    std = data.apply(lambda row: getStd(row), axis=1)
    GN = data.apply(lambda row: getGN(row), axis=1)
    Fft1 = data.apply(lambda row: fft1(row), axis=1)
    Fft2 = data.apply(lambda row: fft2(row), axis=1)
    Fft3 = data.apply(lambda row: fft3(row), axis=1)
    Fft4 = data.apply(lambda row: fft4(row), axis=1)
    GT = data.apply(lambda row: getGT(row), axis=1)
    maxtime = data.apply(lambda row: getMaxTime(row), axis=1)
    derivative = data.apply(lambda row: getDerivative(row), axis=1)
    rolling_mean = data.apply(lambda row: getRollingMean(row), axis=1)
    return pd.DataFrame({"peak": peak, "std": std, "GN": GN, "fft1": Fft1, "fft2": Fft2, "fft3": Fft3, "fft4": Fft4, "GT": GT, "max_time": maxtime, "derivative": derivative, "rolling_mean": rolling_mean})


# In[40]:


mealFeature = featureExtractor(meal)
noMealFeature = featureExtractor(noMeal)


# In[41]:


mealFeature["label"] = meal.label
noMealFeature["label"] = noMeal.label


# In[42]:


combData = pd.concat([mealFeature, noMealFeature])
combineData = combData.sample(frac=1)
trainData = combineData.iloc[0:int(round(len(combineData.index) * 0.8))]
testData = combineData.iloc[int(round(len(combineData.index) * 0.8)):]


# In[43]:


x_train = trainData
x_test = testData
y_train = x_train.label
y_test = x_test.label
x_train = x_train.iloc[:, :-1]
x_test = x_test.iloc[:, :-1]


# In[44]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
clf = SVC(kernel='rbf', gamma=5, C=40).fit(X_train_scaled, y_train)
with open('model.pkl', 'wb') as dump_var:
    pickle.dump(clf, dump_var)

