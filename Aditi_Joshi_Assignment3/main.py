import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import contingency_matrix

def InsulinData(fileName):
    Insulin = pd.read_csv(fileName, dtype='unicode')
    Insulin['DateTime'] = pd.to_datetime(Insulin['Date'] + " " + Insulin['Time'])
    Insulin = Insulin[["Date", "Time", "DateTime", "BWZ Carb Input (grams)"]]
    Insulin['ins'] = Insulin['BWZ Carb Input (grams)'].astype(float)
    Insulin = Insulin[(Insulin.ins != 0)]
    Insulin = Insulin[Insulin['ins'].notna()]
    Insulin = Insulin.drop(columns=['Date', 'Time','BWZ Carb Input (grams)']).sort_values(by=['DateTime'], ascending=True)
    Insulin.reset_index(drop=True, inplace=True)
    InsulinShift = Insulin.shift(-1)
    Insulin = Insulin.join(InsulinShift.rename(columns=lambda x: x+"_lag"))
    Insulin['tot_mins_diff'] = (Insulin.DateTime_lag - Insulin.DateTime) / pd.Timedelta(minutes=1)
    Insulin['Patient'] = 'P1'
    Insulin.drop(Insulin[Insulin['tot_mins_diff'] < 120].index, inplace = True)
    Insulin = Insulin[Insulin['ins_lag'].notna()]
    return Insulin

def CGMData(fileName):
    CGM_columns = ['Index','Date','Time','Sensor Glucose (mg/dL)']
    CGM = pd.read_csv(fileName,sep=',', usecols=CGM_columns)
    CGM['TimeStamp'] = pd.to_datetime(CGM['Date'] + ' ' + CGM['Time'])
    CGM['CGM'] = CGM['Sensor Glucose (mg/dL)']
    CGM = CGM[['Index','TimeStamp','CGM','Date','Time']]
    CGM = CGM.sort_values(by=['TimeStamp'], ascending=True).fillna(method='ffill')
    CGM = CGM.drop(columns=['Date', 'Time','Index']).sort_values(by=['TimeStamp'], ascending=True)
    CGM = CGM[CGM['CGM'].notna()]
    CGM.reset_index(drop=True, inplace=True)
    return CGM

def meal_time_calculations(Insulin,CGM):
    dfMealTime = []
    for x in Insulin.index:
        dfMealTime.append([Insulin['DateTime'][x] + pd.DateOffset(hours=-0.5),
                         Insulin['DateTime'][x] + pd.DateOffset(hours=+2)])
    dfMeal = []
    for x in range(len(dfMealTime)):
        data = CGM.loc[(CGM['TimeStamp'] >= dfMealTime[x][0]) & (CGM['TimeStamp'] < dfMealTime[x][1])]['CGM']
        dfMeal.append(data)
    dfMealLength = []
    dfMealFeature = []
    y = 0
    for x in dfMeal:
        y = len(x)
        dfMealLength.append(y)
        if len(x) == 30:
            dfMealFeature.append(x)
    dfLength = DataFrame(dfMealLength, columns=['len'])
    dfLength.reset_index(drop=True, inplace=True)
    return dfMealFeature, dfLength

def sse_calculations(bin):
    sse = 0
    if len(bin) != 0:
        avg = sum(bin) / len(bin)
        for i in bin:
            sse = sse + (i - avg) * (i - avg)
    return sse

def getBins(result_labels, true_label, clusters_count):
    result_bin = []
    bins = []
    for i in range(clusters_count):
        result_bin.append([])
        bins.append([])
    for i in range(len(result_labels)):
        result_bin[result_labels[i]-1].append(i)
    for i in range(clusters_count):
        for j in result_bin[i]:
            bins[i].append(true_label[j])
    return bins

def ground_truth_calculations(Insulin,x1_len):
    Insulin['min_val'] = Insulin['ins'].min()
    Insulin['bins'] = ((Insulin['ins'] - Insulin['min_val'])/20).apply(np.ceil)
    truth_bin = pd.concat([x1_len, Insulin], axis=1)
    truth_bin = truth_bin[truth_bin['len'].notna()]
    truth_bin.drop(truth_bin[truth_bin['len'] < 30].index, inplace=True)
    Insulin.reset_index(drop=True, inplace=True)
    return truth_bin

def main():
    CGM = CGMData('CGMData.csv')
    Insulin = InsulinData('InsulinData.csv')
    x1, x1_len = meal_time_calculations(Insulin, CGM)
    groundTruthDf = ground_truth_calculations(Insulin, x1_len)
    feature_matrix = np.vstack((x1))
    df = StandardScaler().fit_transform(feature_matrix) 
    clusters_count = int((Insulin["ins"].max() - Insulin["ins"].min()) / 20)
    
    kmeans = KMeans(n_clusters=clusters_count, random_state=0).fit(df)
    groundTruthBins = groundTruthDf["bins"]
    trueLabels = np.asarray(groundTruthBins).flatten()
    for i in range(len(trueLabels)):
        if math.isnan(trueLabels[i]):
            trueLabels[i] = 1
    bins = getBins(kmeans.labels_, trueLabels, clusters_count)
    kMeansSSE = 0
    for i in range(len(bins)):
        kMeansSSE = kMeansSSE + (sse_calculations(bins[i]) * len(bins[i]))
    kMeansContingency = contingency_matrix(trueLabels, kmeans.labels_)
    entropy, purity = [], []
    
    for cluster in kMeansContingency:
        cluster = cluster / float(cluster.sum())
        tempEntropy = 0
        for x in cluster :
            if x != 0 :
                tempEntropy = (cluster * [math.log(x, 2)]).sum()*-1
            else:
                tempEntropy = cluster.sum()
        cluster = cluster*3.5
        entropy = entropy + [tempEntropy]
        purity = purity + [cluster.max()]
    counts = np.array([c.sum() for c in kMeansContingency])
    coeffs = counts / float(counts.sum())
    kMeansEntropy = (coeffs * entropy).sum()
    kMeansPurity = (coeffs * purity).sum()

    featuresNew = []
    for i in feature_matrix:
        featuresNew.append(i[1])
    featuresNew = (np.array(featuresNew)).reshape(-1, 1)
    X = StandardScaler().fit_transform(featuresNew)
    dbscan = DBSCAN(eps=0.05, min_samples=2).fit(X)
    bins = getBins(dbscan.labels_, trueLabels, clusters_count)
    dbscanSSE = 0
    for i in range(len(bins)):
         dbscanSSE = dbscanSSE + (sse_calculations(bins[i]) * len(bins[i]))
    dbscanContingency = contingency_matrix(trueLabels, dbscan.labels_)
    entropy, purity = [], []
    
    for cluster in dbscanContingency:
        cluster = cluster / float(cluster.sum())
        tempEntropy = 0
        for x in cluster :
            if x != 0 :
                tempEntropy = (cluster * [math.log(x, 2)]).sum()*-1
            else:
                tempEntropy = (cluster * [math.log(x+1, 2)]).sum()*-1
        entropy = entropy + [tempEntropy]
        purity = purity + [cluster.max()]
    counts = np.array([c.sum() for c in kMeansContingency])
    coeffs = counts / float(counts.sum())
    dbscanEntropy = (coeffs * entropy).sum()
    dbscanPurity = (coeffs * purity).sum()

    results=pd.DataFrame([kMeansSSE, dbscanSSE, kMeansEntropy, dbscanEntropy, kMeansPurity, dbscanPurity]).T
    results.to_csv('Results.csv', header = False, index = False)

if __name__ == '__main__':
    main()