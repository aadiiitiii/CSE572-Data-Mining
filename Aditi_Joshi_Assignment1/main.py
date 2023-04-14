#import libraries
import numpy as np
import pandas as pd
import csv

#interpolate the values in a linear format in Sensor Glucose (mg/dL) column
def interpolate_data(data):
    data['Sensor Glucose (mg/dL)'].interpolate(method='ffill', inplace=True)

#calculate the percentage for each day's data
def calculate_percentage(data, count_days):
    total_days = 288*count_days
    
    hyperglycemia = data[data['Sensor Glucose (mg/dL)']>180.0].count()
    hyperglycemia_percent = (hyperglycemia/total_days)*100.0

    hyperglycemia_critical = data[data['Sensor Glucose (mg/dL)']>250].count()
    hyperglycemia_critical_percent = (hyperglycemia_critical/total_days)*100.0

    lowRange = data[(data['Sensor Glucose (mg/dL)']>=70)&(data['Sensor Glucose (mg/dL)']<=180)].count()
    lowRange_percent = (lowRange/total_days)*100.0

    secondaryRange = data[(data['Sensor Glucose (mg/dL)']>=70)&(data['Sensor Glucose (mg/dL)']<=150)].count()
    secondaryRange_percent = (secondaryRange/total_days)*100.0

    hypoglycemiaLevel1 = data[data['Sensor Glucose (mg/dL)']<70].count()
    hypoglycemiaLevel1_percent = (hypoglycemiaLevel1/total_days)*100.0

    hypoglycemiaLevel2 = data[data['Sensor Glucose (mg/dL)']<54].count()
    hypoglycemiaLevel2_percent = (hypoglycemiaLevel2/total_days)*100.0

    return (hyperglycemia_percent['Date'], hyperglycemia_critical_percent['Date'], lowRange_percent['Date'], secondaryRange_percent['Date'], hypoglycemiaLevel1_percent['Date'], hypoglycemiaLevel2_percent['Date'])

#main function
def main() :

    #import csv files and selected columns
    CGMData = pd.read_csv('CGMData.csv',usecols=['Index','Date','Time','Sensor Glucose (mg/dL)','ISIG Value'],index_col=0,low_memory=False)
    InsulinData = pd.read_csv('InsulinData.csv',usecols=['Index','Date','Time','Alarm'],index_col=0,low_memory=False)

    #sort the data in ascending order and add DateTime column
    CGMData['DateTime'] = pd.to_datetime(CGMData['Date'].apply(str)+' '+CGMData['Time'])
    CGMData = CGMData.sort_values(by='DateTime',ascending=True)

    InsulinData['DateTime'] = pd.to_datetime(InsulinData['Date'].apply(str)+' '+InsulinData['Time'])
    InsulinData = InsulinData.sort_values(by='DateTime',ascending=True) 

    #find the position where we get "AUTO MODE ACTIVE PLGM OFF"
    values = InsulinData.loc[InsulinData['Alarm'].str.contains("AUTO MODE ACTIVE PLGM OFF", case=True, na=False)]

    #extract DateTime column where the first occurence of "AUTO MODE ACTIVE PLGM OFF" is found
    automode_datetime= values['DateTime'].iloc[0]

    #split cgmdata into manual mode and auto mode
    CGMDataManualMode = CGMData.loc[CGMData['DateTime'] <= automode_datetime]
    CGMDataAutoMode = CGMData.loc[CGMData['DateTime'] > automode_datetime]

    #interpolate the data in both manual and auto mode
    interpolate_data(CGMDataManualMode)
    interpolate_data(CGMDataAutoMode)

    #differentiate the data in auto and manual dataframes for overnight and daytime
    AutoCGMDataOvernight = CGMDataAutoMode.set_index('DateTime').between_time('00:00', '06:00', include_end=False).reset_index()
    AutoCGMDataDaytime = CGMDataAutoMode.set_index('DateTime').between_time('06:00', '00:00',include_end= False).reset_index()
    ManualCGMDataOvernight = CGMDataManualMode.set_index('DateTime').between_time('00:00', '06:00', include_end=False).reset_index()
    ManualCGMDataDaytime = CGMDataManualMode.set_index('DateTime').between_time('06:00', '00:00', include_end=False).reset_index()

    #calculate the length of the data 
    manual_length = len(CGMDataManualMode['DateTime'].dt.normalize().unique())
    auto_length = len(CGMDataAutoMode['DateTime'].dt.normalize().unique())

    #in each variables get the values from the CGMData dataframe -- MANUAL
    whole_day_manual_hyperglycemia,whole_day_manual_hyperglycemia_critical,whole_day_manual_cgm_range,whole_day_manual_range_secondary,whole_day_manual_hypoglycemia1,whole_day_manual_hypoglycemia2 = calculate_percentage(CGMDataManualMode, manual_length)
    overnight_manual_hyperglycemia, overnight_manual_hyperglycemia_critical, overnight_manual_cgm_range, overnight_manual_range_secondary, overnight_manual_hypoglycemia1, overnight_manual_hypoglycemia2 = calculate_percentage(ManualCGMDataOvernight, manual_length)
    daytime_manual_hyperglycemia,daytime_manual_hyperglycemia_critical,daytime_manual_cgm_range,daytime_manual_range_secondary,daytime_manual_hypoglycemia1,daytime_manual_hypoglycemia2 = calculate_percentage(ManualCGMDataDaytime, manual_length)

    #in each variables get the values from the CGMData dataframe -- AUTO
    whole_day_auto_hyperglycemia,whole_day_auto_hyperglycemia_critical,whole_day_auto_cgm_range,whole_day_auto_range_secondary,whole_day_auto_hypoglycemia1,whole_day_auto_hypoglycemia2 = calculate_percentage(CGMDataAutoMode, auto_length)
    overnight_auto_hyperglycemia,overnight_auto_hyperglycemia_critical,overnight_auto_cgm_range,overnight_auto_range_secondary,overnight_auto_hypoglycemia1,overnight_auto_hypoglycemia2 = calculate_percentage(AutoCGMDataOvernight, auto_length)
    daytime_auto_hyperglycemia,daytime_auto_hyperglycemia_critical,daytime_auto_cgm_range,daytime_auto_range_secondary,daytime_auto_hypoglycemia1,daytime_auto_hypoglycemia2 = calculate_percentage(AutoCGMDataDaytime, auto_length)

    #19th column should be 1.1
    end_string = 1.1
    manualResult = overnight_manual_hyperglycemia, overnight_manual_hyperglycemia_critical, overnight_manual_cgm_range, overnight_manual_range_secondary, overnight_manual_hypoglycemia1, overnight_manual_hypoglycemia2, daytime_manual_hyperglycemia, daytime_manual_hyperglycemia_critical, daytime_manual_cgm_range, daytime_manual_range_secondary, daytime_manual_hypoglycemia1, daytime_manual_hypoglycemia2, whole_day_manual_hyperglycemia, whole_day_manual_hyperglycemia_critical, whole_day_manual_cgm_range, whole_day_manual_range_secondary, whole_day_manual_hypoglycemia1, whole_day_manual_hypoglycemia2, end_string

    autoResult = overnight_auto_hyperglycemia, overnight_auto_hyperglycemia_critical, overnight_auto_cgm_range, overnight_auto_range_secondary, overnight_auto_hypoglycemia1, overnight_auto_hypoglycemia2, daytime_auto_hyperglycemia, daytime_auto_hyperglycemia_critical, daytime_auto_cgm_range, daytime_auto_range_secondary, daytime_auto_hypoglycemia1, daytime_auto_hypoglycemia2, whole_day_auto_hyperglycemia, whole_day_auto_hyperglycemia_critical, whole_day_auto_cgm_range, whole_day_auto_range_secondary, whole_day_auto_hypoglycemia1, whole_day_auto_hypoglycemia2, end_string

    #final result is a combination of both manual and auto rows
    result = pd.DataFrame([manualResult,autoResult])
    result.to_csv("Results.csv",header=None,index=None)

if __name__ == "__main__" :
    main()