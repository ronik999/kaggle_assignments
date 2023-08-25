import pandas as pd
import numpy as np

def map_months(date):
    month_dict = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    split_value = date.split('-')
    if len(split_value)==2:
        if (split_value[0] in month_dict):
            feet = month_dict.get(split_value[0],1)
            inch = 0
            return feet, inch
            
        elif(split_value[1] in month_dict):
            feet = month_dict.get(split_value[1],1)
            inch = split_value[0]
            return feet, inch
        
        else:
            return 0, 0
    else:
        return 0, 0
    
        
def convert_height_to_cm(df):
    ft, inch = map_months(df)
    cm = (int(ft) * 30.48) + (int(inch) * 2.54) 
    if cm == 0:
        return np.nan
    return cm