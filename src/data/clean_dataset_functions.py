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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def perform_label_encoding(train_data, test_data, list_col_name):
    for col in list_col_name:
        labelenc = LabelEncoder()
        train_data[col] = labelenc.fit_transform(train_data[col])
        test_data[col] = labelenc.transform(test_data[col])
    return train_data, test_data

def perform_one_hot_encoding(train_data, test_data, to_encode):
    ohe = OneHotEncoder().fit(train_data[to_encode])
    train_ohe = ohe.transform(train_data[to_encode]).toarray()
    encoded_train_data = pd.DataFrame(train_ohe, index=train_data.index, columns=ohe.get_feature_names_out(to_encode))
    train_data_enc = pd.merge(train_data, encoded_train_data, left_index=True, right_index=True)
    test_ohe = ohe.transform(test_data[to_encode]).toarray()
    encoded_test_data = pd.DataFrame(test_ohe, index=test_data.index, columns=ohe.get_feature_names_out(to_encode))
    test_data_enc = pd.merge(test_data, encoded_test_data, left_index=True, right_index=True)
    
    return train_data_enc, test_data_enc