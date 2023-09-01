from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Add train data and test data in the encoder
# Encoder is performed only on train data and uses trained encoder in the test data

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
    train_data_enc = train_data_enc.drop(to_encode, axis=1)
    test_data_enc = test_data_enc.drop(to_encode, axis=1)

    return train_data_enc, test_data_enc