import pickle
import pandas as pd
def model_prediction(test_df_one_enc,df_test, model_name, csv_file_name, save_file):
    
    loaded_model = pickle.load(open('../models/at1_week4_model/'+str(model_name)+'.sav', 'rb'))
    result = pd.DataFrame(loaded_model.predict_proba(test_df_one_enc))[1]
    df_test['drafted'] = result
    print(df_test[['player_id', 'drafted']])
    if save_file == True:
        df_test[['player_id', 'drafted']].to_csv('../models/scores_week_4/'+str(csv_file_name)+'.csv', index=False)
        print("csv saved") 
