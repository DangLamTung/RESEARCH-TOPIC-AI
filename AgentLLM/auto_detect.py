# importing basic libraries 
import pandas as pd 
import numpy as np 

import lightgbm as lgb


# model tunning 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, roc_auc_score

# saving models 
import pickle, joblib

# import warnings
import warnings
warnings.filterwarnings('ignore')
min_val = [-0.39130435, -0.42937853,  0., -0.01790742]
scale =  [2.17391304e-02, 2.69034167e-02, 2.00000000e-01, 1.59620603e-05]



# Load the model
loaded_model = lgb.Booster(model_file="./model_contract/lgbm_model.txt")


def read_insurance_data(file_path):
    data_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            age, sex, bmi, children, smoker, region = line.strip().split(',')
            
            # Convert numerical values to appropriate types
            age = int(age)
            bmi = float(bmi)
            children = int(children)
            smoker = 1 if smoker.lower() == 'yes' else 0  # Convert to 1/0
            
            # Store the values in a list
            data_list.append([age, sex, bmi, children, smoker, region])

    # Convert to DataFrame
    df = pd.DataFrame(data_list, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
    # Encode categorical columns
    df = pd.get_dummies(df, columns=['sex', 'region'], drop_first=False)
    
    # Ensure all region columns exist
    for region_col in ['region_northwest', 'region_southeast', 'region_southwest']:
        if region_col not in df.columns:
            df[region_col] = 0  # Add missing columns with default value 0

    return df

def predict_insurance(X_test):
        
    # Step 1: Define the best model
    charges = loaded_model.predict(X_test)


    # print((charges - min_val[3])/scale[3])
    # Step 3: Predict on the test set
    # y_pred = loaded_model.predict(X_test) # TestData

    y_pred = loaded_model.predict(X_test)


    # print((charges - min_val[3])/scale[3])

    return (charges - min_val[3])/scale[3]



test = read_insurance_data("data_test.txt")
print(test)
predict_insurance(test)