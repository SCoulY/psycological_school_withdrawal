import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from preprocess.ch2en import column_name2eng
from sklearn.calibration import CalibratedClassifierCV

plt.rcParams['font.family'] = 'SimHei'
# print(plt.style.available)

'''Empirical top 10 features from the analysis'''
top10_features_adults = ['HEI_TS', 'SCL-90 DEP', 'CSES_TS', 'SCL-90 PSY', 'SCL-90 ANX', 'EMBU-F EW', 'DES-Ⅱ_TS', 'DES-Ⅱ_ABS', 'DES-Ⅱ_AMN', 'EMBU-M FS']

top10_features_teens = ['SCL-90 DEP', 'A-DES-Ⅱ_PI', 'SCL-90 ANX', 'CSES_TS', 'EMBU-F EW', 'HEI_TS', 'SCL-90 PSY', 'A-DES-Ⅱ_DPDR', 'A-DES-Ⅱ_AII', 'EMBU-M FS']

top10_features_children = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_PI', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'EMBU-F EW', 'EMBU-F PUN', 'EMBU-M FS', 'CSQ_REP', 'A-SSRS_OS']

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file_path', type=str, help='Path to the excel file', default='C:/Users/SCoulY/Desktop/psycology/data/clean_teens.csv')
    args.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file', default='C:/Users/SCoulY/Desktop/psycology/ckpt_5runs/children_correct/full/teens')
    args.add_argument('--output_path', type=str, help='Path to save the plot', default='C:/Users/SCoulY/Desktop/psycology/data/test_RF_confidence')
    args.add_argument('--disable_top10', default=False, action='store_true', help='Whether to use top 10 features')

    args = args.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    
    file_path = args.file_path
    # df = pd.read_excel(file_path)
    df = pd.read_csv(file_path, encoding='utf-8')
        #drop unnamed column
    df = df.drop(df.columns[0], axis=1)
    #drop name 
    if '姓名' in df.columns:
        name_col = df['姓名']
        df = df.drop(columns=['姓名'])
    X = column_name2eng(df)
    Y = df['School Withdrawal/ Reentry Status']
    df = df.drop(columns=['School Withdrawal/ Reentry Status'])

    if not args.disable_top10:
        if 'teen' in file_path and 'wo_scl' not in file_path:
            top10_features = top10_features_teens
        elif 'wo_scl' in file_path:
            top10_features = top10_features_children
        elif 'adults' in file_path:
            top10_features = top10_features_adults
        X = X[top10_features]
        print('Using top 10 features:', top10_features)


    Y = Y.replace('休学', 0).replace('复学', 1)

    ### convert Chinese to English
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    file_name = os.path.basename(file_path).split('.')[0]


    for ckpt in os.listdir(args.ckpt_path):
        if '.pkl' not in ckpt or 'SVM' in ckpt:
            continue
        model_name = os.path.basename(ckpt).split('.pkl')[0]

        clf = joblib.load(os.path.join(args.ckpt_path, ckpt))

        y_pred = clf.predict_proba(X)

        ### insert y_pred to the second column of dataframe
        df.insert(1, model_name, y_pred[:, 1])

    if 'name_col' in locals():
        df.insert(1, 'name', name_col) 
        df.insert(2, 'status', Y)
    else:
        df.insert(1, 'status', Y)
        
    df.to_excel(os.path.join(args.output_path, f'{file_name}_RF_models_risk_prob.xlsx'), index=False)
 

