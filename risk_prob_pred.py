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
    args.add_argument('--file_path', type=str, help='Path to the excel file', default='C:/Users/SCoulY/Desktop/psycology/data/clean_children.csv')
    args.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file', default='C:/Users/SCoulY/Desktop/psycology/ckpt_w_scaler/children_correct/children')
    args.add_argument('--output_path', type=str, help='Path to save the plot', default='C:/Users/SCoulY/Desktop/psycology/data/risk_uncertainty_top10_children_correct/top10')
    args.add_argument('--disable_top10', default=False, action='store_true', help='Whether to use top 10 features')
    args.add_argument('--scaler_path', type=str, default='C:/Users/SCoulY/Desktop/psycology/ckpt_w_scaler/children_correct/children/clean_children_scaler_top10.pkl', help='Path to previously saved scaler bundle (.pkl) to reuse statistics')

    args = args.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    
    file_path = args.file_path
    # df = pd.read_excel(file_path)
    df = pd.read_csv(file_path, encoding='utf-8')

    #drop name 
    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])


    Y = df['School Withdrawal/ Reentry Status']
    X = df.drop(columns=['School Withdrawal/ Reentry Status'])

 
    if not args.disable_top10:
        if 'teens' in file_path:
            top10_features = top10_features_teens
        elif 'children' in file_path:
            top10_features = top10_features_children
        elif 'adults' in file_path:
            top10_features = top10_features_adults
        X = X[top10_features]
        print('Using top 10 features:', top10_features)


    exclude_cols = ['Gender']
    scale_cols = [c for c in X.columns if c not in exclude_cols]

    if args.scaler_path and os.path.isfile(args.scaler_path):
        # Reuse existing scaler statistics
        scaler_bundle = joblib.load(args.scaler_path)
        scaler = scaler_bundle['scaler']
        saved_scale_cols = scaler_bundle.get('scale_cols', scale_cols)
        # Align columns (only transform intersection to avoid KeyErrors)
        cols_to_transform = [c for c in saved_scale_cols if c in X.columns]
        X_scaled_part = scaler.transform(X[cols_to_transform])
        X_scaled_df = X.copy()
        X_scaled_df[cols_to_transform] = X_scaled_part
        X = X_scaled_df
    else:
        # Fit new scaler (will not save here; done in training script)
        scaler = StandardScaler()
        X_scaled_part = scaler.fit_transform(X[scale_cols])
        X_scaled_df = X.copy()
        X_scaled_df[scale_cols] = X_scaled_part
        X = X_scaled_df


    file_name = os.path.basename(file_path).split('.')[0]

    five_fold_pred_LR = []
    five_fold_pred_RF = []
    all_ckpts = os.listdir(args.ckpt_path)
    #filter out top10 using args.disable_top10
    if args.disable_top10:
        all_ckpts = [ckpt for ckpt in all_ckpts if 'top10' not in ckpt]
    else:
        all_ckpts = [ckpt for ckpt in all_ckpts if 'top10' in ckpt]

    for ckpt in all_ckpts:
        if 'logisticregression' in ckpt.lower() or 'randomforest' in ckpt.lower():
            model_name = os.path.basename(ckpt).split('acc')[0][:-1].split('_')[-1] #LogisticRegression
        else:
            continue

        clf = joblib.load(os.path.join(args.ckpt_path, os.path.normpath(ckpt)))

        y_pred = clf.predict_proba(X)
        if model_name == 'LogisticRegression':
            five_fold_pred_LR.append(y_pred[:, 1])
        elif model_name == 'RandomForest':
            five_fold_pred_RF.append(y_pred[:, 1])
        else:
            print(f'Unknown model: {model_name}')
    
    ### average the predictions and insert to the dataframe
    if len(five_fold_pred_LR) > 0:
        y_pred_LR = np.mean(five_fold_pred_LR, axis=0)
    if len(five_fold_pred_RF) > 0:
        y_pred_RF = np.mean(five_fold_pred_RF, axis=0)

    ### insert y_pred to the second column of dataframe
    df.insert(1, 'LogisticRegression', y_pred_LR)
    df.insert(1, 'RandomForest', y_pred_RF)

    df.insert(1, 'status', Y)
        
    df.to_excel(os.path.join(args.output_path, f'{file_name}_risk_prob.xlsx'), index=False)
 

