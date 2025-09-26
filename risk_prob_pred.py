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
    args.add_argument('--file_path', type=str, help='Path to the excel file', default='data/clean_children.csv')
    args.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file', default='ckpt/children')
    args.add_argument('--output_path', type=str, help='Path to save the plot', default=None)
    args.add_argument('--disable_top10', default=False, action='store_true', help='Whether to use top 10 features')
    args.add_argument('--predict_both', action='store_true', help='Predict risk for both top-10 and full feature models', default=False)
    args.add_argument('--scaler_path', type=str, default=None, help='Path to previously saved scaler bundle (.pkl) to reuse statistics')

    args = args.parse_args()

    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    file_path = args.file_path
    # df = pd.read_excel(file_path)
    df = pd.read_csv(file_path, encoding='utf-8')

    #drop name 
    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if 'Name' in df.columns:
        df = df.drop(columns=['Name'])

    Y = df['School Withdrawal/ Reentry Status']
    file_name = os.path.basename(file_path).split('.')[0]

    # Determine which feature configurations to predict
    if args.predict_both:
        feature_configs = [False, True]  # [top10, full_features]
        print("Predicting risk for both top-10 and full feature models")
    else:
        feature_configs = [args.disable_top10]
        config_name = "full features" if args.disable_top10 else "top-10 features"
        print(f"Predicting risk for {config_name} model only")

    # Process each feature configuration
    for disable_top10 in feature_configs:
        feature_type = "full features" if disable_top10 else "top-10 features"
        print(f"\n{'='*50}")
        print(f"Processing {feature_type}")
        print(f"{'='*50}")

        X = df.drop(columns=['School Withdrawal/ Reentry Status'])

        if not disable_top10:
            if 'teens' in file_path:
                top10_features = top10_features_teens
            elif 'children' in file_path:
                top10_features = top10_features_children
            elif 'adults' in file_path:
                top10_features = top10_features_adults

            X = X[top10_features]
            print('Using top 10 features:', top10_features)        
            
            output_subdir = 'top10'
            if not args.output_path:
                current_output_path = os.path.join('risk_prob', output_subdir)
            else:
                current_output_path = os.path.join(args.output_path, output_subdir)
            os.makedirs(current_output_path, exist_ok=True)
        else:
            print(f'Using all features ({len(X.columns)} features)')
            output_subdir = 'full'
            if not args.output_path:
                current_output_path = os.path.join('risk_prob', output_subdir)
            else:
                current_output_path = os.path.join(args.output_path, output_subdir)
            os.makedirs(current_output_path, exist_ok=True)

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
            ##auto-detect scaler under the same directory as ckpt_path
            scaler_suffix = '_top10' if not disable_top10 else ''
            scaler_path_auto = os.path.join(args.ckpt_path, f'{file_name}_scaler{scaler_suffix}.pkl')
            if os.path.isfile(scaler_path_auto):
                print(f'Auto-detected existing scaler at {scaler_path_auto}. Reusing it.')
                scaler_bundle = joblib.load(scaler_path_auto)
                scaler = scaler_bundle['scaler']
                saved_scale_cols = scaler_bundle.get('scale_cols', scale_cols)
                cols_to_transform = [c for c in saved_scale_cols if c in X.columns]
                X_scaled_part = scaler.transform(X[cols_to_transform])
                X_scaled_df = X.copy()
                X_scaled_df[cols_to_transform] = X_scaled_part
                X = X_scaled_df
            else:
                print('No existing scaler found. Fitting a new one.')
                # Fit new scaler (will not save here; done in training script)
                scaler = StandardScaler()
                X_scaled_part = scaler.fit_transform(X[scale_cols])
                X_scaled_df = X.copy()
                X_scaled_df[scale_cols] = X_scaled_part
                X = X_scaled_df

        five_fold_pred_LR = []
        five_fold_pred_RF = []
        all_ckpts = os.listdir(args.ckpt_path)
        
        #filter checkpoints based on feature type
        if disable_top10:
            all_ckpts = [ckpt for ckpt in all_ckpts if 'top10' not in ckpt and ckpt.endswith('.pkl') and 'scaler' not in ckpt]
        else:
            all_ckpts = [ckpt for ckpt in all_ckpts if 'top10' in ckpt and ckpt.endswith('.pkl') and 'scaler' not in ckpt]

        print(f"Found {len(all_ckpts)} model checkpoints for {feature_type}")

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
        df_output = df.copy()
        
        if len(five_fold_pred_LR) > 0:
            y_pred_LR = np.mean(five_fold_pred_LR, axis=0)
            df_output.insert(1, 'LogisticRegression', y_pred_LR)
            print(f"Averaged predictions from {len(five_fold_pred_LR)} LogisticRegression models")
        else:
            print("No LogisticRegression models found")
            
        if len(five_fold_pred_RF) > 0:
            y_pred_RF = np.mean(five_fold_pred_RF, axis=0)
            df_output.insert(1, 'RandomForest', y_pred_RF)
            print(f"Averaged predictions from {len(five_fold_pred_RF)} RandomForest models")
        else:
            print("No RandomForest models found")

        # Save results
        output_filename = f'{file_name}_risk_prob_{output_subdir}.xlsx'
        output_filepath = os.path.join(current_output_path, output_filename)
        df_output.to_excel(output_filepath, index=False)
        print(f"Results saved to: {output_filepath}")
 

