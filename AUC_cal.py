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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from preprocess.ch2en import column_name2eng
from sklearn.preprocessing import LabelEncoder

plt.rcParams['font.family'] = 'SimHei'
# print(plt.style.available)

'''Empirical top 10 features from the analysis'''
top10_features = ['华西心晴指数-总分', 'SCl-90 精神病性','SCl-90 抑郁', 'SCl-90 焦虑', '父教-情感温暖、理解', '母教-偏爱被试', '核心自我评价-总分', '分离-总分', '分离-遗忘性分离', '分离-专注与想象性参与']

top10_features_teen = ['华西心晴指数-总分', 'SCl-90 精神病性','SCl-90 抑郁', 'SCl-90 焦虑', '父教-情感温暖、理解', '母教-偏爱被试', '核心自我评价-总分', '青分离-专注与想象性投入', '青分离-被动影响', '青分离-现实解体与人格解体']

top10_features_teen_wo_scl = ['华西心晴指数-总分', '父教-情感温暖、理解', '父教-惩罚、严厉', '母教-惩罚、严厉', '父教-偏爱被试', '母教-偏爱被试', '核心自我评价-总分', '青分离-专注与想象性投入', '青分离-被动影响', '青分离-现实解体与人格解体']

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file_path', type=str, help='Path to the excel file', default='C:/Users/SCoulY/Desktop/psycology/data/clean_teens_wo_scl.csv')
    args.add_argument('--ckpt_path', nargs='+', type=str, help='Path to the checkpoint file', default=['C:/Users/SCoulY/Desktop/psycology/ckpt/clean_adults_LogisticRegression_acc_0.89.pkl', 'C:/Users/SCoulY/Desktop/psycology/ckpt/clean_adults_RandomForest_acc_0.89.pkl'])
    args.add_argument('--output_path', type=str, help='Path to save the plot', default='C:/Users/SCoulY/Desktop/psycology/data/risk_all')
    args.add_argument('--disable_top10', action='store_true', help='Disable top 10 features for building the model', default=False)

    args = args.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    
    file_path = args.file_path
    # df = pd.read_excel(file_path)
    df = pd.read_csv(file_path, encoding='utf-8')

    #drop unnamed column if exists
    if df.columns[0].startswith('Unnamed'):
        # print(f'Dropping unnamed column: {df.columns[0]}')
        df = df.drop(df.columns[0], axis=1)
    #drop name
    if '姓名' in df.columns:
        df = df.drop(columns=['姓名'])

    ### get features and labels
    if args.disable_top10:
        feat = df.drop(columns=['状态'])
    else:
        if 'teen' in args.file_path and 'wo_scl' not in args.file_path:
            top10_features = top10_features_teen
        if 'children' in args.file_path or 'wo_scl' in args.file_path:
            top10_features = top10_features_teen_wo_scl
        feat = df[top10_features]
    label = df['状态']

    ### Scale the features
    scaler = StandardScaler()
    feat = scaler.fit_transform(feat)
    label = LabelEncoder().fit_transform(label)

    X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=42)

    file_name = os.path.basename(file_path).split('.')[0]

    for ckpt in args.ckpt_path:
        if '.pkl' not in ckpt:
            exit('Please provide a valid checkpoint file.')

        model_name = ckpt.split('acc')[0][:-1].split('_')[-1] #LogisticRegression
        
        clf = joblib.load(os.path.normpath(ckpt))
    
        model_name = os.path.basename(ckpt)[:-4]
        if 'SVM' in model_name:
            y_pred = clf.decision_function(X_test)
            auc = roc_auc_score(1-y_test, 1-y_pred)
        else:
            y_pred = clf.predict_proba(X_test)
            auc = roc_auc_score(1-y_test, y_pred[:, 0])
        print(f'{model_name} AUC: {auc:.4f}')
