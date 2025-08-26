from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
import argparse
import joblib
from sklearn.metrics import roc_auc_score
from preprocess.ch2en import column_name2eng

'''Empirical top 10 features from the analysis'''
# top10_features = ['华西心晴指数', 'SCL-90强迫症状', 'SCL-90抑郁', 'SCL-90偏执', '父亲教养-情感温暖、理解', '母亲教养-情感温暖、理解', '核心自我评价','应付方式-自责', '应付方式-退避', '分离体验']

top10_features = ['SSRS_TS', 'SSRS_SS','EMBU-F OI', 'EMBU-F PUN', 'EMBU-F REJ', 'EMBU-F OP', 'SSRS_SU', 'SSRS_OS', 'EMBU-M OI', 'EMBU-F EW']

top10_features_teen = ['CSQ_RAT', 'CSQ_REP','CSQ_FAN', 'EMBU-F PUN', 'EMBU-F REJ', 'CSQ_SB', 'EMBU-F OP', 'EMBU-F OI', 'A-SSRS_SU', 'EMBU-M OI']

top10_features_teen_wo_scl = ['EMBU-F REJ', 'EMBU-F OP', 'EMBU-F PUN', 'A-SSRS_SU', 'EMBU-F OI', 'CSQ_HS', 'CSQ_RAT', 'A-SSRS_TS', 'CSQ_REP', 'EMBU-F EW']


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file_path', type=str, help='Path to the excel file', default='C:/Users/SCoulY/Desktop/psycology/data/clean_adults.xlsx')
    args.add_argument('--out_path', type=str, help='Path to the output model file', default='C:/Users/SCoulY/Desktop/psycology/ckpt_unsupervised_top10_5runs')
    args.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file', default=None)
    args.add_argument('--classifier', type=str, help='Classifier to use', default='LogisticRegression', choices=['RandomForest', 'SVM', 'LogisticRegression'])
    args.add_argument('--disable_top10', action='store_true', help='Disable top 10 features for building the model', default=False)
    args = args.parse_args()

    ### read in data
    df = pd.read_excel(args.file_path)
    df = df.drop(df.columns[0], axis=1)
    df = column_name2eng(df)

    table_name = os.path.basename(args.file_path).split('.')[0]

    ### get features and labels
    if args.disable_top10:
        feat = df.drop(columns=['School Withdrawal/ Reentry Status'])
    else:
        if 'teen' in args.file_path and 'wo_scl' not in args.file_path:
            top10_features = top10_features_teen
        if 'wo_scl' in args.file_path:
            top10_features = top10_features_teen_wo_scl
        feat = df[top10_features]


    label = df['School Withdrawal/ Reentry Status']

    ### Scale the features
    scaler = StandardScaler()
    feat = scaler.fit_transform(feat)
    label = LabelEncoder().fit_transform(label)

    seeds = [42, 123, 2025, 6, 255]  # random seeds for reproducibility
    accs = []
    cms = []
    reports = []

    if args.ckpt_path:  # perform prediction and evaluation
        clf = joblib.load(args.ckpt_path)
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=seed)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
            cms.append(confusion_matrix(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
        print(f'Accuracy (mean ± std): {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    else:
        print(f'No checkpoint provided. Training {args.classifier} model...')
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=seed)
            if args.classifier == 'RandomForest':
                clf = RandomForestClassifier()
            elif args.classifier == 'SVM':
                clf = SVC(kernel='rbf', class_weight="balanced", probability=True)
            elif args.classifier == 'LogisticRegression':
                clf = LogisticRegression()
            else:
                raise ValueError(f"Invalid classifier: {args.classifier}")

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
            cms.append(confusion_matrix(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
            # Compute AUC
            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test)[:, 1]
            else:
                # For SVM without probability, use decision_function
                y_score = clf.decision_function(X_test)
            auc = roc_auc_score(y_test, y_score)
            if 'aucs' not in locals():
                aucs = []
            aucs.append(auc)

        print(f'Accuracy (mean ± std): {np.mean(accs):.4f} ± {np.std(accs):.4f}')
        print(f'AUC (mean ± std): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
        # # Optionally print confusion matrix and classification report for the last run
        # print(cms[-1])
        # print(classification_report(y_test, y_pred))

        # Print average precision, recall, and F1 with std
        precisions = [r['macro avg']['precision'] for r in reports]
        recalls = [r['macro avg']['recall'] for r in reports]
        f1s = [r['macro avg']['f1-score'] for r in reports]
        print(f'Precision (mean ± std): {np.mean(precisions):.4f} ± {np.std(precisions):.4f}')
        print(f'Recall (mean ± std): {np.mean(recalls):.4f} ± {np.std(recalls):.4f}')
        print(f'F1-score (mean ± std): {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')

        ### save model from the last run
        # if args.out_path:
        #     os.makedirs(args.out_path, exist_ok=True)
        #     if_top10 = '_top10.pkl' if not args.disable_top10 else '.pkl'
        #     joblib.dump(clf, os.path.join(args.out_path, f'{table_name}_{args.classifier}_acc_{acc:.2f}'+if_top10))
        #     # save the confusion matrix
        #     cm_df = pd.DataFrame(cms[-1], index=['复学', '休学'], columns=['复学', '休学'])
        #     cm_df.to_excel(os.path.join(args.out_path, f'{table_name}_{args.classifier}_confusion_matrix_'+if_top10+'.xlsx'), index=True)