from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import os
import argparse
import joblib
from sklearn.metrics import roc_auc_score

'''Empirical top 10 features from the analysis'''

top10_features_adults = ['HEI_TS', 'SCL-90 DEP', 'CSES_TS', 'SCL-90 PSY', 'SCL-90 ANX', 'EMBU-F EW', 'DES-Ⅱ_TS', 'DES-Ⅱ_ABS', 'DES-Ⅱ_AMN', 'EMBU-M FS']

top10_features_teens = ['SCL-90 DEP', 'A-DES-Ⅱ_PI', 'SCL-90 ANX', 'CSES_TS', 'EMBU-F EW', 'HEI_TS', 'SCL-90 PSY', 'A-DES-Ⅱ_DPDR', 'A-DES-Ⅱ_AII', 'EMBU-M FS']

top10_features_children = ['CSES_TS', 'HEI_TS', 'A-DES-Ⅱ_PI', 'A-DES-Ⅱ_AII', 'A-DES-Ⅱ_DPDR', 'EMBU-F EW', 'EMBU-F PUN', 'EMBU-M FS', 'CSQ_REP', 'A-SSRS_OS']


def train_classifier(classifier, ckpt_path=None, out_path=None, disable_top10=False):
    accs = []
    cms = []
    reports = []
    seeds = [42, 123, 2025, 6, 255]  # random seeds for reproducibility

    if ckpt_path:  # perform prediction and evaluation
        clf = joblib.load(ckpt_path)
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=seed)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
            cms.append(confusion_matrix(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
        print(f'Accuracy (mean ± std): {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    else:
        print(f'No checkpoint provided. Training {classifier} model...')
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=seed)

            if classifier == 'RandomForest':
                clf = RandomForestClassifier(random_state=seed)
            elif classifier == 'SVM':
                clf = SVC(kernel='rbf', probability=True, random_state=seed)
            elif classifier == 'LogisticRegression':
                clf = LogisticRegression(random_state=seed)
            else:
                raise ValueError(f"Invalid classifier: {classifier}")
            
            # clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
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

            ### save model from the every run
            if out_path:
                os.makedirs(out_path, exist_ok=True)
                if_top10 = '_top10.pkl' if not disable_top10 else '.pkl'
                run_str = f'_run_{seed}'
                joblib.dump(clf, os.path.join(out_path, f'{table_name}_{classifier}_acc_{acc:.2f}'+run_str+if_top10))

    return accs, cms, reports, aucs if 'aucs' in locals() else None

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file_path', type=str, help='Path to the excel/csv file', default='data/clean_adults.csv')
    args.add_argument('--out_path', type=str, help='Path to the output model file', default='ckpt/adults')
    args.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file', default=None)
    args.add_argument('--classifier', type=str, nargs='+', help='Classifier to use', default=['SVM', 'RandomForest', 'LogisticRegression'])
    args.add_argument('--disable_top10', action='store_true', help='Disable top 10 features for building the model', default=True)
    args.add_argument('--scaler_path', type=str, help='Path to a saved scaler bundle (joblib) containing scaler and column info', default=None)
    args = args.parse_args()

    ### read in data
    if args.file_path.endswith('.xlsx'):
        df = pd.read_excel(args.file_path)
    else:
        df = pd.read_csv(args.file_path)

    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    table_name = os.path.basename(args.file_path).split('.')[0]

    ### get features and labels
    if args.disable_top10:
        feat = df.drop(columns=['School Withdrawal/ Reentry Status'])
    else:
        if 'adults' in args.file_path:
            top10_features = top10_features_adults
        elif 'teens' in args.file_path:
            top10_features = top10_features_teens
        elif 'children' in args.file_path:
            top10_features = top10_features_children
        else:
            raise ValueError("File path does not match any known dataset.")
        feat = df[top10_features]

    label = df['School Withdrawal/ Reentry Status']

    ### Scale the features
    # scaler = StandardScaler()
    # feat = scaler.fit_transform(feat)

    exclude_cols = ['Gender']
    scale_cols = [c for c in feat.columns if c not in exclude_cols]

    if args.scaler_path and os.path.isfile(args.scaler_path):
        # Load existing scaler bundle
        scaler_bundle = joblib.load(args.scaler_path)
        scaler = scaler_bundle['scaler']
        saved_scale_cols = scaler_bundle.get('scale_cols', scale_cols)
        # Transform only columns present both in feat and saved_scale_cols to be safe
        cols_to_transform = [c for c in saved_scale_cols if c in feat.columns]
        X_scaled_part = scaler.transform(feat[cols_to_transform])
        X_scaled_df = feat.copy()
        X_scaled_df[cols_to_transform] = X_scaled_part
        feat = X_scaled_df
    else:
        # Fit new scaler and save bundle for future reuse
        scaler = StandardScaler()
        X_scaled_part = scaler.fit_transform(feat[scale_cols])
        X_scaled_df = feat.copy()
        X_scaled_df[scale_cols] = X_scaled_part
        feat = X_scaled_df
        if args.out_path:
            os.makedirs(args.out_path, exist_ok=True)
            scaler_bundle = {
                'scaler': scaler,
                'scale_cols': scale_cols,
                'exclude_cols': exclude_cols,
                'feature_order': feat.columns.tolist()
            }
            scaler_suffix = '_top10' if not args.disable_top10 else ''
            joblib.dump(scaler_bundle, os.path.join(args.out_path, f'{table_name}_scaler{scaler_suffix}.pkl'))
            # Example usage for new samples (documentation comment):
            # loaded = joblib.load('..._scaler.pkl'); new_df[loaded['scale_cols']] = loaded['scaler'].transform(new_df[loaded['scale_cols']])

    for clf_name in args.classifier:
        accs, cms, reports, aucs = train_classifier(clf_name, ckpt_path=args.ckpt_path, out_path=args.out_path, disable_top10=args.disable_top10)

        # Print average precision, recall, and F1 with std
        precisions = [r['macro avg']['precision'] for r in reports]
        recalls = [r['macro avg']['recall'] for r in reports]
        f1s = [r['macro avg']['f1-score'] for r in reports]
        print(f'Precision (mean ± std): {np.mean(precisions):.4f} ± {np.std(precisions):.4f}')
        print(f'Recall (mean ± std): {np.mean(recalls):.4f} ± {np.std(recalls):.4f}')
        print(f'F1-score (mean ± std): {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')

        print(f'Accuracy (mean ± std): {np.mean(accs):.4f} ± {np.std(accs):.4f}')
        print(f'AUC (mean ± std): {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')

        ### save model from the last run
        # if args.out_path:
        #     os.makedirs(args.out_path, exist_ok=True)
        #     if_top10 = '_top10.pkl' if not args.disable_top10 else '.pkl'
        #     joblib.dump(clf, os.path.join(args.out_path, f'{table_name}_{args.classifier}_acc_{acc:.2f}'+if_top10))
        #     # save the confusion matrix
        #     cm_df = pd.DataFrame(cms[-1], index=['复学', '休学'], columns=['复学', '休学'])
        #     cm_df.to_excel(os.path.join(args.out_path, f'{table_name}_{args.classifier}_confusion_matrix_'+if_top10+'.xlsx'), index=True)