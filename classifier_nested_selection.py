from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import numpy as np
import pandas as pd
import os
import argparse
import joblib
from collections import Counter


def get_top_features_from_model(clf, feature_names, n_features=10):
    """Extract top features from trained classifier based on feature importance."""
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_[0])
    else:
        return None
    
    top_indices = np.argsort(importances)[-n_features:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    return top_features, top_importances


def derive_top10_per_group_classifier(classifier, feat, label, seeds):
    """
    Train full-feature models across all seeds and collect top-10 features.
    Returns list of all top-10 features across seeds.
    """
    all_top_features = []
    feature_names = feat.columns.tolist()
    
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=seed)
        
        if classifier == 'RandomForest':
            clf = RandomForestClassifier(random_state=seed)
        elif classifier == 'LogisticRegression':
            clf = LogisticRegression(random_state=seed, max_iter=1000)
        else:
            raise ValueError(f"Invalid classifier: {classifier}")
        
        clf.fit(X_train, y_train)
        
        result = get_top_features_from_model(clf, feature_names, n_features=10)
        if result is not None:
            top_features, importances = result
            all_top_features.extend(top_features)
    
    return all_top_features


def categorize_feature(feature_name):
    """Categorize a feature into its conceptual group."""
    feature_upper = feature_name.upper()
    
    if 'HEI' in feature_upper:
        return 'HEI'
    elif 'CSES' in feature_upper:
        return 'CSES'
    elif 'EMBU' in feature_upper:
        return 'EMBU'
    elif 'DES' in feature_upper:
        return 'DES'
    elif 'SCL-90' in feature_upper or 'SCL90' in feature_upper:
        return 'SCL-90'
    elif 'SSRS' in feature_upper:
        return 'SSRS'
    elif 'CSQ' in feature_upper:
        return 'CSQ'
    else:
        return 'OTHER'


def normalize_feature_name(feature_name, group_name):
    """
    Normalize feature names to match the dataset's naming convention.
    Adults: DES-Ⅱ_* and SSRS_*
    Teens/Children: A-DES-Ⅱ_* and A-SSRS_*
    
    For adults DES features, map teen/child DES features to available adult DES features:
    - A-DES-Ⅱ_PI -> DES-Ⅱ_AMN (Passive Influence -> Amnesia, closest available)
    - Other DES features map to same subscale if available
    """
    if group_name == 'adults':
        # Remove A- prefix for adults
        if feature_name.startswith('A-DES'):
            base_feature = feature_name.replace('A-DES', 'DES')
            # Special mapping for PI (not available in adults) -> use AMN instead
            if '_PI' in base_feature:
                return 'DES-Ⅱ_AMN'
            return base_feature
        elif feature_name.startswith('A-SSRS'):
            return feature_name.replace('A-SSRS', 'SSRS')
    else:
        # Add A- prefix for teens/children if not present
        if feature_name.startswith('DES-') and not feature_name.startswith('A-DES'):
            return 'A-' + feature_name
        elif feature_name.startswith('SSRS_') and not feature_name.startswith('A-SSRS'):
            return 'A-' + feature_name
    
    return feature_name


def select_cross_age_top10(all_groups_features, scl90_features_to_replace=None, children_specific_features=None):
    """
    Select the cross-age-group top-10 features ensuring conceptual diversity and empirical robustness.
    
    The selection ensures representation from all multi-item conceptual groups:
    EMBU, DES, SCL-90, SSRS, and CSQ
    
    Args:
        all_groups_features: List of features from all groups and classifiers
        scl90_features_to_replace: For children, list of SCL-90 features to replace
        children_specific_features: For children, candidate features to use as replacements
    
    Returns:
        List of 10 selected features
    """
    # Step 1: Automatic inclusion
    required_features = ['HEI_TS', 'CSES_TS']
    selected_features = required_features.copy()
    
    # Step 2: Frequency-based selection with conceptual diversity
    feature_counts = Counter(all_groups_features)
    
    # Remove required features from counting (already included)
    for req_feat in required_features:
        if req_feat in feature_counts:
            del feature_counts[req_feat]
    
    # Group features by conceptual category
    categorized_features = {}
    for feat, count in feature_counts.items():
        category = categorize_feature(feat)
        if category not in categorized_features:
            categorized_features[category] = []
        categorized_features[category].append((feat, count))
    
    # Sort features within each category by frequency
    for category in categorized_features:
        categorized_features[category].sort(key=lambda x: x[1], reverse=True)
    
    # Multi-item conceptual groups that should be represented
    multi_item_groups = ['EMBU', 'DES', 'SCL-90', 'SSRS', 'CSQ']
    
    if scl90_features_to_replace and children_specific_features:
        # For children: exclude SCL-90
        multi_item_groups = ['EMBU', 'DES', 'SSRS', 'CSQ']
        
        # Categorize children-specific features
        children_counts = Counter(children_specific_features)
        for req_feat in required_features:
            if req_feat in children_counts:
                del children_counts[req_feat]
        
        children_categorized = {}
        for feat, count in children_counts.items():
            category = categorize_feature(feat)
            if category not in children_categorized:
                children_categorized[category] = []
            children_categorized[category].append((feat, count))
        
        for category in children_categorized:
            children_categorized[category].sort(key=lambda x: x[1], reverse=True)
        
        # Ensure at least 1 feature from each multi-item group (if available)
        for group in multi_item_groups:
            if group in children_categorized and children_categorized[group]:
                feat, count = children_categorized[group][0]
                if feat not in selected_features:
                    selected_features.append(feat)
        
        # Fill remaining slots with most frequent features from children data
        all_children_feats = [(f, c) for f, c in children_counts.items() if categorize_feature(f) in multi_item_groups]
        all_children_feats.sort(key=lambda x: x[1], reverse=True)
        
        for feat, count in all_children_feats:
            if feat not in selected_features and len(selected_features) < 10:
                selected_features.append(feat)
    else:
        # For adults/teens: ensure representation from all 5 multi-item groups
        # First, ensure at least 1 feature from each group
        for group in multi_item_groups:
            if group in categorized_features and categorized_features[group]:
                feat, count = categorized_features[group][0]
                if feat not in selected_features:
                    selected_features.append(feat)
        
        # Fill remaining slots with most frequent features overall from multi-item groups
        all_multi_item_feats = []
        for group in multi_item_groups:
            if group in categorized_features:
                all_multi_item_feats.extend(categorized_features[group])
        
        all_multi_item_feats.sort(key=lambda x: x[1], reverse=True)
        
        for feat, count in all_multi_item_feats:
            if feat not in selected_features and len(selected_features) < 10:
                selected_features.append(feat)
    
    return selected_features[:10]


def train_classifier(classifier, feat, label, table_name, seeds, out_path=None, disable_top10=False):
    """Train classifier and return performance metrics."""
    accs = []
    cms = []
    reports = []
    aucs = []
    
    print(f'Training {classifier} model across {len(seeds)} seeds...')
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=seed)

        if classifier == 'RandomForest':
            clf = RandomForestClassifier(random_state=seed)
        elif classifier == 'LogisticRegression':
            clf = LogisticRegression(random_state=seed, max_iter=1000)
        else:
            raise ValueError(f"Invalid classifier: {classifier}")
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        cms.append(confusion_matrix(y_test, y_pred))
        reports.append(classification_report(y_test, y_pred, output_dict=True))
        
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)
        auc = roc_auc_score(y_test, y_score)
        aucs.append(auc)

        if out_path:
            os.makedirs(out_path, exist_ok=True)
            if_top10 = '_top10.pkl' if not disable_top10 else '.pkl'
            run_str = f'_run_{seed}'
            joblib.dump(clf, os.path.join(out_path, f'{table_name}_{classifier}_acc_{acc:.2f}'+run_str+if_top10))

    return accs, cms, reports, aucs


def process_single_group(file_path, group_name, classifiers, seeds, out_base_path):
    """Process a single age group and return top-10 features from each classifier."""
    print(f"\n{'='*70}")
    print(f"Processing {group_name.upper()} dataset")
    print(f"{'='*70}")
    
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    feat_full = df.drop(columns=['School Withdrawal/ Reentry Status'])
    label = df['School Withdrawal/ Reentry Status']
    
    print(f"Total features: {len(feat_full.columns)}")
    print(f"Total samples: {len(label)} (Positive: {sum(label)}, Negative: {len(label)-sum(label)})")
    
    # Scale features
    exclude_cols = ['Gender'] if 'Gender' in feat_full.columns else []
    scale_cols = [c for c in feat_full.columns if c not in exclude_cols]
    
    scaler = StandardScaler()
    X_scaled_part = scaler.fit_transform(feat_full[scale_cols])
    X_scaled_df = feat_full.copy()
    X_scaled_df[scale_cols] = X_scaled_part
    feat_full = X_scaled_df
    
    # Save scaler
    out_path = os.path.join(out_base_path, group_name)
    os.makedirs(out_path, exist_ok=True)
    scaler_bundle = {
        'scaler': scaler,
        'scale_cols': scale_cols,
        'exclude_cols': exclude_cols,
        'feature_order': feat_full.columns.tolist()
    }
    joblib.dump(scaler_bundle, os.path.join(out_path, f'clean_{group_name}_scaler.pkl'))
    
    # Derive top-10 features for each classifier
    group_features = {}
    all_group_features = []
    
    print(f"\nDeriving top-10 features from models...")
    for clf_name in classifiers:
        print(f"  {clf_name}...")
        top_features = derive_top10_per_group_classifier(clf_name, feat_full, label, seeds)
        group_features[clf_name] = top_features
        all_group_features.extend(top_features)
        
        # Show top 5 most frequent
        counts = Counter(top_features)
        print(f"    Most frequent: {counts.most_common(5)}")
    
    return {
        'group_name': group_name,
        'features': feat_full,
        'label': label,
        'all_features': all_group_features,
        'by_classifier': group_features,
        'out_path': out_path
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    args.add_argument('--out_path', type=str, default='ckpt/cross_age_selection', help='Output directory')
    args.add_argument('--classifier', type=str, nargs='+', default=['RandomForest', 'LogisticRegression'])
    args = args.parse_args()
    
    seeds = [42, 123, 2025, 6, 255]
    
    # Process all three age groups
    print(f"\n{'#'*70}")
    print("PHASE 1: COLLECT TOP-10 FEATURES FROM ALL AGE GROUPS")
    print(f"{'#'*70}")
    
    groups_data = {}
    for group in ['adults', 'teens', 'children']:
        file_path = os.path.join(args.data_dir, f'clean_{group}.csv')
        groups_data[group] = process_single_group(file_path, group, args.classifier, seeds, args.out_path)
    
    # Collect all features from adults and teens (for SCL-90 included set)
    print(f"\n{'#'*70}")
    print("PHASE 2: CROSS-AGE-GROUP FEATURE SELECTION")
    print(f"{'#'*70}")
    
    # Combine features from adults and teens for general top-10
    adults_teens_features = groups_data['adults']['all_features'] + groups_data['teens']['all_features']
    
    print(f"\nTotal feature appearances across adults and teens:")
    print(f"  Adults: {len(groups_data['adults']['all_features'])} features")
    print(f"  Teens: {len(groups_data['teens']['all_features'])} features")
    print(f"  Combined: {len(adults_teens_features)} features")
    
    # Select cross-age top-10 (with SCL-90)
    print(f"\n--- Selecting Cross-Age Top-10 (for Adults & Teens) ---")
    cross_age_top10 = select_cross_age_top10(adults_teens_features)
    
    print(f"\nSelected Cross-Age Top-10 Features:")
    feature_freq = Counter(adults_teens_features)
    for i, feat in enumerate(cross_age_top10, 1):
        freq = feature_freq.get(feat, 0)
        total_possible = len(args.classifier) * 2 * len(seeds)  # 2 groups (adults, teens)
        category = categorize_feature(feat)
        required_mark = " [REQUIRED]" if feat in ['HEI_TS', 'CSES_TS'] else ""
        print(f"  {i}. {feat} - {category} (appeared {freq}/{total_possible} times){required_mark}")
    
    # For children: replace SCL-90 features
    print(f"\n--- Selecting Children-Specific Top-10 ---")
    scl90_in_cross_age = [f for f in cross_age_top10 if 'SCL-90' in f or 'SCL90' in f]
    print(f"SCL-90 features to replace: {scl90_in_cross_age}")
    
    # Get children-specific top features (excluding SCL-90)
    children_features = groups_data['children']['all_features']
    children_no_scl90 = [f for f in children_features if 'SCL-90' not in f and 'SCL90' not in f]
    
    # Start with non-SCL-90 features from cross-age top-10 and normalize for children
    children_top10 = []
    for f in cross_age_top10:
        if 'SCL-90' not in f and 'SCL90' not in f:
            # Normalize the feature name for children (add A- prefix if needed)
            normalized = normalize_feature_name(f, 'children')
            children_top10.append(normalized)
    
    # Add most frequent children-specific features to reach 10
    children_freq = Counter(children_no_scl90)
    for feat in ['HEI_TS', 'CSES_TS']:
        if feat in children_freq:
            del children_freq[feat]
    
    for feat, count in children_freq.most_common():
        if feat not in children_top10 and len(children_top10) < 10:
            children_top10.append(feat)
    
    print(f"\nSelected Children-Specific Top-10 Features:")
    for i, feat in enumerate(children_top10, 1):
        freq = children_freq.get(feat, 0) if feat not in ['HEI_TS', 'CSES_TS'] else 'N/A'
        category = categorize_feature(feat)
        required_mark = " [REQUIRED]" if feat in ['HEI_TS', 'CSES_TS'] else ""
        print(f"  {i}. {feat} - {category} (freq: {freq}){required_mark}")
    
    # Show conceptual diversity
    print(f"\n--- Conceptual Diversity Analysis ---")
    
    print(f"\nCross-Age Top-10 by Conceptual Group:")
    cross_age_groups = {}
    for feat in cross_age_top10:
        cat = categorize_feature(feat)
        if cat not in cross_age_groups:
            cross_age_groups[cat] = []
        cross_age_groups[cat].append(feat)
    
    for group in ['HEI', 'CSES', 'EMBU', 'DES', 'SCL-90', 'SSRS', 'CSQ']:
        if group in cross_age_groups:
            print(f"  {group}: {len(cross_age_groups[group])} features - {cross_age_groups[group]}")
    
    print(f"\nChildren-Specific Top-10 by Conceptual Group:")
    children_groups = {}
    for feat in children_top10:
        cat = categorize_feature(feat)
        if cat not in children_groups:
            children_groups[cat] = []
        children_groups[cat].append(feat)
    
    for group in ['HEI', 'CSES', 'EMBU', 'DES', 'SCL-90', 'SSRS', 'CSQ']:
        if group in children_groups:
            print(f"  {group}: {len(children_groups[group])} features - {children_groups[group]}")
    
    # Save the selected features
    selection_summary = {
        'cross_age_top10': cross_age_top10,
        'cross_age_top10_adults': [normalize_feature_name(f, 'adults') for f in cross_age_top10],
        'cross_age_top10_teens': [normalize_feature_name(f, 'teens') for f in cross_age_top10],
        'children_top10': children_top10,
        'scl90_replaced': scl90_in_cross_age,
        'feature_frequencies': {
            'adults_teens': dict(Counter(adults_teens_features)),
            'children': dict(Counter(children_features))
        }
    }
    joblib.dump(selection_summary, os.path.join(args.out_path, 'cross_age_feature_selection.pkl'))
    
    # Train models with selected features
    print(f"\n{'#'*70}")
    print("PHASE 3: TRAIN MODELS WITH SELECTED FEATURES")
    print(f"{'#'*70}")
    
    results = {}
    
    for group in ['adults', 'teens', 'children']:
        print(f"\n{'='*70}")
        print(f"TRAINING: {group.upper()}")
        print(f"{'='*70}")
        
        gdata = groups_data[group]
        feat_full = gdata['features']
        label = gdata['label']
        out_path = gdata['out_path']
        
        # Select appropriate top-10 for this group and normalize names
        if group == 'children':
            top10_raw = children_top10
        else:
            top10_raw = cross_age_top10
        
        # Normalize feature names for this group's dataset
        top10 = []
        for feat in top10_raw:
            normalized_feat = normalize_feature_name(feat, group)
            if normalized_feat in feat_full.columns:
                top10.append(normalized_feat)
        
        print(f"\nUsing Top-10 Features ({len(top10)} available):")
        for i, f in enumerate(top10, 1):
            print(f"  {i}. {f}")
        
        if len(top10) < 10:
            print(f"\n⚠ Warning: Only {len(top10)}/10 features available in {group} dataset")
        
        feat_top10 = feat_full[top10]
        
        results[group] = {}
        
        for clf_name in args.classifier:
            print(f"\n--- {clf_name} ---")
            
            # Train with full features
            print("Training with FULL features...")
            accs_full, _, reports_full, aucs_full = train_classifier(
                clf_name, feat_full, label, f'clean_{group}', seeds, 
                out_path=out_path, disable_top10=True
            )
            
            # Train with top-10 features
            print("Training with TOP-10 features...")
            accs_top10, _, reports_top10, aucs_top10 = train_classifier(
                clf_name, feat_top10, label, f'clean_{group}', seeds,
                out_path=out_path, disable_top10=False
            )
            
            # Calculate metrics
            def extract_metrics(reports):
                precisions = [r['macro avg']['precision'] for r in reports]
                recalls = [r['macro avg']['recall'] for r in reports]
                f1s = [r['macro avg']['f1-score'] for r in reports]
                return precisions, recalls, f1s
            
            prec_full, rec_full, f1_full = extract_metrics(reports_full)
            prec_top10, rec_top10, f1_top10 = extract_metrics(reports_top10)
            
            results[group][clf_name] = {
                'full': {
                    'precision': (np.mean(prec_full), np.std(prec_full)),
                    'recall': (np.mean(rec_full), np.std(rec_full)),
                    'f1': (np.mean(f1_full), np.std(f1_full)),
                    'accuracy': (np.mean(accs_full), np.std(accs_full)),
                    'auc': (np.mean(aucs_full), np.std(aucs_full))
                },
                'top10': {
                    'precision': (np.mean(prec_top10), np.std(prec_top10)),
                    'recall': (np.mean(rec_top10), np.std(rec_top10)),
                    'f1': (np.mean(f1_top10), np.std(f1_top10)),
                    'accuracy': (np.mean(accs_top10), np.std(accs_top10)),
                    'auc': (np.mean(aucs_top10), np.std(aucs_top10))
                }
            }
            
            # Print results
            print(f"\nFull Features ({len(feat_full.columns)}):")
            print(f"  Precision: {np.mean(prec_full):.4f}±{np.std(prec_full):.4f}")
            print(f"  Recall:    {np.mean(rec_full):.4f}±{np.std(rec_full):.4f}")
            print(f"  F1-score:  {np.mean(f1_full):.4f}±{np.std(f1_full):.4f}")
            print(f"  Accuracy:  {np.mean(accs_full):.4f}±{np.std(accs_full):.4f}")
            print(f"  AUC:       {np.mean(aucs_full):.4f}±{np.std(aucs_full):.4f}")
            
            print(f"\nTop-10 Features ({len(top10)}):")
            print(f"  Precision: {np.mean(prec_top10):.4f}±{np.std(prec_top10):.4f}")
            print(f"  Recall:    {np.mean(rec_top10):.4f}±{np.std(rec_top10):.4f}")
            print(f"  F1-score:  {np.mean(f1_top10):.4f}±{np.std(f1_top10):.4f}")
            print(f"  Accuracy:  {np.mean(accs_top10):.4f}±{np.std(accs_top10):.4f}")
            print(f"  AUC:       {np.mean(aucs_top10):.4f}±{np.std(aucs_top10):.4f}")
            
            print(f"\nΔ Change (Top-10 vs Full):")
            print(f"  Precision: {np.mean(prec_top10)-np.mean(prec_full):+.4f}")
            print(f"  Recall:    {np.mean(rec_top10)-np.mean(rec_full):+.4f}")
            print(f"  F1-score:  {np.mean(f1_top10)-np.mean(f1_full):+.4f}")
            print(f"  Accuracy:  {np.mean(accs_top10)-np.mean(accs_full):+.4f}")
            print(f"  AUC:       {np.mean(aucs_top10)-np.mean(aucs_full):+.4f}")
    
    # Save results
    joblib.dump(results, os.path.join(args.out_path, 'performance_results.pkl'))
    
    print(f"\n{'#'*70}")
    print("SUMMARY")
    print(f"{'#'*70}")
    
    print(f"\nCross-Age Top-10 (base naming):")
    print(cross_age_top10)
    
    print(f"\nCross-Age Top-10 for Adults (with correct DES/SSRS naming):")
    adults_top10 = [normalize_feature_name(f, 'adults') for f in cross_age_top10]
    print(adults_top10)
    
    print(f"\nCross-Age Top-10 for Teens (with correct DES/SSRS naming):")
    teens_top10 = [normalize_feature_name(f, 'teens') for f in cross_age_top10]
    print(teens_top10)
    
    print(f"\nChildren-Specific Top-10 (with correct naming):")
    print(children_top10)
    
    print(f"\nAll results saved to: {args.out_path}")
    print(f"{'#'*70}\n")
