import pandas as pd
import os
import argparse
from scipy.stats import norm
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.optimize import bisect
from tqdm import tqdm  
import json
from preprocess.ch2en import column_name2eng   


# ----------------------------
# Function to estimate quantile using KDE
# ----------------------------
def kde_quantile(data, target_quantile, tol=1e-4, fallback=True):
    data = data.dropna().values
    if len(data) < 5:
        if fallback:
            return np.quantile(data, target_quantile)
        else:
            raise ValueError("Not enough data for KDE.")

    kde = gaussian_kde(data)
    padding = 0.1 * (data.max() - data.min())
    xmin = data.min() - padding
    xmax = data.max() + padding

    def cdf(x):
        return quad(kde, xmin, x)[0]

    # Check if target quantile is within CDF support
    try:
        cdf_min = cdf(xmin)
        cdf_max = cdf(xmax)
        if not (cdf_min < target_quantile < cdf_max):
            raise ValueError("Target quantile outside KDE support.")

        return bisect(lambda x: cdf(x) - target_quantile, xmin, xmax, xtol=tol)
    except Exception as e:
        if fallback:
            return np.quantile(data, target_quantile)
        else:
            raise e


def compute_kde_quantiles(df_group, features, quantiles=[0.05, 0.95]):
    kde_qs = {}
    for feat in tqdm(features, desc="Computing KDE quantiles"):
        qs = {}
        for q in quantiles:
            try:
                qs[q] = kde_quantile(df_group[feat], q)
            except Exception as e:
                print(f"Failed for feature {feat} at quantile {q}: {e}")
                qs[q] = np.nan
        kde_qs[feat] = qs
    return pd.DataFrame(kde_qs)  # Features x Quantiles


def quantile(mu, std):
    # Calculate the 5th and 95th percentiles of a normal distribution
    q05 = norm.ppf(0.05, loc=mu, scale=std)
    q95 = norm.ppf(0.95, loc=mu, scale=std)
    return q05, q95

def select_with_fallback(df, model_cols, high=True, min_samples=10):
    if high:
        sel = (df['School Withdrawal/ Reentry Status'] == 0) & (
            (df[model_cols[0]] <= 0.3) & (df[model_cols[1]] <= 0.3)
        )
    else:
        sel = (df['School Withdrawal/ Reentry Status'] == 1) & (
            (df[model_cols[0]] >= 0.7) & (df[model_cols[1]] >= 0.7)
        )

    selected = df[sel]

    # Try top-10
    if len(selected) < min_samples:
        if high:
            ind_top10 = df[model_cols].max(axis=1).nsmallest(10).index
            ind_top10 = df[model_cols].min(axis=1).nlargest(10).index
            sel = (df['School Withdrawal/ Reentry Status'] == 0) & (df.index.isin(ind_top10))
        else:
            ind_top10 = df[model_cols].min(axis=1).nlargest(10).index
            sel = (df['School Withdrawal/ Reentry Status'] == 1) & (df.index.isin(ind_top10))
        selected = df[sel]

    # Try top-20
    if len(selected) < min_samples:
        if high:
            ind_top20 = df[model_cols].min(axis=1).nsmallest(20).index
            sel = (df['School Withdrawal/ Reentry Status'] == 0) & (df.index.isin(ind_top20))
        else:
            ind_top20 = df[model_cols].max(axis=1).nlargest(20).index
            sel = (df['School Withdrawal/ Reentry Status'] == 1) & (df.index.isin(ind_top20))
        selected = df[sel]

    # Final fallback: ensure at least min_samples
    if len(selected) < min_samples:
        if high:
            selected = df[df['School Withdrawal/ Reentry Status'] == 0].sort_values(by=model_cols, ascending=True).head(min_samples)
        else:
            selected = df[df['School Withdrawal/ Reentry Status'] == 1].sort_values(by=model_cols, ascending=False).head(min_samples)

    return selected

def cal_stats(df, save_path=None):
    df_high_risk = select_with_fallback(df, ['LogisticRegression', 'RandomForest'], high=True)
    print(f'high risk: {df_high_risk.shape[0]}, total_withdraw: {df[df["School Withdrawal/ Reentry Status"] == 0].shape[0]}')

    df_low_risk = select_with_fallback(df, ['LogisticRegression', 'RandomForest'], high=False)
    print(f'low risk: {df_low_risk.shape[0]}, total_recover: {df[df["School Withdrawal/ Reentry Status"] == 1].shape[0]}')

    if save_path:
        #save description to excel
        df_low_risk.describe().to_excel(os.path.join(save_path, 'low_risk.xlsx'), index=True)

        #save description to excel
        df_high_risk.describe().to_excel(os.path.join(save_path, 'high_risk.xlsx'), index=True)
    return df_low_risk, df_high_risk

def find_uncertain_intervals(value, high_q, low_q):
    '''Determine the uncertain level for a given value based on quantiles.'''
    l5, l95 = low_q['0.05'], low_q['0.95']
    h5, h95 = high_q['0.05'], high_q['0.95']

    low_bound = min(l5, h5)
    high_bound = max(l95, h95)
    overlap_low = max(l5, h5)
    overlap_high = min(l95, h95)

    if value < low_bound or value > high_bound:
        return 'high'
    elif overlap_low <= value <= overlap_high:
        return 'low'
    else:
        return 'medium'
        

def compute_signed_anomaly_score(value, high_q, low_q):
    '''Compute a signed continuous anomaly score:
    - [-0.5, 0.5] → low uncertainty (normal, but informative)
    - [-1, -0.5) and (0.5, 1] → medium uncertainty
    - < -1 or > 1 → high uncertainty
    '''
    l5, l95 = low_q[0], low_q[1]
    h5, h95 = high_q[0], high_q[1]

    low_bound = min(l5, h5)
    high_bound = max(l95, h95)
    overlap_low = max(l5, h5)
    overlap_high = min(l95, h95)

    # Handle low (normal) region with continuous informative score
    if overlap_low <= value <= overlap_high:
        center = (overlap_low + overlap_high) / 2
        half_range = (overlap_high - overlap_low) / 2
        if half_range == 0:
            return 0.0  # avoid division by zero if overlap collapses
        return 0.5 * (value - center) / half_range  # ranges from -0.5 to 0.5

    # Below the overlap region
    elif value < overlap_low:
        if value >= low_bound:
            # Medium uncertainty
            scale = overlap_low - low_bound
            return -0.5 - 0.5 * (overlap_low - value) / scale if scale != 0 else -0.5
        else:
            # High uncertainty
            scale = overlap_low - low_bound
            return -1.0 - ((low_bound - value) / scale) if scale != 0 else -1.0

    # Above the overlap region
    else:  # value > overlap_high
        if value <= high_bound:
            # Medium uncertainty
            scale = high_bound - overlap_high
            return 0.5 + 0.5 * (value - overlap_high) / scale if scale != 0 else 0.5
        else:
            # High uncertainty
            scale = high_bound - overlap_high
            return 1.0 + ((value - high_bound) / scale) if scale != 0 else 1.0


 
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file_path', type=str, help='Path to the excel file', default='risk_prob/full/clean_children_risk_prob.xlsx')
    args.add_argument('--kde_q_high_path', type=str, help='Path to the kde_q_high json file', default='children_kde_q_high.json')
    args.add_argument('--kde_q_low_path', type=str, help='Path to the kde_q_low json file', default='children_kde_q_low.json')
    args.add_argument('--output_path', type=str, help='Path to save the plot', default='risk_prob/full/')

    args = args.parse_args()

    if args.file_path.endswith('.xlsx') or args.file_path.endswith('.xls'):
        df = pd.read_excel(args.file_path)
    elif args.file_path.endswith('.csv'):
        df = pd.read_csv(args.file_path, encoding='utf-8')
    # df = column_name2eng(df)

    print(df.columns.tolist())
    df_uncertainty = df.copy()
    if 'Age' in df.columns:
        df.drop(columns=['Age'], inplace=True)
    if 'Gender' in df.columns:
        df.drop(columns=['Gender'], inplace=True)
    if 'name' in df.columns:
        df.drop(columns=['name'], inplace=True)


    if 'adults' in os.path.basename(args.file_path):
        name = 'adults'
    elif 'children' in os.path.basename(args.file_path):
        name = 'children'
    elif 'teens' in os.path.basename(args.file_path):
        name = 'teens'
    else:
        raise ValueError("Unknown file type, please check the file name.")
    df_low_risk, df_high_risk = cal_stats(df)
    print(df_low_risk.columns.tolist())

    ## quantile calculation and save to json
    df_high_risk = df_high_risk.drop(columns=['School Withdrawal/ Reentry Status', 'LogisticRegression', 'RandomForest'])
    df_low_risk = df_low_risk.drop(columns=['School Withdrawal/ Reentry Status', 'LogisticRegression', 'RandomForest'])

    features = df_high_risk.columns.to_list()

    # df_high_risk and df_low_risk are your filtered groups
    kde_q_high = compute_kde_quantiles(df_high_risk, features)
    kde_q_low = compute_kde_quantiles(df_low_risk, features)

    # print(kde_q_high)
    # print(kde_q_low)

    stats_path = os.path.join(args.output_path, 'stats')
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    # Save the quantiles to json files
    kde_q_high.to_json(os.path.join(stats_path, f'{name}_kde_q_high.json'), orient='index')
    kde_q_low.to_json(os.path.join(stats_path, f'{name}_kde_q_low.json'), orient='index')

    ### place each data point in the three defined confidence intervals
    df = df.drop(columns=['School Withdrawal/ Reentry Status', 'LogisticRegression', 'RandomForest'])
    features = df.columns.to_list()

    with open(os.path.join(stats_path, args.kde_q_high_path), 'r') as f:
        kde_q_high = json.load(f)
        kde_q_high = pd.DataFrame(kde_q_high).T
    with open(os.path.join(stats_path, args.kde_q_low_path), 'r') as f:
        kde_q_low = json.load(f)
        kde_q_low = pd.DataFrame(kde_q_low).T

    for feat in features:
        q_low = kde_q_low[feat]
        q_high = kde_q_high[feat]

        # uncertainty = df[feat].apply(find_uncertain_intervals, args=(q_low, q_high))
        # df_uncertainty[feat] = uncertainty
        anomaly = df[feat].apply(compute_signed_anomaly_score, args=(q_high, q_low))
        df_uncertainty[feat] = anomaly

    # save the dataframe with uncertainty intervals to excel
    df_uncertainty.to_excel(os.path.join(args.output_path, f'{name}_anomaly.xlsx'), index=False)