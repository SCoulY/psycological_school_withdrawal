import os
import pandas as pd
import numpy as np

#read in excel file
def read_in_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        return pd.read_excel(file_path, sheet_name=None)

#drop columns with all NaN values and unnamed/uncorrelated columns
def drop_nan_columns(df):
    df = df.loc[:, ~df.columns.str.contains('原始分')]
    df = df.loc[:, ~df.columns.str.contains('备注')]
    df = df.loc[:, ~df.columns.str.contains('用户名')]
    df = df.loc[:, ~df.columns.str.contains('真实姓名')]
    # df = df.loc[:, ~df.columns.str.contains('性别')]
    df = df.loc[:, ~df.columns.str.contains('ID')]
    df = df.loc[:, ~df.columns.str.contains('序号')]
    df = df.loc[:, ~df.columns.str.contains('编号')]
    # df = df.loc[:, ~df.columns.str.contains('姓名')]
    df = df.loc[:, ~df.columns.str.contains('出生日期（年）')]
    df = df.loc[:, ~df.columns.str.contains('出生日期（年月）')]
    df = df.loc[:, ~df.columns.str.contains('自杀危险评估表')]
    df = df.loc[:, ~df.columns.str.contains('青少年分离体验量表（A-DES）标准分')]
    df = df.loc[:, ~df.columns.str.contains('舒格尔方格测试')]
    df = df.loc[:, ~df.columns.str.contains('年龄段')]

    return df.dropna(axis=1, how='all').dropna(axis=0, how='all')

#find mutual columns between two dataframes
def find_mutual_columns(df1, df2):
    return list(set(df1.columns).intersection(df2.columns))

#drop columns that are not numeric except for the first column
def select_numeric_columns(df):
    first = df['状态']
    second = df['姓名']
    df = df.select_dtypes(include=[np.number])
    df.insert(0, '状态', first)
    df.insert(1, '姓名', second)
    return df

def fill_missing_per_category(df):
    for col in df.columns[2:]:  # Iterate over columns to be filled (skip 'Category')
        df[col] = df.groupby('状态')[col].transform(lambda x: x.fillna(x.mean()))
    return df

def check_missing_values(df):
    return df.isnull().sum().sum() > 0

# Clean and convert the column that contains comma-separated integers
def process_feat_column(value):
    # Replace non-standard commas and strip whitespace
    cleaned = value.replace('，', ',').replace('.', ',').strip()
    # Remove trailing commas, if any
    cleaned = cleaned.rstrip(',')
    # Split into list and convert to integers
    return sum([int(x) for x in cleaned.split(',')])

    


if __name__ == "__main__":
    # file_2021= "C:/Users/SCoulY/Desktop/psycology/data/20-21.xls"
    # df_2021 = read_in_data(file_2021)
    # df_2021 = drop_nan_columns(df_2021)
    # print(list(df_2021.columns))

    file = "C:/Users/SCoulY/Desktop/psycology/data/SCXNSFJD_AllScales_20250320.xls"
    df = read_in_data(file)['Adults']
    df = drop_nan_columns(df)
    # print(list(df_2024.columns))

    df['性别'] = df['性别'].replace('男', 0).replace('女', 1)

    #fill missing values
    if check_missing_values(df):
        #identify columns that contain comma-separated integers
        df = df.map(lambda x: process_feat_column(x) 
                                   if (isinstance(x, str) and ('，' in x or '.' in x or ',' in x) and
                                       x[0].isdigit()) else x)
        df['母教-偏爱被试'] = df['母教-偏爱被试'].transform(lambda x: x.fillna(0)) #fill 父母偏爱
        df['父教-偏爱被试'] = df['父教-偏爱被试'].transform(lambda x: x.fillna(0)) #fill 父母偏爱
        df = select_numeric_columns(df)
        df = fill_missing_per_category(df)

    #drop rows which does not contain '休学' or '复学' in '休学复学' column
    df = df.dropna(subset=['状态'])


    #save cleaned data to a new file
    df.to_excel("C:/Users/SCoulY/Desktop/psycology/data/clean_adults.xlsx")
    df.to_csv("C:/Users/SCoulY/Desktop/psycology/data/clean_adult.csv")

    #convert '休学' to 0 and '复学' to 1
    label = df['状态']
    label = label.replace('休学', 0).replace('复学', 1)

    # select features
    feat = df.drop('状态', axis=1)
    print(list(feat.columns))

    #print(find_mutual_columns(df_2021, df_2024))
