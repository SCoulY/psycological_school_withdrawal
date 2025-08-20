import numpy as np
import pandas as pd
import os
from preprocess.ch2en import column_name2eng

adult_csv = 'C:/Users/SCoulY/Desktop/psycology/data/clean_adults.csv'
teens_csv = 'C:/Users/SCoulY/Desktop/psycology/data/clean_teens.csv'
children_csv = 'C:/Users/SCoulY/Desktop/psycology/data/clean_teens_wo_scl.csv'

adult_df = pd.read_csv(adult_csv, encoding='utf-8')
teens_df = pd.read_csv(teens_csv, encoding='utf-8')
children_df = pd.read_csv(children_csv, encoding='utf-8')

#drop index column
adult_df = adult_df.drop(adult_df.columns[0], axis=1)
teens_df = teens_df.drop(teens_df.columns[0], axis=1)
children_df = children_df.drop(children_df.columns[0], axis=1)

#chinese to english
adult_df = column_name2eng(adult_df)
teens_df = column_name2eng(teens_df)
children_df = column_name2eng(children_df)

adult_df['School Withdrawal/ Reentry Status'] = adult_df['School Withdrawal/ Reentry Status'].replace('休学', 0).replace('复学', 1)
teens_df['School Withdrawal/ Reentry Status'] = teens_df['School Withdrawal/ Reentry Status'].replace('休学', 0).replace('复学', 1)
children_df['School Withdrawal/ Reentry Status'] = children_df['School Withdrawal/ Reentry Status'].replace('休学', 0).replace('复学', 1)

#save
adult_df.to_csv(adult_csv.replace('.csv', '_eng.csv'), index=False, encoding='utf-8')
teens_df.to_csv(teens_csv.replace('.csv', '_eng.csv'), index=False, encoding='utf-8')
children_df.to_csv(children_csv.replace('.csv', '_eng.csv'), index=False, encoding='utf-8')