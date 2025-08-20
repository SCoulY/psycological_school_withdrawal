from pandas import DataFrame 
import numpy as np

name_dic = {
            '状态': 'School Withdrawal/ Reentry Status',
            'ID': 'ID',
            '用户名': 'Username',
            '姓名': 'Name',
            '性别': 'Gender',
            '年龄段': 'Age Group',
            '年龄（与实际有出入）': 'Age',

            'SCl-90 躯体化': 'SCL-90 SOM',
            'SCl-90 强迫': 'SCL-90 OC',
            'SCl-90 人际敏感': 'SCL-90 IS',
            'SCl-90 抑郁': 'SCL-90 DEP',
            'SCl-90 焦虑': 'SCL-90 ANX',
            'SCl-90 敌对': 'SCL-90 HOS',
            'SCl-90 恐怖': 'SCL-90 PHOB',
            'SCl-90 偏执': 'SCL-90 PAR',
            'SCl-90 精神病性': 'SCL-90 PSY',
            'SCl-90 其他': 'SCL-90 ADD',
            'SCl-90 总分': 'SCL-90 TS',
            'SCl-90 总均分': 'SCL-90 GSI',
            'SCl-90 阳性项目数': 'SCL-90 PST',
            'SCl-90 阴性项目数': 'SCL-90 NST',
            'SCl-90 阳性症状均分': 'SCL-90 PSDI',

            '母教-情感温暖、理解': 'EMBU-M EW',
            '母教-过度保护、过分干涉': 'EMBU-M OI',     
            '母教-拒绝、否认': 'EMBU-M REJ',
            '母教-惩罚、严厉': 'EMBU-M PUN',
            '母教-偏爱被试': 'EMBU-M FS',

            '父教-情感温暖、理解': 'EMBU-F EW',
            '父教-过分干涉': 'EMBU-F OI',
            '父教-拒绝、否认': 'EMBU-F REJ',
            '父教-惩罚、严厉': 'EMBU-F PUN',
            '父教-偏爱被试': 'EMBU-F FS',
            '父教-过度保护': 'EMBU-F OP',

            '核心自我评价-总分': 'CSES_TS',
            '华西心晴指数-总分': 'HEI_TS',

            '应付-解决问题': 'CSQ_PS',
            '应付-自责': 'CSQ_SB',
            '应付-求助': 'CSQ_HS',
            '应付-幻想': 'CSQ_FAN',
            '应付-退避': 'CSQ_REP',
            '应付-合理化': 'CSQ_RAT',

            '社支-总分': 'SSRS_TS',
            '社支-主观支持': 'SSRS_SS',
            '社支-客观支持': 'SSRS_OS',
            '社支-对支持利用度': 'SSRS_SU',
            '青社支-总分': 'A-SSRS_TS',
            '青社支-主观支持': 'A-SSRS_SS',
            '青社支-客观支持': 'A-SSRS_OS',
            '青社支-支持利用度': 'A-SSRS_SU',


            '分离-专注与想象性参与': 'DES-Ⅱ_ABS',
            '分离-遗忘性分离': 'DES-Ⅱ_AMN',
            '分离-人格解体和现实解体': 'DES-Ⅱ_DPDR',
            '分离-总分': 'DES-Ⅱ_TS',
            '青分离-分离性遗忘': 'A-DES-Ⅱ_DA',
            '青分离-专注与想象性投入': 'A-DES-Ⅱ_AII',
            '青分离-被动影响': 'A-DES-Ⅱ_PI',
            '青分离-现实解体与人格解体': 'A-DES-Ⅱ_DPDR',
            '青分离-总分': 'A-DES-Ⅱ_TS'
            }

def column_name2eng(df: DataFrame) -> DataFrame:
    '''Convert Chinese column names to English'''
    for col in df.columns:
        if col in name_dic:
            df.rename(columns={col: name_dic[col]}, inplace=True)
    return df




           
  