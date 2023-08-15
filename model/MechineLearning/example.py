# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(zhangqiude)s
"""
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')
# %matplotlib inline

## 数据处理
from sklearn import preprocessing

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

## 模型预测的arima
import lightgbm as lgb   # svr
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

path = '../../input/'
## 1) 载入训练集和测试集；
Train_data = pd.read_csv(path+'file1.txt', sep='\t')
TestA_data = pd.read_csv(path+'file2.txt', sep='\t')

Train_data.info()
TestA_data.info()


##删除某些属性
Train_data=Train_data.drop(['f4','f7','f15'],axis=1)
TestA_data=TestA_data.drop(['f4','f7','f15'],axis=1)


## 输出数据的大小信息
# print('Train data shape:',Train_data.shape)
# print('TestA data shape:',TestA_data.shape)

# Train_data.price.hist()
# Train_data.info()
# TestA_data.info()

#预处理函数
def reduce_mem(df): 
    """ iterate through all the columns of a dataframe and modify the data type  to reduce memory usage.             """ 
    start_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem)) 
    df1 = df.copy() 
    for col in df.columns: 
        col_type = df[col].dtype 
         
        if col_type != object: 
            c_min = df[col].min() 
            c_max = df[col].max() 
            if str(col_type)[:3] == 'int': 
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: 
                    df[col] = df[col].astype(np.int8) 
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: 
                    df[col] = df[col].astype(np.int16) 
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: 
                    df[col] = df[col].astype(np.int32) 
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: 
                    df[col] = df[col].astype(np.int64)   
            else: 
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: 
                    df[col] = df[col].astype(np.float16) 
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: 
                    df[col] = df[col].astype(np.float32) 
                else: 
                    df[col] = df[col].astype(np.float64) 
        #Treatment for category columns
        else: 
            df[col] = df[col].astype('category') 
 
    end_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem)) 
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem)) 
    if end_mem > start_mem:
        print(f'Memory usage increases! Return the original df.')
        return df1
    return df 

Train_data = reduce_mem(Train_data)
TestA_data = reduce_mem(TestA_data)

##输出前五行
# display(Train_data.head())
# display(TestA_data.head())

numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
# print(numerical_cols)
categorical_cols = Train_data.select_dtypes(include = 'object').columns
# print(categorical_cols)

# Train_data.info()
# TestA_data.info()

##时间属性
# date_features = ['tradeTime', 'registerDate','licenseDate','f7','f15']
date_features = ['tradeTime', 'registerDate','licenseDate']
date_feature_s=['f13']
##连续属性
# numeric_features = ['mileage', 'displacement','newprice','price'] 
numeric_features = ['mileage', 'displacement','newprice','price','tradeTime_year','tradeTime_month','tradeTime_day','registerDate_year','registerDate_month','registerDate_day','licenseDate_year','licenseDate_month','licenseDate_day'] 


##离散属性
# categorical_features = ['carid', 'brand', 'serial', 'model', 'color', 'cityId', 'carCode', 'transferCount','seatings','country','maketype','modelyear','gearbox','oiltype']+\
#     ['f{}'.format(i) for i in range(1,7)]+['f{}'.format(i) for i in range(8,13)]+['f14']
categorical_features = ['carid', 'brand', 'serial', 'model', 'color', 'cityId', 'carCode', 'transferCount','seatings','country','maketype','modelyear','gearbox','oiltype']+\
    ['f{}'.format(i) for i in range(1,4)]+['f{}'.format(i) for i in range(5,7)]+['f{}'.format(i) for i in range(8,11)]+['f13','f14']
# ##2021-06-25
# print(Train_data['tradeTime'].head())
# print(Train_data['registerDate'].head())
# print(Train_data['licenseDate'].head())
# print(Train_data['f7'].head())
# print(Train_data['f15'].head())

# ##201204.0
# print(Train_data['f13'].head())



from tqdm import tqdm
#时间特征处理
def date_proc(x):
    # m = int(x[4:6])
    # if m == 0:
        # m = 1
    # return x[:4] + '-' + str(m) + '-' + x[6:]
    return x

def num_to_date(df,date_cols):
    for f in tqdm(date_cols):
        df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
        df[f + '_year'] = df[f].dt.year
        df[f + '_month'] = df[f].dt.month
        df[f + '_day'] = df[f].dt.day
    return df
Train_data = num_to_date(Train_data,date_features)
TestA_data = num_to_date(TestA_data,date_features)

Train_data=Train_data.drop(['f11','f12','tradeTime','registerDate','licenseDate'],axis=1)
TestA_data=TestA_data.drop(['f11','f12','tradeTime','registerDate','licenseDate'],axis=1)
from scipy.stats import mode
#类别统计
def sta_cate(df,cols):
    sta_df = pd.DataFrame(columns = ['column','nunique','miss_rate','most_value','most_value_counts','max_value_counts_rate'])
    for col in cols:
        count = df[col].count()
        nunique = df[col].nunique()
        miss_rate = (df.shape[0] - count) / df.shape[0]
        most_value = df[col].value_counts().index[0]
        most_value_counts = df[col].value_counts().values[0]
        max_value_counts_rate = most_value_counts / df.shape[0]
        
        sta_df = sta_df.append({'column':col,'nunique':nunique,'miss_rate':miss_rate,'most_value':most_value,
                                'most_value_counts':most_value_counts,'max_value_counts_rate':max_value_counts_rate},ignore_index=True)
    return sta_df 
##统计信息
# stati_df=sta_cate(Train_data,categorical_features)
# print(stati_df)

#补齐数据
features1=['carCode','country','maketype','modelyear','gearbox','f1','f8','f9','f10','f13']
features2=['country','maketype','modelyear','f1','f8','f9','f10','f13']
def put_data(df,cols):
    for col in cols:
       max_num1  = df[col].value_counts().values[0]
       df[col].fillna(max_num1,inplace=True)
    return df
Train_data=put_data(Train_data, features1)
TestA_data=put_data(TestA_data, features2)

##数据信息
# Train_data.info()
# TestA_data.info()
display(Train_data.head())
display(TestA_data.head())

# print(x_train)
# print(Y_data)

##模型训练
def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators = 150)
    param_grid = {
        'learning_rate': [0.1],#
    }
    gbm = GridSearchCV(estimator, param_grid)

    gbm.fit(x_train, y_train)
    print(f"Best:  {gbm}" )

    return gbm
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7) #,objective ='reg:squarederror'
    
from sklearn.model_selection import KFold

def cv_predict(model,X_data,Y_data,X_test,sub):

    oof_trn = np.zeros(X_data.shape[0])
    oof_val = np.zeros(X_data.shape[0])
    feature_importance_df = pd.DataFrame()
    ## 5折交叉验证方式
    kf = KFold(n_splits=5,shuffle=True, random_state=0)
    for idx, (trn_idx,val_idx) in enumerate(kf.split(X_data,Y_data)):
        print('--------------------- {} fold ---------------------'.format(idx))
        trn_x, trn_y = X_data.iloc[trn_idx].values, Y_data.iloc[trn_idx]
        val_x, val_y = X_data.iloc[val_idx].values, Y_data.iloc[val_idx]

        xgr.fit(trn_x,trn_y,eval_set=[(val_x, val_y)],eval_metric='mae',verbose=30,early_stopping_rounds=20,)

        oof_trn[trn_idx] = xgr.predict(trn_x)
        oof_val[val_idx] = xgr.predict(val_x)
        sub['price'] += xgr.predict(X_test.values) / kf.n_splits

        pred_trn_xgb=xgr.predict(trn_x)
        pred_val_xgb=xgr.predict(val_x)
        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_data.columns.to_list()
        fold_importance_df["importance"] = xgr.feature_importances_#(importance_type='gain')
        fold_importance_df["fold"] = idx
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)     
        print('trn mae:', mean_absolute_error(trn_y, oof_trn[trn_idx]))
        print('val mae:', mean_absolute_error(val_y, oof_val[val_idx]))
        if idx==0:
            break
    feature_importance_df.sort_values(by='importance',inplace=True)
    return model,oof_trn,oof_val,sub,feature_importance_df
# #训练集和测试集
# Y_train=Train_data['price']
# X_train=Train_data.drop(['price'],axis=1)
# X_test=TestA_data.drop(['price'],axis=1)

# x_train=X_train.iloc[:25000,:].copy()
# x_test=X_train.iloc[25000:,:].copy()
# y_train=Y_train[:25000].copy()
# y_test=Y_train[25000:].copy()


# sub2 = X_test[['carid']].copy()
# sub2['price'] = 0
# model2,oof_trn,oof_val,sub2,feature_importance_df = cv_predict(xgr,x_train,y_train,x_test,sub2)
# print('Train mae:',mean_squared_error(y_train,oof_trn))
# print('Val mae:', mean_squared_error(y_train,oof_val))
# select = feature_importance_df

# Mape=0
# count=0
# # print(y_test.iloc[1])


# for i in range(len(sub2)):
#     # if(y_test[i][1]!=0):
          
#           Ape=abs((sub2['price'][i]-y_test.iloc[i])/y_test.iloc[i])
#           if Ape<=0.05:
#               count+=1
#           Mape+=Ape
# Accuracy=count/len(sub2)
# Mape=Mape/len(sub2)
# Acc=0.2*(1-Mape)+0.8*Accuracy
# print(Mape,Accuracy,Acc)
# # print(sub2)
# # print('Y:',y_test)
# # display(feature_importance_df[feature_importance_df['importance']>0].head(10))
# # display(feature_importance_df[feature_importance_df['importance']>0])










