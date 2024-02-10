#!/usr/bin/env python3
import pandas as pd
import numpy as np
from matplotlib.dates import date2num
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials
import btalib as bl
import pickle


coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
    ]

mins = [
    15460, 0.239, 0.24, 1606, 0.036,4,0.11,183, 0.315,5, 6,45.6, 32.31,
    869, 8, 2.4, 9.32, 1, 0.05, 1 , 0.015, 1.23, 5.5, 0.003, 0.0153, 5.94, 0.05, 0.275, 3.1, 96.5, 0.37, 5.1, 0.036, 2.48, 21.6, 0.230, 1.1
    ]
tops = [
    69020,3.1,6.68,2070, 0.95,55.13,2.95,693,2.94,53.1,91.24, 666.7,544.7,
    4867, 260, 238, 147, 60, 1.37, 21.3, 0.842, 20.6, 44.8, 0.09, 0.26, 141, 2.9, 8, 45, 520, 8.5, 9.22, 0.167, 44.2,  625, 4.85, 19.23
    ]

mins_dict = dict(zip(coins, mins))
tops_dict = dict(zip(coins, tops))


def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    #df.set_index('date', inplace=True)
    #df.index = pd.to_datetime(df.index, unit='ms')
    return df

def norm_augment(df) :
    #for the gradient d1 and d2
    x = np.array([date2num(d) for d in  df.index])
    macd = bl.macd(df, pfast = 8, pslow = 63, psignal = 11)
    rsi = bl.rsi(df, period = 8)
    df = df.join([macd.df, rsi.df])
    # Adding the indicators : shifting brings the data foreward from the past to present observation
    #-------- Last candle is from an hour ago, shift 1 is from 2 h ago ...etc
    #---------last 9 candles------
    df.loc[:, "movement"] = [1 if df.loc[i, "open"] < df.loc[i, "close"] else 0 for i in df.index]
    df.loc[:, "past_2_movement"] = (df.loc[:, "movement"].shift(periods=1))
    df.loc[:, "past_3_movement"] = (df.loc[:, "movement"].shift(periods=2))
    df.loc[:, "past_4_movement"] = (df.loc[:, "movement"].shift(periods=3))
    df.loc[:, "past_5_movement"] = (df.loc[:, "movement"].shift(periods=4))
    df.loc[:, "past_6_movement"] = (df.loc[:, "movement"].shift(periods=5))
    df.loc[:, "past_7_movement"] = (df.loc[:, "movement"].shift(periods=6))
    df.loc[:, "past_8_movement"] = (df.loc[:, "movement"].shift(periods=7))
    df.loc[:, "past_9_movement"] = (df.loc[:, "movement"].shift(periods=8))
    #----------------------------------------------------------------------- 
    df.loc[:,'h_2'] = df.loc[:,'histogram'].shift(periods=1)
    df.loc[:,'h_3'] = df.loc[:,'histogram'].shift(periods=2)
    df.loc[:,'h_4'] = df.loc[:,'histogram'].shift(periods=3)
    df.loc[:,'h_5'] = df.loc[:,'histogram'].shift(periods=4)

    df.loc[:, 'c_2'] = df.loc[:,'close'].shift(periods=1)
    df.loc[:, 'o_2'] = df.loc[:,'open'].shift(periods=1)
    df.loc[:, 'hi_2'] = df.loc[:,'high'].shift(periods=1)
    df.loc[:, 'l_2'] = df.loc[:,'low'].shift(periods=1)

    df.loc[:, 'c_3'] = df.loc[:,'close'].shift(periods=2)
    df.loc[:, 'c_4'] = df.loc[:,'close'].shift(periods=3)
    df.loc[:, 'c_5'] = df.loc[:,'close'].shift(periods=4)
    df.loc[:, 'c_6'] = df.loc[:,'close'].shift(periods=5)

    y = df.loc[:,'histogram'].values
    df.loc[:, "d1"] = np.gradient(y, x)
    df.loc[:, "d2"] = np.gradient(df.loc[:, "d1"].values, x)

    y = df.loc[:,'close'].values
    df.loc[:, "c1"] = np.gradient(y, x)
    df.loc[:, "c2"] = np.gradient(df.loc[:, "c1"].values, x)
    
    y = df.loc[:,'rsi'].values
    df.loc[:, "r1"] = np.gradient(y, x)
    df.loc[:, "r2"] = np.gradient(df.loc[:, "r1"].values, x)

    # ADDING TARGET VAR
    # the target variable is the colour of PREZENT the candle that just opened
    # Shift -1 brings data back one candle. 
    df.loc[:, "y_pred"] = df.loc[:, "movement"].shift(periods= -1)

    df.dropna(inplace = True)
    #df = df.drop(columns=['open', 'high', 'close', "low", "macd", "signal"], inplace = True)

    return df

of_interest = ["BTC", "MINA",  "AGIX", "ALGO", "MATIC", "KAVA", "FIL", "RNDR", "ADA", "BNB", "AR"]
pairs = [f'{coin}BUSD' for coin in of_interest]


data_files = [f'./data/h/{coin}BUSD.csv' for coin in of_interest]
# Load data
c_data = [import_coin_data(pth) for pth in data_files]
augmented_data = [norm_augment(df = c_data[p], coin = of_interest[p]) for p in range(len(pairs)) ]  
c_dict = dict(zip(pairs, augmented_data))

# TRAIN on all data #############################################################



xgb_model = xgb.XGBClassifier(n_estimators = 838,
                              colsample_bytree = 0.88,
                              gamma = 1,
                              max_depth = 4, 
                              learning_rate = 0.4,
                              objective='binary:logistic',
                              eval_metric = 'logloss',
                              reg_alpha = 7,
                              reg_lambda = 8,
                              subsample = 0.61,
                              use_label_encoder = False )


for pair, df in c_dict.items():
    X = df.iloc[:, :-1].values # Features
    y = df.iloc[:, -1].values # Target variable
    xgb_model.fit(X, y)
    with open(f'./models/{pair}_xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f'{pair} XGB model saved!')
""" 
#########################################################################################

# TRAIN / TEST on BTC ###################################################################

# Define training data


df = augmented_data[0]


# Split data into training and testing sets
X = df.iloc[:, :-1].values # Features
y = df.iloc[:, -1].values # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators = 837,
                              colsample_bytree = 0.63,
                              gamma = 3,
                              max_depth = 6, 
                              learning_rate = 0.05,
                              objective='binary:logistic',
                              eval_metric = 'error',
                              reg_alpha = 9.0,
                              reg_lambda = 0.0,
                              subsample = 0.470,
                              use_label_encoder = False )

xgb_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = xgb_model.predict(X_test)
#y_prob = xgb_model.predict_proba(X_test)
#print(y_pred[0:5])
#print(y_prob[0:5])


# Evaluate model performance
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification report:\n', classification_report(y_test, y_pred))
exit()

#######################################################################################################



# GRID SEARCH FINE TUNING##############################################################

# Split data into training and testing sets

df = augmented_data[0]

X = df.iloc[:, :-1].values # Features
y = df.iloc[:, -1].values # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define search space for hyperparameters
space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000)),
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 10, 1),
    'reg_alpha': hp.quniform('reg_alpha', 0, 10, 1),
    'reg_lambda': hp.quniform('reg_lambda', 0, 10, 1),
}

# Define objective function to minimize
def objective(params):
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    print(1 - accuracy_score(y_test, y_pred))
    return 1 - accuracy_score(y_test, y_pred)

# Perform hyperparameter tuning
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

# Print best hyperparameters
print(best)

"""
########################################################################################