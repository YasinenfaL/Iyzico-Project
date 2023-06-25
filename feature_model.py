import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')


# Function

def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.weekofyear
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby("merchant_id")['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")['Total_Transaction'].transform(
                    lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


def iyzico_prep():
    df = pd.read_csv("https://raw.githubusercontent.com/YasinenfaL/new/main/iyzico_data.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    df = create_date_features(df, "transaction_date")

    df = lag_features(df,
                      [91, 92, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
                       188, 189, 190,
                       350, 351, 352, 352, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368,
                       369, 370,
                       538, 539, 540, 541, 542,
                       718, 719, 720, 721, 722])

    df = roll_mean_features(df,
                            [91, 92, 178, 179, 180, 181, 182, 359, 360, 361, 449, 450, 451, 539, 540, 541, 629, 630,
                             631,
                             720])

    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [91, 92, 178, 179, 180, 181, 182, 359, 360, 361, 449, 450, 451, 539, 540, 541, 629, 630, 631, 720]

    df = ewm_features(df, alphas, lags)

    df["is_black_friday"] = 0
    df.loc[df["transaction_date"].isin(["2018-11-22", "2018-11-23", "2019-11-29", "2019-11-30"]), "is_black_friday"] = 1

    df["is_summer_solstice"] = 0
    df.loc[df["transaction_date"].isin(["2018-06-19", "2018-06-20", "2018-06-21", "2018-06-22",
                                        "2019-06-19", "2019-06-20", "2019-06-21",
                                        "2019-06-22"]), "is_summer_solstice"] = 1

    # Üye iş yerlerinin yıl ve ay bazında işlem sayılarının incelenmesi
    df.groupby(["merchant_id", "year", "month", "day_of_month"]).agg({"Total_Transaction": ["sum", "mean", "median"]})

    # Üye iş yerlerinin yıl ve ay bazında toplam ödeme miktarlarının incelenmesi
    df.groupby(["merchant_id", "year", "month"]).agg({"Total_Paid": ["sum", "mean", "median"]})

    df = pd.get_dummies(df, columns=['merchant_id', 'day_of_week', 'month'])
    df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)

    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    df["sales_ewm_alpha_09_lag_91"]

    # 2020'nin 10.ayına kadar train seti.
    train = df.loc[(df["transaction_date"] < "2020-10-01"), :]

    # 2020'nin son 3 ayı validasyon seti.
    val = df.loc[(df["transaction_date"] >= "2020-10-01"), :]

    cols = [col for col in train.columns if
            col not in ['transaction_date', 'id', "Total_Transaction", "Total_Paid", "year"]]

    Y_train = train['Total_Transaction']
    X_train = train[cols]

    Y_val = val['Total_Transaction']
    X_val = val[cols]

    return X_train, Y_train, Y_val, X_val, cols, val


def low_importance(X_train, Y_train, X_val, Y_val, lgb_params, threshold=1):
    lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
    lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

    model = lgb.train(lgb_params, lgbtrain,
                      valid_sets=[lgbtrain, lgbval],
                      num_boost_round=lgb_params['num_boost_round'],
                      early_stopping_rounds=lgb_params['early_stopping_rounds'],
                      feval=lgbm_smape,
                      verbose_eval=100)

    gain = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    low_importance_features = feat_imp[feat_imp['gain'] < threshold]

    new_X = X_train.drop(low_importance_features['feature'].tolist(), axis=1)
    new_X_val = X_val.drop(low_importance_features["feature"].tolist(), axis=1)

    return new_X, model, new_X_val


def final_model(X_train, Y_train, X_val, Y_val, lgb_params, smape, threshold=1):
    new_X, model, new_X_val = low_importance(X_train, Y_train, X_val, Y_val, lgb_params, threshold)

    lgbtrain = lgb.Dataset(data=new_X, label=Y_train, feature_name=list(new_X.columns))
    lgbval = lgb.Dataset(data=new_X_val, label=Y_val, reference=lgbtrain, feature_name=list(new_X.columns))

    fınal_model = lgb.train(lgb_params, lgbtrain,
                            valid_sets=[lgbtrain, lgbval],
                            num_boost_round=lgb_params['num_boost_round'],
                            early_stopping_rounds=lgb_params['early_stopping_rounds'],
                            feval=lgbm_smape,
                            verbose_eval=100
                            )

    y_pred_val = fınal_model.predict(new_X_val, num_iteration=model.best_iteration)
    smape = smape(np.expm1(y_pred_val), np.expm1(Y_val))

    print("SMAPE Skoru:", smape)
    return fınal_model, y_pred_val, new_X

########################
#  Model
########################

# Load Prep
X_train, Y_train, Y_val, X_val, cols, val = iyzico_prep()

# Parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

# Load Final Model
fınal_model, y_pred_val, new_X = final_model(X_train, Y_train, X_val, Y_val, lgb_params, smape, threshold=1)


plot_lgb_importances(fınal_model, True, num=10)

# Save Model
joblib.dump(fınal_model, "iyizco_model1.pkl")

iyzico_load = joblib.load("iyizco_model1.pkl")
