import numpy as np
import pandas as pd

from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from xgboost import callback


class Common_LGB_Modelling:
    """
    Train and test data should contain the same selected features for ML models.
    Train, test data and target should be the same data type. (Pandas or Numpy)
    """

    def __init__(self, model_class):
        self.model_class = model_class

    def train(self, x_tr, y_tr, params):

        model = self.model_class(**params)
        model.fit(x_tr, y_tr)

        return model

    def train_and_valid(self, x_tr, y_tr, x_val, y_val, params):

        callbacks = [
            early_stopping(stopping_rounds=50, first_metric_only=True),
            log_evaluation(100),
        ]

        model = self.model_class(**params)
        model = model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], callbacks=callbacks)
        valid_pred = model.predict(x_val)

        return model, valid_pred

    def test(self, models, test):
        test_pred = [model.predict(test) for model in models]
        test_pred = np.mean(test_pred, axis=0)
        return test_pred

    def test_by_batch(self, models, test, batch_size):
        test_pred_all = []
        for idx in range(0, len(test), batch_size):
            test_pred_batch = [
                model.predict(test.iloc[idx : idx + batch_size]) for model in models
            ]
            test_pred_batch = np.mean(test_pred_batch, axis=0)
            test_pred_all.append(test_pred_batch)
        return np.concatenate(test_pred_all)

    def numpy_test_by_batch(self, models, test, batch_size):
        test_pred_all = []
        for idx in range(0, len(test), batch_size):
            test_pred_batch = [
                model.predict(test[idx : idx + batch_size]) for model in models
            ]
            test_pred_batch = np.mean(test_pred_batch, axis=0)
            test_pred_all.append(test_pred_batch)
        return np.concatenate(test_pred_all)


class Common_CB_Modelling:
    """
    Train and test data should contain the same selected features for ML models.
    Train, test data and target should be the same data type. (Pandas or Numpy)
    """

    def __init__(self, model_class):
        self.model_class = model_class

    def train(self, x_tr, y_tr, params, cat):

        train_pool = Pool(data=x_tr, label=y_tr, cat_features=cat)

        model = self.model_class(**params)
        model.fit(train_pool)

        return model

    def train_and_valid(self, x_tr, y_tr, x_val, y_val, params, cat):
        train_pool = Pool(data=x_tr, label=y_tr, cat_features=cat)
        valid_pool = Pool(data=x_val, label=y_val, cat_features=cat)

        model = self.model_class(**params)

        if isinstance(model, CatBoostClassifier):
            model.fit(
                train_pool,
                eval_set=[valid_pool],
                early_stopping_rounds=50,
                verbose_eval=100,
            )
            valid_pred = model.predict_proba(x_val)[:, 1]
        elif isinstance(model, CatBoostRegressor):
            model.fit(
                train_pool,
                eval_set=[valid_pool],
                early_stopping_rounds=50,
                verbose_eval=100,
            )
            valid_pred = model.predict(x_val)

        return model, valid_pred

    def test(self, models, test):
        if isinstance(models[0], CatBoostClassifier):
            test_pred = [model.predict_proba(test)[:, 1] for model in models]
        elif isinstance(models[0], CatBoostRegressor):
            test_pred = [model.predict(test) for model in models]
        test_pred = np.mean(test_pred, axis=0)
        return test_pred

    def test_by_batch(self, models, test, batch_size):
        test_pred_all = []
        for idx in range(0, len(test), batch_size):
            if isinstance(models[0], CatBoostClassifier):
                test_pred_batch = [
                    model.predict_proba(test.iloc[idx : idx + batch_size])[:, 1]
                    for model in models
                ]
            elif isinstance(models[0], CatBoostRegressor):
                test_pred_batch = [
                    model.predict(test.iloc[idx : idx + batch_size]) for model in models
                ]
            test_pred_batch = np.mean(test_pred_batch, axis=0)
            test_pred_all.append(test_pred_batch)
        return np.concatenate(test_pred_all)

    def numpy_test_by_batch(self, models, test, batch_size):
        test_pred_all = []
        for idx in range(0, len(test), batch_size):
            if isinstance(models[0], CatBoostClassifier):
                test_pred_batch = [
                    model.predict_proba(test[idx : idx + batch_size])[:, 1]
                    for model in models
                ]
            elif isinstance(models[0], CatBoostRegressor):
                test_pred_batch = [
                    model.predict(test[idx : idx + batch_size]) for model in models
                ]
            test_pred_batch = np.mean(test_pred_batch, axis=0)
            test_pred_all.append(test_pred_batch)
        return np.concatenate(test_pred_all)


class Common_XGB_Modelling:
    """
    Train and test data should contain the same selected features for ML models.
    Train, test data and target should be the same data type. (Pandas or Numpy)
    """

    def __init__(self, model_class):
        self.model_class = model_class

    def train_and_valid(self, x_tr, y_tr, x_val, y_val, params):

        model = self.model_class(
            **params, callbacks=[callback.EvaluationMonitor(rank=0, period=100)]
        )
        model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)])
        valid_pred = model.predict(x_val, iteration_range=(0, model.best_iteration))

        return model, valid_pred

    def test(self, models, test):
        test_pred = [
            model.predict(test, iteration_range=(0, model.best_iteration))
            for model in models
        ]
        test_pred = np.mean(test_pred, axis=0)
        return test_pred


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df
