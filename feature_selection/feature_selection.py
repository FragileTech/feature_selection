from itertools import product
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import param
from pycaret.classification import compare_models, get_config, predict_model, tune_model
from pycaret.utils import check_metric

from feature_selection.run_pycaret_setup import run_pycaret_setup


class FeatureSelection(param.Parameterized):

    # Class attributes
    model_class_to_name = {
        "RidgeClassifier": "ridge",
        "LogisticRegression": "lr",
        "LinearDiscriminantAnalysis": "lda",
        "GradientBoostingClassifier": "gbc",
        "QuadraticDiscriminantAnalysis": "qda",
        "LGBMClassifier": "lightgbm",
        "AdaBoostClassifier": "ada",
        "RandomForestClassifier": "rf",
        "ExtraTreesClassifier": "et",
        "GaussianNB": "nb",
        "DecisionTreeClassifier": "dt",
        "KNeighborsClassifier": "knn",
        "SGDClassifier": "svm",
        "CatBoostClassifier": "catboost",
        "SVC": "rbfsvm",
        "GaussianProcessClassifier": "gpc",
        "MLPClassifier": "mlp",
        "XGBClassifier": "xgboost",
    }

    metrics_list = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]

    # Private class attributes
    _filter_metric = {
        "Accuracy": 0.5,
        "AUC": 0.5,
        "Recall": 0.6,
        "Precision": 0.6,
        "F1": 0.6,
        "Kappa": 0.1,
        "MCC": 0.1,
    }

    _setup_kwargs = dict(
        preprocess=True,
        train_size=0.75,
        # test_data=test_data,
        session_id=123,
        normalize=True,
        transformation=True,
        ignore_low_variance=True,
        remove_multicollinearity=False,
        multicollinearity_threshold=0.4,
        n_jobs=-1,
        use_gpu=False,
        profile=False,
        ignore_features=None,
        fold_strategy="timeseries",
        remove_perfect_collinearity=True,
        create_clusters=False,
        fold=4,
        feature_selection=False,
        # you can use this to keep the 95 % most relevant features (fat_sel_threshold)
        feature_selection_threshold=0.4,
        combine_rare_levels=False,
        rare_level_threshold=0.02,
        pca=False,
        pca_method="kernel",
        pca_components=30,
        polynomial_features=False,
        polynomial_degree=2,
        polynomial_threshold=0.01,
        trigonometry_features=False,
        remove_outliers=False,
        outliers_threshold=0.01,
        feature_ratio=False,
        feature_interaction=False,
        # Makes everything slow AF. use to find out possibly interesting features
        interaction_threshold=0.01,
        fix_imbalance=False,
        log_experiment=False,
        verbose=False,
        silent=True,
        experiment_name="lagstest",
    )

    _numerics = ["int16", "int32", "int64", "float16", "float32", "float64", "int", "float"]

    # Init values
    ## Feature selection parameters
    target = param.String("goal_2.5")
    number_features = param.Number(.5, bounds=(0, None), inclusive_bounds=(False, True))
    target_features = param.Number(0.3, bounds=(0, None), inclusive_bounds=(False, True))
    feature_division = param.Number(3, bounds=(1, 100))
    ## Metric parameters
    filter_metrics = param.Dict(_filter_metric)
    ## Model setup and model optimization parameters
    numerics = param.List(_numerics)
    ignore_features = param.List(default=None, allow_None=True)
    setup_kwargs = param.Dict(_setup_kwargs)
    include = param.List(default=None, item_type=str, allow_None=True)
    exclude = param.List(["qda", "knn", "nb"], item_type=str)
    sort = param.String("AUC")
    number_models = param.Integer(10, bounds=(2, 13))
    top_models = param.List(default=None, allow_None=True)
    optimize = param.Boolean(False)
    opt_list = param.List(["Accuracy", "Precision", "Recall", "F1", "AUC"], item_type=str)
    ## Class selectors
    user_df = param.ClassSelector(class_=pd.DataFrame)
    dict_models = param.ClassSelector(class_=dict)
    tune_dict_models = param.ClassSelector(class_=dict)
    x_train = param.ClassSelector(class_=pd.DataFrame)
    model_df = param.ClassSelector(class_=pd.DataFrame)
    model_tuned_df = param.ClassSelector(class_=pd.DataFrame)
    features_df = param.ClassSelector(class_=pd.DataFrame)

    def __init__(self, **kwargs):
        # Compute the upper bound of number_features and target_features
        if "user_df" in kwargs.keys():
            total_features = user_df.shape[1]
            self.param.number_features.bounds = (0, total_features)
            self.param.target_features.bounds = (0, total_features)
        else:
            raise AttributeError("You should provide a 'user_df' dataframe as input.")
        # Call super
        super(FeatureSelection, self).__init__(**kwargs)
        # Get the features of the dataframe
        self.feature_list = self.user_df.columns.tolist()
        self.feature_list.remove(self.target)  # target column should not be counted
        # Compute target features
        self.target_features = self.calculate_number_features(
            number_features=self.target_features, df=self.feature_list
        )

    def _compute_numeric_features(self, df: pd.DataFrame):
        """Return those columns from the given dataset whose data type is numeric."""
        return df.select_dtypes(include=self.numerics).columns.tolist()

    @staticmethod
    def calculate_number_features(
        number_features: Union[int, float], df: Union[pd.DataFrame, List]
    ) -> int:
        n_features = (
            int(number_features)
            if (number_features > 1)
            else int(number_features_is_pct * len(df))
        )
        return n_features

    def train_model(self):
        """Preprocess the data and select self.number_models top models."""
        # Selected dataset
        selected_cols = self.feature_list + [self.target]
        train_data = self.user_df[selected_cols]
        # Numeric features
        numeric_features = self._compute_numeric_features(
            df=train_data.drop(columns=[self.target])
        )
        self.setup_kwargs["numeric_features"] = numeric_features
        # Ignore features
        self.setup_kwargs["ignore_features"] = self.ignore_features
        # Initialize pycaret setup
        run_pycaret_setup(train_data=train_data, target=self.target, **self.setup_kwargs)
        # Get train dataset and best models
        self.x_train = get_config("X_train")
        # Compare models
        compare_dict = {'exclude': self.exclude} if self.include is None else {'include': self.include}
        self.top_models = compare_models(
            n_select=self.number_models,
            sort=self.sort,
            **compare_dict,
            verbose=False,
        )

    def create_dict_models(self):
        """Create a dictionary whose values are pycaret standard models."""
        self.dict_models = {
            str(top_model).split("(")[0]: top_model for top_model in self.top_models
        }
        # Remove bad catboost key
        oldkey = [key for key in self.dict_models.keys() if key.startswith("<catboost")]
        if oldkey:
            self.dict_models["CatBoostClassifier"] = self.dict_models.pop(oldkey[0])
        # Remap
        self.dict_models = {
            self.model_class_to_name[key]: self.dict_models[key] for key in self.dict_models.keys()
        }

    def create_dict_tuned_models(self):
        """Create a dictionary whose keys and values are pycaret tuned models."""
        self.tune_dict_models = {}
        for (model_str, py_model), optimize in product(self.dict_models.items(), self.opt_list):
            self.tune_dict_models[f"{model_str}_tune_{optimize}"] = tune_model(
                py_model, optimize=optimize, verbose=False
            )

    def get_metrics_df(self, test_predicted, model, dataframe):
        """Compute different metric values for the given model."""
        value_dct = dict()
        for metric in self.metrics_list:
            try:
                value_dct[metric] = check_metric(
                    actual=test_predicted[self.target],
                    prediction=test_predicted["Label"],
                    metric=metric,
                )
            except AttributeError:
                value_dct[metric] = np.nan
        for key, val in value_dct.items():
            dataframe.loc[model, key] = val
        return dataframe

    def remove_bad_models(self, dataframe: pd.DataFrame):
        """Filter and remove the models whose metrics do not satisfy the given conditions."""
        remove_dict = dict()
        models = dataframe.index.tolist()
        for model, (metric, cond) in product(models, self.filter_metrics.items()):
            if dataframe.loc[model, metric] < cond:
                remove_dict[model] = metric
        remove_models = list(set(remove_dict.keys()))
        dataframe.drop(labels=remove_models, axis="index", inplace=True)
        return dataframe

    def filter_best_features(self, key_model: str, models_dict: Dict):
        """Compute the most relevant features used by the given model."""
        py_model = models_dict[key_model]
        cond = any([key_model.startswith(name) for name in ["lr", "lda", "ridge", "svm"]])
        score_metric = abs(py_model.coef_[0]) if cond else py_model.feature_importances_
        metrics_dict = {
            "model_id": key_model,
            "model": key_model.split("_")[0],
            "feature": self.x_train.columns,
            "score": score_metric,
        }
        df = pd.DataFrame(metrics_dict).sort_values(by="score", ascending=False)
        top_n_features = self.calculate_number_features(
            number_features=self.number_features, df=df
        )
        return df.iloc[:top_n_features]

    def extract_features(self, dataframe: pd.DataFrame, dict: Dict):
        """Update self.features_df with the most relevant features used by the given model."""
        models = dataframe.index.tolist()
        for model in models:  # model extracted from dataframe
            # Check
            if model not in dict.keys():  # check no errors have been produced during operations
                raise KeyError("The selected model is not listed in dict_models.keys()")
            df_conc = self.filter_best_features(key_model=model, models_dict=dict)
            self.features_df = pd.concat([self.features_df, df_conc])

    def compute_metrics_df(self):
        """Update self.features_df with the most relevant features used by the standard models."""
        self.model_df = pd.DataFrame(
            data=[], index=self.dict_models.keys(), columns=self.metrics_list
        )
        for model, py_model in self.dict_models.items():
            predict = predict_model(py_model)
            self.model_df = self.get_metrics_df(
                test_predicted=predict,
                model=model,
                dataframe=self.model_df,
            )
        self.model_df = self.remove_bad_models(dataframe=self.model_df)
        self.extract_features(dataframe=self.model_df, dict=self.dict_models)

    def filter_tuned_duplicate(self):
        """Remove tuned models with identical metrics."""

        def drop(x):
            x.drop_duplicates(subset=self.metrics_list, keep="first", inplace=True)
            return x

        df = self.model_tuned_df.groupby("model").apply(drop)
        return df.drop(columns="model").reset_index(level="model")

    def tune_df(self):
        """Update self.features_df with the most relevant features used by tuned models."""
        # Tune dataframe
        self.model_tuned_df = pd.DataFrame(
            data=[],
            index=self.tune_dict_models.keys(),
            columns=["model"] + self.metrics_list,
        )
        # Model entry
        for prim_model in self.dict_models.keys():
            ix = [ind.startswith(prim_model) for ind in self.model_tuned_df.index]
            self.model_tuned_df.loc[ix, "model"] = prim_model
        # Fill dataframe
        for model, py_model in self.tune_dict_models.items():
            predict = predict_model(py_model)
            self.model_tuned_df = self.get_metrics_df(
                test_predicted=predict, model=model, dataframe=self.model_tuned_df
            )
        # Remove duplicate and filter
        self.model_tuned_df = self.filter_tuned_duplicate()
        self.model_tuned_df = self.remove_bad_models(dataframe=self.model_tuned_df)
        # Get features
        self.extract_features(dataframe=self.model_tuned_df, dict=self.tune_dict_models)

    def run_feature_extraction(self):
        """Update self.features_df with the most relevant features used by each model."""
        # Initialize feature dataframe and train model
        self.features_df = pd.DataFrame(data=[], columns=["model_id", "model", "feature", "score"])
        self.train_model()
        # Run standard models
        self.create_dict_models()
        self.compute_metrics_df()
        # Run tuned models
        if self.optmize is True:
            self.create_dict_tuned_models()
            if not bool(self.tune_dict_models):
                raise ValueError("The tune dictionary is empty!")
            self.tune_df()
        # Return the list containing the features and their score
        self.features_df.index.name = "Index_rem"
        return self.features_df.reset_index().drop(columns="Index_rem")

    def remove_zeros(self):
        """Remove non-relevant features (those with a zero score)."""
        ix = self.features_df["score"] <= 0
        self.features_df.drop(index=self.features_df.loc[ix].index, inplace=True)
        self.features_df.reset_index(drop=True, inplace=True)

    @staticmethod
    def normalize(dataframe):
        """Normalize the pycaret score of each feature."""

        def norm(x):
            x["normal"] = x["score"] / x["score"].max()
            return x

        return (
            dataframe.groupby("model_id")
            .apply(norm)
            .sort_values(["model_id", "normal"], ascending=[True, False])
        )

    @staticmethod
    def feature_score(dataframe):
        """Assign the score to the selected features."""
        group = dataframe.groupby("feature")
        sorted_data = group.agg(
            counts=pd.NamedAgg(column="normal", aggfunc="count"),
            normal_sum=pd.NamedAgg(column="normal", aggfunc="sum"),
        ).sort_values(by=["counts", "normal_sum"], ascending=False)
        sorted_data["final_score"] = sorted_data["normal_sum"] / sorted_data["counts"]
        return sorted_data.sort_values("final_score", ascending=False)

    def create_feature_list(self):
        """Run all necessary methods to extract the list of relevant features."""
        # Call creation features dataframe
        self.features_df = self.run_feature_extraction()
        # Remove zeros and normalize
        self.remove_zeros()
        self.features_df = self.normalize(dataframe=self.features_df)
        # Get score
        self.features_df = self.feature_score(dataframe=self.features_df)
        top_n_features = self.calculate_number_features(
            number_features=self.number_features, df=self.features_df
        )
        filtered = self.features_df.iloc[:top_n_features]
        self.feature_list = filtered.index.tolist()

    def repeat_pipeline(self):
        """Iterate over the process to create the feature list and repeat it self.repeat times."""
        while len(self.feature_list) > self.target_features:
            # Call iteration
            self.create_feature_list()
            self.number_features = (
                int(self.number_features / self.feature_division)
                if self.number_features > 1
                else self.number_features
            )
            if len(self.feature_list) <= 1:
                break
        return self.feature_list
