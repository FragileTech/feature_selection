from itertools import product
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import param
from pycaret.classification import (
    compare_models,
    create_model,
    get_config,
    predict_model,
    setup,
    tune_model,
)
from pycaret.utils import check_metric

from feature_selection.constants import (
    FILTER_METRIC,
    METRICS_LIST,
    MODEL_CLASS_TO_NAME,
    NUMERICS,
    OPT_LIST,
    SETUP_KWARGS,
)
from feature_selection.functions import calculate_number_features, compute_numeric_features, normalize, feature_score
from feature_selection.run_pycaret_setup import run_pycaret_setup


class FeatSel(param.Parameterized):
    # Init values
    # Feature selection parameters
    target = param.String("goal_2.5", doc="Target feature. Feature to be predicted.")
    number_features = param.Number(
        0.5,
        bounds=(0, 1),
        inclusive_bounds=(False, False),
        doc="Number of features (percentage) selected each iteration. Only the first nth "
        "features will be kept for the next iteration.",
    )
    target_features = param.Number(
        0.3,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Final total number of features. The goal of the package is to reduce "
        "the incoming columns of the dataset to this 'target_features' number.",
    )
    percentage_reduction = param.Number(
        0.65,
        bounds=(0, 1),
        inclusive_bounds=(False, False),
        doc="Number of features (percentage) selected from each model.",
    )
    iterations = param.Number(
        1, bounds=(0, 10), doc="Number of steps the process will be repeated."
    )
    # Metric parameters
    filter_metrics = param.Dict(
        FILTER_METRIC, doc="Metric thresholds. Minimum acceptance value of a model"
    )
    # Model setup and model optimization parameters
    numerics = param.List(NUMERICS, doc="List including the numeric types.")
    ignore_features = param.List(
        default=[], allow_None=True, doc="Features to be ignored during the process."
    )
    include = param.List(
        default=None,
        item_type=str,
        allow_None=True,
        doc="List of Machine Learning models to be used during the process.",
    )
    exclude = param.List(
        ["qda", "knn", "nb"],
        item_type=str,
        doc="List of Machine Learning models not considered during the process.",
    )
    setup_kwargs = param.Dict(SETUP_KWARGS, doc="Configuration dictionary of the model setup.")
    sort = param.String("AUC", doc="Establishes how the models are sorted.")
    number_models = param.Integer(10, bounds=(2, 13), doc="Number of models to be selected")
    optimize = param.Boolean(False, doc="Start a optimization process for the selected models.")
    opt_list = param.List(
        OPT_LIST, item_type=str, doc="List containing the parameters to be optimize."
    )
    opt_kwargs = param.Dict(
        {}, doc='Additional parameters passed to `pycaret.tune_model` function.'
    )
    # Class selectors
    _flag = param.Boolean(True)
    _top_models = param.List(default=None, allow_None=True)
    _dataset = param.ClassSelector(class_=pd.DataFrame)
    _dict_models = param.ClassSelector(class_=dict)
    _tune_dict_models = param.ClassSelector(class_=dict)
    _x_train = param.ClassSelector(class_=pd.DataFrame)
    _x_df = param.DataFrame(pd.DataFrame())
    _model_df = param.ClassSelector(class_=pd.DataFrame)
    _model_tuned_df = param.ClassSelector(class_=pd.DataFrame)
    _features_df = param.ClassSelector(class_=pd.DataFrame)


class FeatureSelection(FeatSel):
    def __init__(self, dataset: pd.DataFrame, **kwargs):
        # Copy of the incoming dataset
        dataset = dataset.copy()
        # Compute the upper bound of number_features, target_features, number_models
        total_features = dataset.shape[1]
        self.param.target_features.bounds = (0, total_features)
        self.param.number_features.bounds = (0, total_features - 1)
        if "include" in kwargs:
            self.param.number_models.default = len(kwargs["include"])
            self.param.number_models.bounds = (0, len(kwargs["include"]))
        # Call super
        super(FeatureSelection, self).__init__(_dataset=dataset, **kwargs)
        # Get the features of the dataframe
        self.feature_list = self.dataset.columns.tolist()
        self.feature_list.remove(self.target)  # target column should not be counted
        # Compute target features
        self.target_features = calculate_number_features(
            number_features=self.target_features, features=self.feature_list
        )
        # Compute how the selected features are reduced
        self._reduction = (
            self.number_features
            if ("iterations" not in kwargs) and ("number_features" in kwargs)
            else (len(self.feature_list) - self.target_features) / self.iterations
        )
        # Get the evaluator and the arguments. Depends on the "include" parameter
        self._training_function, self._args = self._decide_model_eval()
        # Get all the columns whose type is numeric
        self.numeric_features = compute_numeric_features(
            include=self.numerics, df=self.dataset[self.feature_list]
        )

    @property
    def dataset(self):
        return self._dataset

    @property
    def x_df(self):
        return self._x_df

    @property
    def x_train(self):
        return self._x_train

    @property
    def dict_models(self):
        return self._dict_models

    @property
    def tune_dict_models(self):
        return self._tune_dict_models

    @property
    def features_df(self):
        return self._features_df

    def _decide_model_eval(self):
        """
        Define the pycaret model evaluator depending on the number of included models.

        If the 'include' list parameter equals 1, the method will return
        the 'create_models' pycaret object.
        If 'include' parameter list is greater than 1, the method will
        return the 'compare_model' pycaret object and its arguments.
        If 'include' parameter equals None, the method will return the
        'compare_models' pycaret object, where all possible models are
        considered for evaluation, except those included within the 'exclude'
        list.
        """
        args = {"n_select": self.number_models, "sort": self.sort, "verbose": False}
        training_function = compare_models
        if not self.include:
            args["exclude"] = self.exclude
        elif len(self.include) == 1:
            training_function = lambda *rgs, **kwargs: [create_model(*rgs, **kwargs)]
            args = {"estimator": self.include[0], "verbose": False}
        else:
            args["include"] = self.include
        return training_function, args

    def train_model(self):
        """Preprocess the data and select self.number_models top models."""
        # Selected dataset
        selected_cols = self.feature_list + [self.target]
        train_data = self.dataset[selected_cols]  # if self.x_df.empty else self.x_df[selected_cols].copy()
        # Numeric features
        self.setup_kwargs["numeric_features"] = [
            c for c in self.numeric_features if c in self.feature_list
        ]
        # Ignore features
        self.setup_kwargs["ignore_features"] = [
            c for c in self.ignore_features if c in self.feature_list
        ]
        # Initialize pycaret setup
        setup(data=train_data, target=self.target, **self.setup_kwargs)
        # Get train dataset and preprocessed dataframe
        self._x_train = get_config("X_train")
        if self._flag:
            self._dataset = pd.concat([get_config("X"), get_config("y")], axis=1)
            self.setup_kwargs["preprocess"] = False  # Turn off preprocessing
            self._flag = False
        # Compare models
        self._top_models = self._training_function(**self._args)

    def create_dict_models(self):
        """Create a dictionary whose values are pycaret standard models."""
        self._dict_models = {
            str(top_model).split("(")[0]: top_model for top_model in self._top_models
        }
        # Remove bad catboost key
        oldkey = [key for key in self._dict_models.keys() if key.startswith("<catboost")]
        if oldkey:
            self._dict_models["CatBoostClassifier"] = self._dict_models.pop(oldkey[0])
        # Remap
        self._dict_models = {
            MODEL_CLASS_TO_NAME[key]: self._dict_models[key] for key in self._dict_models.keys()
        }

    def create_dict_tuned_models(self):
        """Create a dictionary whose keys and values are pycaret tuned models."""
        self._tune_dict_models = {}
        for (model_str, py_model), optimize in product(self._dict_models.items(), self.opt_list):
            self._tune_dict_models[f"{model_str}_tune_{optimize}"] = tune_model(
                py_model,
                optimize=optimize,
                verbose=False,
                **self.opt_kwargs
            )

    def get_metrics_df(self, test_predicted, model, dataframe):
        """Compute different metric values for the given model."""
        value_dct = dict()
        for metric in METRICS_LIST:
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
        top_n_features = calculate_number_features(
            number_features=self.percentage_reduction,
            features=df,
        )
        return df.iloc[:top_n_features]

    def extract_features(self, models: List, dict_models: Dict):
        """Update self.features_df with the most relevant features used by the given model."""
        for model in models:  # model extracted from dataframe
            # Check
            if (
                model not in dict_models.keys()
            ):  # check no errors have been produced during operations
                raise KeyError(f"The selected model: {model} is not listed in dict_models.keys()")
            df_conc = self.filter_best_features(key_model=model, models_dict=dict_models)
            self._features_df = pd.concat([self._features_df, df_conc])

    def compute_metrics_df(self):
        """Update self.features_df with the most relevant features used by the standard models."""
        self._model_df = pd.DataFrame(
            data=[], index=self._dict_models.keys(), columns=METRICS_LIST
        )
        for model, py_model in self._dict_models.items():
            predict = predict_model(py_model)
            self._model_df = self.get_metrics_df(
                test_predicted=predict,
                model=model,
                dataframe=self._model_df,
            )
        self._model_df = self.remove_bad_models(dataframe=self._model_df)
        self.extract_features(models=self._model_df.index.tolist(), dict_models=self._dict_models)

    def filter_tuned_duplicate(self):
        """Remove tuned models with identical metrics."""

        def drop(x):
            x.drop_duplicates(subset=METRICS_LIST, keep="first", inplace=True)
            return x

        df = self._model_tuned_df.groupby("model").apply(drop)
        df = (
            df.drop(columns="model").reset_index(level="model")
            if isinstance(df.index, pd.MultiIndex)
            else df
        )
        return df

    def tune_df(self):
        """Update self.features_df with the most relevant features used by tuned models."""
        # Tune dataframe
        self._model_tuned_df = pd.DataFrame(
            data=[],
            index=self.tune_dict_models.keys(),
            columns=["model"] + METRICS_LIST,
        )
        # Model entry
        for prim_model in self._dict_models.keys():
            ix = [ind.startswith(prim_model) for ind in self._model_tuned_df.index]
            self._model_tuned_df.loc[ix, "model"] = prim_model
        # Fill dataframe
        for model, py_model in self._tune_dict_models.items():
            predict = predict_model(py_model)
            self._model_tuned_df = self.get_metrics_df(
                test_predicted=predict, model=model, dataframe=self._model_tuned_df
            )
        # Remove duplicate and filter
        self._model_tuned_df = self.filter_tuned_duplicate()
        self._model_tuned_df = self.remove_bad_models(dataframe=self._model_tuned_df)
        # Get features
        self.extract_features(models=self._model_tuned_df.index.tolist(), dict_models=self._tune_dict_models)

    def run_feature_extraction(self):
        """Update self.features_df with the most relevant features used by each model."""
        # Initialize feature dataframe and train model
        self._features_df = pd.DataFrame(data=[], columns=["model_id", "model", "feature", "score"])
        self.train_model()
        # Run standard models
        self.create_dict_models()
        self.compute_metrics_df()
        # Run tuned models
        if self.optimize:
            self.create_dict_tuned_models()
            if not bool(self._tune_dict_models):
                raise ValueError("The tune dictionary is empty!")
            self.tune_df()
        # Return the list containing the features and their score
        self._features_df.index.name = "index_rem"
        return self._features_df.reset_index().drop(columns="index_rem")

    def remove_zeros(self):
        """Remove non-relevant features (those with a zero score)."""
        ix = self._features_df["score"] <= 0
        self._features_df.drop(index=self._features_df.loc[ix].index, inplace=True)
        self._features_df.reset_index(drop=True, inplace=True)

    def create_feature_list(self):
        """Run all necessary methods to extract the list of relevant features."""
        # Call creation features dataframe
        self._features_df = self.run_feature_extraction()
        # Remove zeros and normalize
        self.remove_zeros()
        self._features_df = normalize(dataframe=self._features_df)
        # Get score
        scoreboard = feature_score(dataframe=self._features_df)
        top_n_features = calculate_number_features(
            number_features=self.number_features,
            features=scoreboard,
        )
        filtered = scoreboard.iloc[:top_n_features]
        self.feature_list = filtered.index.tolist()

    def repeat_pipeline(self):
        """Iterate over the process to create the feature list and repeat it self.repeat times."""
        while len(self.feature_list) > self.target_features:
            # Call iteration
            self.create_feature_list()
            if len(self.feature_list) <= 1:
                break
        return self.feature_list
