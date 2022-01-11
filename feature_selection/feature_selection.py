from itertools import product
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import param
from pycaret.classification import compare_models, get_config, predict_model, tune_model
from pycaret.utils import check_metric



#from ml_bets.modeling.match_model import PipelineDatasets, run_pycaret_setup


class FeatureSelection(param.Parameterized):

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

    setup_kwargs = dict(
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

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64", "int", "float"]

    # Init values
    target = param.String('goal_2.5')
    number_features = param.Number(50, bounds=(0, None))
    target_features = param.Number(0.3, bounds=(0, None))
    feature_division = param.Number(3, bounds=(1, 100))
    filter_metric =

    dict_models = param.ClassSelector(class_=dict)
    tune_dict_models = param.ClassSelector(class_=dict)
    model_df = param.ClassSelector(class_=pd.DataFrame)
    model_tuned_df = param.ClassSelector(class_=pd.DataFrame)
    features_df = param.ClassSelector(class_=pd.DataFrame)


    def __init__(
        self,



        feature_division: Optional[int] = 3,
        filter_metric: Optional[Dict] = None,
        optimize: Optional[bool] = False,
        exclude: Optional[List] = ["qda", "knn", "nb"],
        number_models: Optional[int] = 10,
        sort: Optional[str] = "AUC",
        odds_features: Optional[bool] = True,
        odds_ranking: Optional[bool] = True,
        summary: Optional[bool] = True,
        setup_kwargs: Optional[Dict] = None,
        test_size: Optional[str] = "01-Oct-2021",
        features_path: str = FEATURES_PATH,
        numeric_list: Optional[List] = None,
    ):
        # Instance attributes
        self.target = target
        self.number_features = number_features
        if target_features <= 0:
            raise ValueError("Target features must be a positive, non-zero value.")
        self.target_features = target_features if target_features is not None else number_features # TODO FIX NONE PROBLEM
        self.feature_division = feature_division
        self.filter_metrics = filter_metric
        self.exclude = exclude
        self.number_models = number_models
        self.sort = sort
        self.setup_kwargs = setup_kwargs if setup_kwargs else self.setup_kwargs
        self.numerics = numeric_list if numeric_list is not None else self.numerics
        self.top_models = None
        self.x_train = None
        # Initialize Features class
        features = Features(output=features_path)
        examples = features.create(
            odds_features=odds_features,
            odds_rankings=odds_ranking,
            summary=summary,
        )
        cutoff = self.target.split("_")[-1]
        examples = examples[~examples[f"prob_under_goals_{cutoff}"].isna()]
        # Initialize PipelineDataset class
        pipe_ds = PipelineDatasets(
            features=features,
            target=self.target,
            examples=examples,
            test_size=test_size,
        )
        # Call super
        super(FeatureSelection, self).__init__(
            feat_inst=features,
            features=examples,
            pipeline=pipe_ds,
        )
        # Check if too many features and compute feature list
        self._check_too_many_features()
        self.feature_list = self.pipeline.train_data.columns.tolist()
        self.feature_list.remove(self.target)
        # Compute target features
        self.target_features = self.calculate_number_features(
            number_features=self.target_features, df=self.feature_list
        )

    def _check_too_many_features(self):
        """Check that the number of selected features is smaller than the whole dataset"""
        maximum = max(self.number_features, self.target_features)
        if maximum > self.pipeline.train_data.shape[1]:
            raise ValueError(
                "The number of features selected is greater than those included in "
                "the dataset. Please, introduce a lower value."
            )

    def _compute_numeric_features(self, df: pd.DataFrame):
        """Return those columns from the given dataset whose data type is numeric."""
        return df.select_dtypes(include=self.numerics).columns.tolist()

    def train_model(self):
        """Preprocess the data and select self.number_models top models."""
        # Train dataset
        selected_cols = self.feature_list + [self.target]
        train_data = self.pipeline.train_data[selected_cols]
        # Numeric features
        num_features = self._compute_numeric_features(df=train_data.drop(columns=[self.target]))
        self.setup_kwargs["numeric_features"] = num_features
        # Ignore features
        date_features = [c for c in self.pipeline.train_data.columns if 'date' in c]
        self.setup_kwargs['ignore_features'] = date_features

        run_pycaret_setup(train_data=train_data, target=self.target, **self.setup_kwargs)

        self.x_train = get_config("X_train")
        self.top_models = compare_models(
            n_select=self.number_models,
            sort=self.sort,
            exclude=self.exclude,
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

    def create_dict_tuned_models(self, opt_list: Optional[List] = None):
        """Create a dictionary whose keys are values are pycaret tuned models."""
        self.tune_dict_models = {}
        # Optimize parameters
        optimize_default = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        opt_list = opt_list if opt_list else optimize_default
        # Tuned dict
        for (model_str, py_model), optimize in product(self.dict_models.items(), opt_list):
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
        metric_cond_default = {
            "Accuracy": 0.5,
            "AUC": 0.5,
            "Recall": 0.6,
            "Precision": 0.6,
            "F1": 0.6,
            "Kappa": 0.1,
            "MCC": 0.1,
        }
        remove_dict = dict()
        metrics_cond = self.filter_metrics if self.filter_metrics else metric_cond_default
        models = dataframe.index.tolist()
        for model, (metric, cond) in product(models, metrics_cond.items()):
            if dataframe.loc[model, metric] < cond:
                remove_dict[model] = metric
        remove_models = list(set(remove_dict.keys()))
        dataframe.drop(labels=remove_models, axis="index", inplace=True)
        return dataframe

    @staticmethod
    def calculate_number_features(
        number_features: Union[int, float], df: Union[pd.DataFrame, List]
    ) -> int:
        n_features = (
            number_features if isinstance(number_features, int) else int(number_features * len(df))
        )
        return n_features

    def filter_best_features(self, key_model: str, models_dict: Dict):
        """Compute the most relevant features used by the given model."""
        py_model = models_dict[key_model]
        cond = any([key_model.startswith(name) for name in ["lr", "lda", "ridge", "svm"]])
        score_metric = abs(py_model.coef_[0]) if cond else py_model.feature_importances_
        metrics_dict = {
            "Model_id": key_model,
            "Model": key_model.split("_")[0],
            "Feature": self.x_train.columns,
            "Score": score_metric,
        }
        df = pd.DataFrame(metrics_dict).sort_values(by="Score", ascending=False)
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

        df = self.model_tuned_df.groupby("Model").apply(drop)
        return df.drop(columns="Model").reset_index(level="Model")

    def tune_df(self):
        """Update self.features_df with the most relevant features used by tuned models."""
        # Tune dataframe
        self.model_tuned_df = pd.DataFrame(
            data=[],
            index=self.tune_dict_models.keys(),
            columns=["Model"] + self.metrics_list,
        )
        # Model entry
        for prim_model in self.dict_models.keys():
            ix = [ind.startswith(prim_model) for ind in self.model_tuned_df.index]
            self.model_tuned_df.loc[ix, "Model"] = prim_model
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
        self.features_df = pd.DataFrame(data=[], columns=["Model_id", "Model", "Feature", "Score"])
        self.train_model()
        # Run standard models
        self.create_dict_models()
        self.compute_metrics_df()
        # Run tuned models
        self.create_dict_tuned_models()
        if not bool(self.tune_dict_models):
            raise ValueError("The tune dictionary is empty!")
        self.tune_df()
        self.features_df.index.name = "Index_rem"
        return self.features_df.reset_index().drop(columns="Index_rem")

    def remove_zeros(self):
        """Remove non-relevant features (those with a zero punctuation)."""
        ix = self.features_df["Score"] <= 0
        self.features_df.drop(index=self.features_df.loc[ix].index, inplace=True)
        self.features_df.reset_index(drop=True, inplace=True)

    @staticmethod
    def normalize(dataframe):
        """Normalize the pycaret score of each feature."""

        def norm(x):
            x["Normal"] = x["Score"] / x["Score"].max()
            return x

        return (
            dataframe.groupby("Model_id")
            .apply(norm)
            .sort_values(["Model_id", "Normal"], ascending=[True, False])
        )

    @staticmethod
    def feature_punctuation(dataframe):
        """Assign the score to the selected features."""
        group = dataframe.groupby("Feature")
        sorted_data = group.agg(
            Counts=pd.NamedAgg(column="Normal", aggfunc="count"),
            Normal_Sum=pd.NamedAgg(column="Normal", aggfunc="sum"),
        ).sort_values(by=["Counts", "Normal_Sum"], ascending=False)
        sorted_data["Punctuation"] = sorted_data["Normal_Sum"] / sorted_data["Counts"]
        return sorted_data.sort_values("Punctuation", ascending=False)

    def create_feature_list(self):
        """Run all necessary methods to extract the list of relevant features."""
        # Call creation features dataframe
        self.features_df = self.run_feature_extraction()
        # Remove zeros and normalize
        self.remove_zeros()
        self.features_df = self.normalize(dataframe=self.features_df)
        # Get punctuation
        self.features_df = self.feature_punctuation(dataframe=self.features_df)
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
                (self.number_features / self.feature_division)
                if isinstance(self.number_features, int)
                else self.number_features
            )
            if len(self.feature_list) <= 1:
                break
        return self.feature_list
