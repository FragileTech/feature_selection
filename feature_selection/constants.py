"""Module including parameter lists and dictionaries for feature_selection library."""

MODEL_CLASS_TO_NAME = {
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

METRICS_LIST = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]

# Private class attributes
FILTER_METRIC = {
    "Accuracy": 0.5,
    "AUC": 0.5,
    "Recall": 0.6,
    "Precision": 0.6,
    "F1": 0.6,
    "Kappa": 0.1,
    "MCC": 0.1,
}

SETUP_KWARGS = dict(
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
    html=False,
)

NUMERICS = ["int16", "int32", "int64", "float16", "float32", "float64", "int", "float"]

OPT_LIST = ["Accuracy", "Precision", "Recall", "F1", "AUC"]