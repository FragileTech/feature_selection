from pycaret.classification import add_metric, setup
from sklearn.metrics import brier_score_loss


def run_pycaret_setup(train_data, target, **kwargs):
    setup_kwargs = dict(
        preprocess=True,
        data=train_data,
        train_size=0.75,
        target=target,
        session_id=160290,
        normalize=True,
        transformation=True,
        ignore_low_variance=False,
        remove_multicollinearity=False,
        multicollinearity_threshold=0.4,
        use_gpu=False,
        profile=False,
        ignore_features=None,
        fold_strategy="timeseries",
        # numeric_features=train_data.drop(columns=ignore_features + [target]).columns.tolist(),
        remove_perfect_collinearity=True,
        create_clusters=False,
        fold=3,
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
        fix_imbalance=True,
        log_experiment=False,
        verbose=False,
        silent=True,
        experiment_name="lagstest",
        html=False,
    )
    setup_kwargs.update(kwargs)
    setup(**setup_kwargs)
    add_metric(
        "brier",
        "brier",
        brier_score_loss,
        target="pred_proba",
        multiclass=False,
        greater_is_better=False,
    )
