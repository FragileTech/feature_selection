"""Module with different functions for supporting the feature_selection library"""
import pandas as pd


def calculate_number_features(
        number_features: Union[int, float], features: Union[pd.DataFrame, List]
) -> int:
    """Compute the number of features given a dataframe or list."""
    n_features = (
        int(number_features)
        if (number_features >= 1)
        else int(number_features * len(features))
    )
    return n_features


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


def feature_score(dataframe):
    """Assign the score to the selected features."""
    group = dataframe.groupby("feature")
    sorted_data = group.agg(
        counts=pd.NamedAgg(column="normal", aggfunc="count"),
        normal_sum=pd.NamedAgg(column="normal", aggfunc="sum"),
    ).sort_values(by=["counts", "normal_sum"], ascending=False)
    sorted_data["final_score"] = sorted_data["normal_sum"] / sorted_data["counts"]
    return sorted_data.sort_values("final_score", ascending=False)