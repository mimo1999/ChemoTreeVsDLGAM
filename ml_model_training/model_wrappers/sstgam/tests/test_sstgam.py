import numpy as np
import pandas as pd
import pytest

from sstgam import SSTGAM


@pytest.fixture
def longitudinal_data():
    generator = np.random.RandomState(10)
    n_samples = 70
    first = generator.normal(size=n_samples)
    second = first + generator.normal(scale=0.25, size=n_samples)
    static = generator.normal(size=n_samples)
    second[generator.rand(n_samples) < 0.15] = np.nan
    target = (first + static + generator.normal(scale=0.4, size=n_samples) > 0).astype(int)
    return pd.DataFrame({"signal_0": first, "signal_4": second, "context": static}), target


@pytest.mark.parametrize("method", ["spline", "tree"])
def test_fit_predict_and_dataframe_column_alignment(longitudinal_data, method):
    X, y = longitudinal_data
    model = SSTGAM(
        method=method, max_rounds=12, early_stopping_rounds=4,
        random_state=1,
    ).fit(X, y)

    probability = model.predict_proba(X)[:, 1]
    reordered_probability = model.predict_proba(X[["context", "signal_4", "signal_0"]])[:, 1]

    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability, reordered_probability)
    assert model.get_temporal_features() == ["signal"]


def test_interpretation_helpers(longitudinal_data):
    X, y = longitudinal_data
    spline_model = SSTGAM(
        method="spline", max_rounds=12, early_stopping_rounds=4, random_state=2
    ).fit(X, y)
    assert spline_model.value_effect("signal", [0.0, 1.0]).shape == (2,)
    assert spline_model.temporal_surface("signal", [0.0, 1.0], [0, 4]).shape == (2, 2)

    tree_model = SSTGAM(
        method="tree", max_rounds=12, early_stopping_rounds=4, random_state=2
    ).fit(X, y)
    _, table = tree_model.value_time_table("signal")
    assert table.shape[1] == 2


def test_prediction_rejects_mismatched_dataframe_columns(longitudinal_data):
    X, y = longitudinal_data
    model = SSTGAM(max_rounds=8, early_stopping_rounds=3).fit(X, y)
    with pytest.raises(ValueError, match="Prediction columns differ"):
        model.predict_proba(X.drop(columns="context"))
