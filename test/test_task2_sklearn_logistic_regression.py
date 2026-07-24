import pathlib
import pytest
import sklearn
import numpy as np
import pandas as pd
from assg.utils import PROJECT_ROOT
from assg.tasks import task2_impute_missing_data
from assg.tasks import task2_label_encoding
from assg.tasks import task2_sklearn_logistic_regression

@pytest.fixture
def fit():
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'assg-02-weather.csv')
    X, _, _, _, _ = task2_impute_missing_data(df)
    y, _, _, _, _ = task2_label_encoding(df.RainTomorrow)
    model, intercept, slopes, accuracy = \
        task2_sklearn_logistic_regression(X, y)
    return model, intercept, slopes, accuracy

def test_model(fit):
    model, intercept, slopes, accuracy = fit
    assert isinstance(model, sklearn.linear_model._logistic.LogisticRegression)

    # test that parameters are set as asked for for the logistic regression
    assert model.get_params()['C'] == 500.0
    assert model.get_params()['solver'] == 'lbfgs'
    assert model.get_params()['fit_intercept']

def test_intercept(fit):
    model, intercept, slopes, accuracy = fit
    assert intercept == pytest.approx(186.590648)

def test_slopes(fit):
    model, intercept, slopes, accuracy = fit
    assert isinstance(slopes, np.ndarray)
    assert slopes.shape == (2,)
    assert slopes[0] == pytest.approx(-0.320885, rel=1e-4)
    assert slopes[1] == pytest.approx(-0.183120, rel=1e-4)

def test_accuracy(fit):
    model, intercept, slopes, accuracy = fit
    assert accuracy == pytest.approx(0.863388, rel=1e-4)
