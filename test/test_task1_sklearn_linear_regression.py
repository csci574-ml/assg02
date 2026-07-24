import pytest
import sklearn
import numpy as np
import pathlib
from assg.utils import PROJECT_ROOT
from assg.tasks import task1_sklearn_linear_regression

@pytest.fixture
def fit():
    X = np.load(PROJECT_ROOT / 'data' / 'regression_features.npy')
    y = np.load(PROJECT_ROOT / 'data' / 'regression_labels.npy')
    model, intercept, slope, mse, rmse, rsquared = \
        task1_sklearn_linear_regression(X, y)
    return model, intercept, slope, mse, rmse, rsquared

def test_model(fit):
    model, intercept, slope, mse, rmse, rsquared = fit
    assert isinstance(model, sklearn.linear_model._base.LinearRegression)
    assert model.get_params()['fit_intercept']

def test_intercept(fit):
    model, intercept, slope, mse, rmse, rsquared = fit
    assert intercept == pytest.approx(0.37578175021210747)

def test_slope(fit):
    model, intercept, slope, mse, rmse, rsquared = fit
    assert slope == pytest.approx(0.3354845860060065)

def test_mse(fit):
    model, intercept, slope, mse, rmse, rsquared = fit
    assert mse == pytest.approx(3.5473465427798607)

def test_rmse(fit):
    model, intercept, slope, mse, rmse, rsquared = fit
    assert rmse == pytest.approx(1.8834400820784984)

def test_rsquared(fit):
    model, intercept, slope, mse, rmse, rsquared = fit
    assert rsquared == pytest.approx(0.5008050204985712)
