import pathlib
import pytest
import pandas as pd
from assg.utils import PROJECT_ROOT
from assg.tasks import task2_impute_missing_data

@pytest.fixture
def impute():
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'assg-02-weather.csv')
    X, ndim, shape, columns, na_sum = \
        task2_impute_missing_data(df)
    return X, ndim, shape, columns, na_sum

def test_y(impute):
    X, ndim, shape, columns, na_sum = impute
    assert isinstance(X, pd.DataFrame)

def test_ndim(impute):
    X, ndim, shape, columns, na_sum = impute
    assert ndim == 2

def test_shape(impute):
    X, ndim, shape, columns, na_sum = impute
    assert shape == (366, 2)

def test_columns(impute):
    X, ndim, shape, columns, na_sum = impute
    assert list(columns) == ['Sunshine', 'Pressure3pm']

def test_na_sum(impute):
    X, ndim, shape, columns, na_sum = impute
    series_dict = {'Sunshine': 0, 'Pressure3pm': 0}
    expected_series = pd.Series(data=series_dict, index=['Sunshine', 'Pressure3pm'])
    assert isinstance(na_sum, pd.core.series.Series)
    assert na_sum.equals(expected_series)
