import pathlib
import pytest
import numpy as np
import pandas as pd
from assg_utils import PROJECT_ROOT
from assg_tasks import task2_label_encoding

@pytest.fixture
def encode():
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'assg-02-weather.csv')
    y, ndim, shape, num_no, num_yes = \
        task2_label_encoding(df.RainTomorrow)
    return y, ndim, shape, num_no, num_yes

def test_y(encode):
    y, ndim, shape, num_no, num_yes = encode
    assert isinstance(y, np.ndarray)
    #assert isinstance(y.dtype, np.dtypes.Float64DType)

def test_ndim(encode):
    y, ndim, shape, num_no, num_yes = encode
    assert ndim == 1

def test_shape(encode):
    y, ndim, shape, num_no, num_yes = encode
    assert shape == (366,)

def test_num_no(encode):
    y, ndim, shape, num_no, num_yes = encode
    assert num_no == 300

def test_num_yes(encode):
    y, ndim, shape, num_no, num_yes = encode
    assert num_yes == 66