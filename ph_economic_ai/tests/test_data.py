import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.data import generate_dataset


def test_shape():
    df = generate_dataset()
    assert df.shape == (120, 5)


def test_columns():
    df = generate_dataset()
    assert list(df.columns) == ['date', 'oil_price', 'usd_php', 'demand_index', 'gas_price']


def test_no_nulls():
    df = generate_dataset()
    assert df.isnull().sum().sum() == 0


def test_ranges():
    df = generate_dataset()
    assert df['oil_price'].between(75, 105).all()
    assert df['usd_php'].between(54, 62).all()
    assert df['demand_index'].between(55, 90).all()
    assert df['gas_price'].between(62, 82).all()


def test_reproducible():
    df1 = generate_dataset(seed=42)
    df2 = generate_dataset(seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds():
    df1 = generate_dataset(seed=1)
    df2 = generate_dataset(seed=2)
    assert not df1['oil_price'].equals(df2['oil_price'])
