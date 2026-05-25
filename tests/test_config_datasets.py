import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))

from config import get_config, str2datasets


def test_default_datasets_preserve_baseline_order():
    config = get_config(device='cpu')
    assert config.datasets == ['SumMe','TVSum']


def test_dataset_subset_parsing():
    assert str2datasets('SumMe') == ['SumMe']
    assert str2datasets('TVSum') == ['TVSum']
    assert str2datasets('SumMe,TVSum') == ['SumMe','TVSum']
    config = get_config(device='cpu',datasets='TVSum')
    assert config.datasets == ['TVSum']


def test_invalid_dataset_rejected():
    try:
        str2datasets('BadSet')
    except Exception:
        return
    raise AssertionError('invalid dataset should be rejected')


if __name__=='__main__':
    test_default_datasets_preserve_baseline_order()
    test_dataset_subset_parsing()
    test_invalid_dataset_rejected()
