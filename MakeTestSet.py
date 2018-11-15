import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))  # permutation: 순열
    # data 길이만큼의 range를 섞는다
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]  # iloc: integer index를 받아 행렬을 리턴


def test_set_check(identifier, test_ratio):  # identifier: 식별자의 열 이름
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32  # crc는 데이터의 고유한 DNA
# 비트 연산자를 사용하여 return으로 참과 거짓 중 리턴.


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
