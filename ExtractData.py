import os
import tarfile
from six.moves import urllib
import pandas as pd

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'  # 전역 변수는 대문자를 사용
HOUSING_PATH = os.path.join('datasets', 'housing')  # os에 맞는 디렉토리 주소를 리턴 -> 우분투의 경우 ./datasets/housing
# 윈도우즈의 경우 .\datasets\housing
HOUSING_URL = DOWNLOAD_ROOT + 'datatsets/housing/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):  # 함수내 변수와 전역 변수를 대소문자로 구분
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)  # housing_url 파일을 tgz_path에 저장
    housing_tgz = tarfile.open(tgz_path)  # 묶음 파일 열기
    housing_tgz.extractall(path=housing_path)  # 묶음 풀기
    housing_tgz.close()  # 파일은 열었으면 닫아야 한다


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)
