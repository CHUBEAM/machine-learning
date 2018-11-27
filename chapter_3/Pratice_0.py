from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

mnist = fetch_mldata('MNIST original')

X, y = mnist['data'], mnist['target']
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[:60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 5-감지기
## 타깃 벡터 만들기
y_train_5 = (y_train == 5)
y_test_5 = (y_test = = 5)

## SGDCClassifier 모델 만들기
sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)

## 모델을 사용해 숫자 5의 이미지 감지
sgd_clf.predict([some_digit])

# K-겹 교차 검증
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

# 모든 이미지를 '5 아님' 클래스로 분류하는 더미 분류기
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')