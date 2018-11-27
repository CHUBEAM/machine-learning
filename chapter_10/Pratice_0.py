import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)  # infer_real_valued_columns_from_input()
# 함수는 tf.contrib.learn 하위의 DNNClassifier, DNNRegressor, LinearClassifier, LinearRegressor 등의 클래스에 정수 또는
# 실수로 된 특성을 매핑하기 위한 FeatureColumn 객체를 만든다. 범주형 데이터의 경우 sparse_column_with_key() 함수 등을
# 사용할 수 있다.
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

y_pred = dnn_clf.predict(X_test)
accuracy_score(y_test, y_pred['classes'])

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

def neuron_layer(X, n_neurons, name, activation=None):  # 매개변수: 입력, 뉴런 수, 층 이름, 활성화 함수
    with tf.name_scope(name):  # 층 이름으로 이름 범위를 만듦. 여기에 이 층에서 필요한 모든 계산 노드가 포함된다
        n_inputs = int(X.get_shape()[1])  # 입력 특성의 수 구하기
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='kernel')  # 가중치 행렬을 담을 W 변수 만듦기(커널). 이 행렬은 각 입력과 각 뉴런
        # 사이의 모든 연결 가중치를 담고 있는 2D 텐서. 크기는 (n_inputs, n_neurons). 이 행렬은 표준편차가
        # 2 / (n_inputs + n_neurons)^(1/2)인 절단 정규(가우시안) 분포(truncated normal distribution)를 사용해 무작위로
        # 초기화. 이 표준편차를 사용하면 알고리즘이 빠르게 수렴(이것은 신경망을 엄청나게 효율적으로 만들어준 변화).
        # 경사 하강법 알고리즘이 중단되지 않도록 대칭성을 피하기 위해 모든 은닉층의 가중치는 무작위로 초기화하는 것이
        # 중요.
        b = tf.Variable(tf.zeros([n_neurons]), name='bias') # 뉴런마다 하나의 편향을 갖도록 변수 b를 만들고 0으로 초기화
        # (여기서는 대칭 문제가 없다)
        Z = tf.matmul(X, W) + b  #층에 있는 모든 뉴런과 배치에 있는 모든 샘플에 대해 입력에 대한 가중치 합에 편향을
        # 더하는 계산을 수행. 브로드캐스팅.
        if activation is not None:  # tf.nn.relu와 같은 activation 매개변수가 지정되어 있지 않으면 activation(Z)(즉
            # max(0, Z))를 반환, 그렇지 않으면 Z 반환
            return activation(Z)
        else:
            return Z

with tf.name_scope('dnn'):
    hidden1 = neuron_layer(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name='outputs')  # 소프트맥스 활성화 함수로 들어가기 직전의 신경망 출력

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)  # in_top_k(predictions, targets, k): predictions과 targets을 입력받아
    # 타깃 레이블의 예측값이 크기순으로 k 번째 안에 들면 True, 그렇지 않으면 False를 반환.
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))  # len(X) 범위 수를 무작위로 섞는다
    n_batches = len(X) // batch_size  # batch_size 개씩 담을 바구니 수
    for batch_idx in np.array_split(rnd_idx, n_batches):  # rnd_idx를 바구니에 담는다
        X_batch, y_batch = X[batch_idx,], y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, '배치 데이터 정확도:', acc_batch, '검증 세트 정확도:', acc_valid)

    save_path = saver.save(sess, './my_model_final.ckpt')