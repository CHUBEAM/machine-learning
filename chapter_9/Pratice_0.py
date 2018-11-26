import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2  # 계산을 수행하지 않고 계산 그래프만 만든다
# 계산 그래프를 평가하려면 텐서플로 세션을 시작하고 변수를 초기화한 다음 f를 평해야 한다

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)  # 매번 sess.run()을 반복하면 번거로우니 with문 사용
print(result)
sess.close()

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()

interactivesess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
x2.graph is graph
x2.graph is tf.get_default_graph()

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 텐서플로는 자동으로 y가 x에 의존한다는 것과 x가 w에 의존한다는 것을 감지
    # 먼저 w를 평가하고 그다음에 x, 그다음에 y를 평가해서 y 값을 반환
    print(z.eval())

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]  # np.one() 함수에 (m, 1) 튜플 입력

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1,1 ), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session as sess:
    theta_value = theta.eval()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

n_epochs = 1000
learning_rate = 0.01

pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

scaled_housing_data_plus_bias = pipeline.fit_transform(housing_data_plus_bias)

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)  # traning_op는 변수 X, init와 같은 메서드

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
            summary_str = mse_summary.eval()
            step = epoch
            file_writer.add_summary(summary_str, step)
        sess.run(training_op)

    best_theta = theta.eval()
    file_writer.close()

A = tf.placeholder(tf.float32, shape=(None, 3))
B  = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8 ,9]]})

print(B_val_1)
print(B_val_2)

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}'.format(root_logdir, now)

mse_summary = tf.summary.scalar('MSE', mse)  # MSE 값을 평가하고 서머리(summary)라고 부르는
# 텐서보드가 인식하는 이진 로그 문자열에 쓰기 위한 노드를 그래프에 추가
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())  # 첫 번재 매개 변수: FileWriter 객체를 만들어
# 로그 디렉터리에 있는 로그 파일에 서머리를 기록
# 두 번째 매개변수(옵션): 시각화하고자 하는 계산 그래프. FileWriter가 생성될 때 로그 디렉터리가 존재하지 않으면
# 새로 만들고(필요하면 부모 디렉터리도), 이벤트 파일(event file)이라 불리는 이진 로그 파일에 그래프의 정의를 기록

with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')

print(error.op.name)
print(mse.op.name)

def relu(X, threshold):
    with tf.name_scope('relu'):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, threshold, name='relu')

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')
relus = [relu(X) for i in range(5)]  # relu(X) 원소를 5번 반복하게 하는 장치
output = tf.add_in(relus, name='output')

threshold = tf.Variable(0.0, name='threshold')

with tf.variable_scope('relu'):  # 변수를 새로 만든다
    threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))  # shape=()이므로
    # 스칼라 변수

with tf.variable_scope('relu', reuse=True):
    threshold = tf.get_variable('threshold')

with tf.variable_scope('relu') as scope:
    scope.reuse_variables()
    threshold = tf.get_variable('threshold')

def relu(X):
    with tf.variable_scope('relu', reuse=True):
        threshold = tf.get_variable('threshold')
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, threshold, name='relu')

def relu(X):
    threshold = tf.get_variable('threshold', shape=(), initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name='weights')
    b = tf.Variable(0.0, name='bias')
    z = tf.add(tf.matmul(X, w), b, name='z')
    return tf.maximum(z, threshold, name='relu')

relus = []
for relu_index in range(5):
    with tf.variable_scope('relu', reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name='output')