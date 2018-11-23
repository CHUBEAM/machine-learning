import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime

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

now = datetime.utcnow().strftime('%Y%m%d%H%M%s')
root_logdir = 'tf_logs'
logdir = '{}/run-{}'.format(root_logdir, now)

mse_summary = tf.summary.scalar('MSE', mse)  # MSE 값을 평가하고 서머리(summary)라고 부르는
# 텐서보드가 인식하는 이진 로그 문자열에 쓰기 위한 노드를 그래프에 추가
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())  # 첫 번재 매개 변수: FileWriter 객체를 만들어
# 로그 디렉터리에 있는 로그 파일에 서머리를 기록
# 두 번째 매개변수(옵션): 시각화하고자 하는 계산 그래프. FileWriter가 생성될 때 로그 디렉터리가 존재하지 않으면
# 새로 만들고(필요하면 부모 디렉터리도), 이벤트 파일(event file)이라 불리는 이진 로그 파일에 그래프의 정의를 기록

