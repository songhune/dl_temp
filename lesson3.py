import tensorflow as tf
import matplotlib.pyplot as plt

X=[1,2,3]#테스트 데이터
Y=[1,2,3]#실제 데이터

W = tf.placeholder(tf.float32)
#플레이스홀더는 약간 변수선언처럼 생각하면 편할듯- 철수네 강아지

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis-Y))
#이게 코스트 펑션

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#세션을 열면 항상 글로벌 변수에 대한 초기화를 시켜주자

W_val = []
cost_val = []
#값들을 저장할 리스트를 만들어 놓고

for i in range(-30,50):
    feed_W = i *0.1
    #실제로 W에 먹일 값들은 -3~5까지 0.1씩 증가함
    curr_cost, curr_W = sess.run([cost,W], feed_dict={W: feed_W})
    #실제로 들어가는 코스트 값과 w값을 보는것이다.

    W_val.append(curr_W)
    cost_val.append((curr_cost))


#gd_optimizer
lr = 0.1
gradient = tf.reduce_mean((W*X-Y)*X)
descent = W - lr * gradient
update = W.assign(descent)

plt.plot(W_val, cost_val)
plt.show()

