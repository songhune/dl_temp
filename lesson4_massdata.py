import numpy as np
import tensorflow as tf
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

'''승현아, 단계를 생각해봐 - 맨 처음에 뭘 해야되겠니?
X랑 Y를 placeholder로 설정해야 돼요
'''

X = tf.placeholder(dtype=tf.float32, shape =[None,3])
Y = tf.placeholder(dtype=tf.float32, shape =[None,1])

'''그 담엔 뭘 해야 하니
설계를 한다
W/HYPOTHESIS /COST /optimizer/TRAIN '''
#placeholder이랑 variable 차이?
W = tf.Variable(tf.random_normal([3,1]),name='weight')#3개가 들어가서 하나가 나오니까
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.matmul(X,W)+b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

#train의 의미= lost를 최소화 하는 것 optimizer
train= optimizer.minimize(cost)

'''마지막 단계 세션을 열어서 돌린다'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#여기까진 초기화

for i in range(2001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train],feed_dict={X:x_data, Y: y_data})
    if(i%10 == 0):
        print('횟수: ',i, "\n낮을수록 좋은 cost: ", cost_val
              ,"\n 정확할수록 좋은 prediction: ", hy_val)