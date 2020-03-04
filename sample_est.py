import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x_data = np.linspace(0.0,10.0,1000000)
noise  = np.random.randn(len(x_data))


y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data,columns=['X'])
y_df = pd.DataFrame(data=y_true,columns=['Y'])
my_data = pd.concat([x_df,y_df],axis=1)

my_sample = my_data.sample(n=250)
#my_sample.plot(kind='scatter',x='X',y='Y')
batch_size = 10

m = tf.Variable(1.20)
b = tf.Variable(-1.6)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_model = m * xph + b


## Tensorflow estimator

feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3,random_state=101)

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=10,num_epochs=None,shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=10,num_epochs=1000,shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=10,num_epochs=1000,shuffle=False)
estimator.train(input_fn=input_func,steps=1000)
