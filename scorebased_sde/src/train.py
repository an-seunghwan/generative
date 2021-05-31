'''
Generative Modeling by Estimating Gradients of the Data Distribution
with CIFAR-10 dataset
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.ops.gen_array_ops import Pack
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('/Users/anseunghwan/Documents/GitHub/generative')
#%%
PARAMS = {
    "batch_size": 128,
    "epochs": 200, 
    "learning_rate": 0.0005,
    "num_L":10,
    "sigma_low":0.01,
    "sigma_high":1.0
}
#%%
classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classdict = {i:x for i,x in enumerate(classnames)}
#%%
#%%
'''data'''
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
PARAMS["data_dim"] = x_train.shape[1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
#%%
step = 0
# geometric sequence of sigma
sigma_levels = tf.math.exp(tf.linspace(tf.math.log(PARAMS['sigma_high']),
                                        tf.math.log(PARAMS['sigma_low']),
                                        PARAMS['num_L']))
#%%
'''training'''
for epoch in PARAMS['epochs']:
    for x_batch in tqdm(train_dataset, desc='iteration {}'.format(step)):
        
#%%
# step = 0
# progress_bar = tqdm(train_dataset, total=PARAMS['epochs'])
# progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))

# loss_history = []
# avg_loss = 0
# for x_batch in progress_bar:
#     step += 1
#     idx_sigmas = tf.random.uniform([x_batch.shape[0]], 
#                                     minval=0,
#                                     maxval=PARAMS['num_L'],
#                                     dtype=tf.dtypes.int32)
#     sigmas = tf.gather(sigma_levels, idx_sigmas)
#     sigmas = tf.reshape(sigmas, shape=(x_batch.shape[0], 1, 1, 1))
#     x_batch_perturbed = x_batch + tf.random.normal(shape=x_batch.shape) * sigmas
#%%
