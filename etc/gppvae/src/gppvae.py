'''
Gaussian Process Prior Variational Autoencoders
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
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
from PIL import Image
import random
import matplotlib.pyplot as plt
import os
# os.chdir(r'D:/generative/ncsn')
os.chdir('/Users/anseunghwan/Documents/GitHub/generative/gppvae')
#%%
PARAMS = {
    "batch_size": 128,
    "epochs": 100, 
    "learning_rate": 0.001, 
    "data": "rotating_mnist",
    "Q": 16,
    "M": 8,
    "L": 16, # latent dimension
}
#%%
if PARAMS['data'] == "rotating_mnist":
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    digit = 3
    x_train = x_train[np.where(y_train == digit)[0], ...]
    # y_train = y_train[np.where(y_train == digit)[0]]
    x_test = x_test[np.where(y_test == digit)[0], ...]
    # y_train = y_train[np.where(y_train == digit)[0]]
    
    train_test_val = np.vstack([x_train, x_test])
    random.seed(520)
    data_idx = random.sample(range(len(train_test_val)), 400)
    train_test_val = train_test_val[data_idx]
    
    '''rotation'''
    from scipy.ndimage.interpolation import rotate
    rotated = []
    angle_feature = []
    for r in range(PARAMS["Q"]):
        rotated.append(rotate(train_test_val, angle=r*np.pi/PARAMS["Q"], reshape=False))
        angle_feature.extend([r] * len(train_test_val))
    rotated = np.vstack(rotated)
    angle_feature = np.array(angle_feature)
        
    '''partition'''
    train_test_idx = random.sample(range(len(rotated)), int(0.9*len(rotated)))
    train_test = rotated[train_test_idx]
    train_test_angle_feature = angle_feature[train_test_idx]
    val = rotated[[i for i in range(len(rotated)) if i not in train_test_idx]]
    val_angle_feature = angle_feature[[i for i in range(len(rotated)) if i not in train_test_idx]]
    
    remain_train_test_idx = random.sample(range(len(train_test)), int(0.75*len(train_test)))
    remain_val_idx = random.sample(range(len(val)), int(0.75*len(val)))
    train_test = train_test[remain_train_test_idx]
    train_test_angle_feature = train_test_angle_feature[remain_train_test_idx]
    val = val[remain_val_idx]
    val_angle_feature = val_angle_feature[remain_val_idx]
    
    test = train_test[np.where(np.array(train_test_angle_feature) == 8)[0], ...]
    test_angle_feature = train_test_angle_feature[np.where(np.array(train_test_angle_feature) == 8)[0], ...]
    train = train_test[np.where(np.array(train_test_angle_feature) != 8)[0], ...]
    train_angle_feature = train_test_angle_feature[np.where(np.array(train_test_angle_feature) != 8)[0], ...]
    print(train.shape)
    
    PARAMS["data_dim"] = train.shape[1]
    # train_dataset = tf.data.Dataset.from_tensor_slices((train, train_angle_feature)).shuffle(len(train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
    
    PARAMS["channel"] = 1 # grayscale
#%%

#%%