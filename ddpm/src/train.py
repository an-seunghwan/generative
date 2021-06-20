'''
Denoising Diffusion Probabilistic Models
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
import matplotlib.pyplot as plt
import os
# os.chdir(r'D:/generative/ddpm')
os.chdir('/Users/anseunghwan/Documents/GitHub/generative/ddpm')

# from modules import ncsn_models
#%%
PARAMS = {
    "batch_size": 128,
    "epochs": 10000, # 200000
    "learning_rate": 0.0001, 
    "data": "cifar10", # or "mnist"
}
#%%
if PARAMS['data'] == "cifar10":
    classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    classdict = {i:x for i,x in enumerate(classnames)}

    (x_train, _), (_, _) = K.datasets.cifar10.load_data()
    # '''0 ~ 1 scaling'''
    # x_train = x_train.astype('float32') / 255.
    '''-1 ~ +1 scaling'''
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    PARAMS["data_dim"] = x_train.shape[1]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
    
    PARAMS["channel"] = 3 # RGB
    PARAMS["mode"] = "RGB"
    
elif PARAMS['data'] == "mnist":
    (x_train, _), (_, _) = K.datasets.mnist.load_data()
    # '''0 ~ 1 scaling'''
    # x_train = x_train[..., tf.newaxis].astype('float32') / 255.
    paddings = [[0, 0],
                [4, 0],
                [4, 0],
                [0, 0]]
    x_train = tf.pad(x_train[..., tf.newaxis], paddings, "CONSTANT") # same with CIFAR-10 dataset image size
    '''-1 ~ +1 scaling'''
    x_train = (tf.cast(x_train, 'float32') - 127.5) / 127.5
    PARAMS["data_dim"] = x_train.shape[1]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
    
    PARAMS["channel"] = 1 # grayscale
    PARAMS["mode"] = "L"
    
else:
    print('Invalid data type!')
    assert 0 == 1
#%%
base_model = K.applications.ResNet50(input_shape=[32, 32, 3], include_top=False)

# base_model.summary()

layer_names = [
    'conv1_relu',   # 16x16x64
    'conv2_block3_out',   # 8x8x256
    'conv3_block4_out',   # 4x4x512
    'conv4_block6_out',   # 2x2x1024
    'conv5_block3_out',   # 1x1x2048
]
layers = [base_model.get_layer(name).output for name in layer_names]

down_stack = K.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = True
#%%
from tensorflow_examples.models.pix2pix import pix2pix

up_stack = [
    pix2pix.upsample(2048, 3),    
    pix2pix.upsample(1024, 3),  
    pix2pix.upsample(512, 3),  
    pix2pix.upsample(256, 3),  
    pix2pix.upsample(64, 3),   
    pix2pix.upsample(64, 3),   
]
#%%
def unet_model(output_channels, activation=tf.nn.elu):
    inputs = K.layers.Input(shape=[32, 32, 3])
    x = inputs

    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack[:-1], skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    x = up_stack[-1](x)
    x = concat([x, inputs])

    last1 = K.layers.Conv2D(32, 3, activation=activation, padding='same')  
    last2 = K.layers.Conv2D(output_channels, 3, padding='same')  

    x = last2(last1(x))

    return K.Model(inputs=inputs, outputs=x)
#%%
model = unet_model(PARAMS['channel'])
model.summary()
#%%