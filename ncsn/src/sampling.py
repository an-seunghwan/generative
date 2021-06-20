'''
Generative Modeling by Estimating Gradients of the Data Distribution
Technique 3. Noise conditioning is used
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
# os.chdir(r'D:/generative/ncsn')
os.chdir('/Users/anseunghwan/Documents/GitHub/generative/ncsn')

from modules import ncsn_models
#%%
PARAMS = {
    "batch_size": 128,
    "epochs": 200000, 
    "learning_rate": 0.00001, 
    "data": "mnist", 
    "num_L": 200,
    "sigma_high": 20.0,
    "sigma_low": 0.1,
    "T": 100,
    "epsilon": 0.00001
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
# geometric sequence of sigma
sigma_levels = tf.math.exp(tf.linspace(tf.math.log(PARAMS['sigma_high']),
                                        tf.math.log(PARAMS['sigma_low']),
                                        PARAMS['num_L']))
#%%
model = ncsn_models.build_refinenet(PARAMS)
model.load_weights('/Users/anseunghwan/Documents/GitHub/generative_save/weights/weights_{}_{}_{}_{}_{}/weights'.format(PARAMS['data'], 
                                                                                                                        PARAMS['learning_rate'], 
                                                                                                                        PARAMS['num_L'],
                                                                                                                        PARAMS['sigma_high'],
                                                                                                                        PARAMS['sigma_low']))
#%%
@tf.function
def langevin_dynamics(scorenet, x, sigma_i=None, alpha=0.1, T=1000):
    for _ in range(T):
        # score = scorenet(x, sigma_i)
        score = scorenet(x) / sigma_i
        noise = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1)
        x = x + (alpha / 2) * score + tf.sqrt(alpha) * noise
    return x

def annealed_langevin_dynamics(scorenet, x, sigma_levels, T=100, eps=0.1, intermediate=False):
    if intermediate:
        x_list = []
        for i in tqdm(range(len(sigma_levels))):
            sigma_i = sigma_levels[i]
            alpha_i = eps * (sigma_i ** 2) / (sigma_levels[-1] ** 2) # step size
            x = langevin_dynamics(scorenet, x, sigma_i=sigma_i, alpha=alpha_i, T=T) # Langevin dynamics
            x_list.append(x)
        return x_list
    else:
        for i in tqdm(range(len(sigma_levels))):
            sigma_i = sigma_levels[i]
            alpha_i = eps * (sigma_i ** 2) / (sigma_levels[-1] ** 2) # step size
            x = langevin_dynamics(scorenet, x, sigma_i=sigma_i, alpha=alpha_i, T=T) # Langevin dynamics
        return x
#%%
@tf.function
def preprocess_image_to_save(x):
    x = tf.clip_by_value(x, 0, 1)
    x = x * 255
    x = x + 0.5
    x = tf.clip_by_value(x, 0, 255)
    return x

def save_as_grid(images, filename, spacing=2):
    """
    Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    :param images:
    :return:
    """
    # Define grid dimensions
    cols, rows, height, width, channels = images.shape

    # Process image (clip values 0 ~ 1)
    images = preprocess_image_to_save(images)

    # Init image
    grid_cols = rows * height + (rows + 1) * spacing
    grid_rows = cols * width + (cols + 1) * spacing
    im = Image.new(PARAMS['mode'], (grid_rows, grid_cols))
    for row in range(rows):
        for col in range(cols):
            x = col * height + (1 + col) * spacing
            y = row * width + (1 + row) * spacing
            im.paste(tf.keras.preprocessing.image.array_to_img(images[col, row]), (x, y))
    plt.axis('off')
    plt.imshow(im)
    plt.savefig('./assets/{}.png'.format(filename), bbox_inches="tight")
    plt.close()
#%%
'''evaluation
1. generating (intermediate)
2. the nearest neighborhood (l2 dist)
'''

'''1. generating (intermediate)'''
B = 10
intermediate_images = []
tf.random.set_seed(520)
# x_init = tf.random.uniform(shape=(B, PARAMS["data_dim"], PARAMS["data_dim"], PARAMS['channel']))
x_init = tf.random.normal(shape=(B, PARAMS["data_dim"], PARAMS["data_dim"], PARAMS['channel']))
intermediate_images.append(x_init)
intermediate_images += annealed_langevin_dynamics(model, x_init, sigma_levels, T=PARAMS['T'], eps=PARAMS['epsilon'], intermediate=False)
images = tf.stack(intermediate_images)
save_as_grid(images, '{}_samples_{}_{}_{}_{}_{}_{}'.format(PARAMS['data'], 
                                                            PARAMS['learning_rate'], 
                                                            PARAMS['num_L'],
                                                            PARAMS['sigma_high'],
                                                            PARAMS['sigma_low'],
                                                            PARAMS['T'],
                                                            PARAMS['epsilon'],))
#%%
'''2. the nearest neighborhood (l2 dist)'''
#%%