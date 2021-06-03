'''
Generative Modeling by Estimating Gradients of the Data Distribution
with CIFAR-10 dataset
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
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
os.chdir('/Users/anseunghwan/Documents/GitHub/generative')

from modules import ncsn_models
#%%
PARAMS = {
    "batch_size": 128,
    "epochs": 200000, 
    "learning_rate": 0.001,
    "channel": 3, # RGB
    "num_L": 10,
    "sigma_low": 0.1,
    "sigma_high": 1.0,
    "T": 100,
    "epsilon": 2*1e-5
}
#%%
classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classdict = {i:x for i,x in enumerate(classnames)}
#%%
'''data'''
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
PARAMS["data_dim"] = x_train.shape[1]
'''0~1 scaling'''
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
#%%
model = ncsn_models.build_refinenet(PARAMS, activation=tf.nn.elu)
optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS['learning_rate'])

step = 0
# geometric sequence of sigma
sigma_levels = tf.math.exp(tf.linspace(tf.math.log(PARAMS['sigma_high']),
                                        tf.math.log(PARAMS['sigma_low']),
                                        PARAMS['num_L']))
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))
#%%
@tf.function
def dsm_loss(score, x_perturbed, x, sigmas):
    '''
    scaled loss of denoising score matching:
    \lambda(\sigma) * Fisher information
    = \sigma^2 * Fisher information
    '''
    target = (x_perturbed - x) / (tf.square(sigmas))
    loss = tf.square(sigmas) * 0.5 * tf.reduce_sum(tf.square(score + target), axis=[1,2,3], keepdims=True)
    loss = tf.reduce_mean(loss)
    return loss

@tf.function
def train_one_step(model, optimizer, x_batch_perturbed, x_batch, idx_sigmas, sigmas):
    with tf.GradientTape() as tape:
        # scores = model([x_batch_perturbed, idx_sigmas])
        scores = model(x_batch_perturbed) / sigmas
        current_loss = dsm_loss(scores, x_batch_perturbed, x_batch, sigmas)
        gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss
#%%
'''training'''
loss_history = []
for _ in progress_bar:
    x_batch = next(iter(train_dataset))
    step += 1
    
    idx_sigmas = tf.random.uniform([x_batch.shape[0]], 
                                    minval=0,
                                    maxval=PARAMS['num_L'],
                                    dtype=tf.dtypes.int32)
    
    sigmas = tf.gather(sigma_levels, idx_sigmas)
    sigmas = tf.reshape(sigmas, shape=(x_batch.shape[0], 1, 1, 1))
    x_batch_perturbed = x_batch + tf.random.normal(shape=x_batch.shape) * sigmas # reparmetrization trick
    
    current_loss = train_one_step(model, optimizer, x_batch_perturbed, x_batch, idx_sigmas, sigmas)
    loss_history.append(current_loss.numpy())

    progress_bar.set_description('iteration {}/{} | current loss {:.3f}'.format(
        step, PARAMS['epochs'], current_loss
    ))

    if step == PARAMS['epochs']: break
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
        for sigma_i in sigma_levels:
            alpha_i = eps * (sigma_i ** 2) / (sigma_levels[-1] ** 2) # step size
            x = langevin_dynamics(scorenet, x, sigma_i=sigma_i, alpha=alpha_i, T=T) # Langevin dynamics
            x_list.append(x)
        return x_list
    else:
        for sigma_i in sigma_levels:
            alpha_i = eps * (sigma_i ** 2) / (sigma_levels[-1] ** 2) # step size
            x = langevin_dynamics(scorenet, x, sigma_i=sigma_i, alpha=alpha_i, T=T) # Langevin dynamics
        return x
#%%
@tf.function
def preprocess_image_to_save(x):
    x = tf.clip_by_value(x, 0, 1)
    # x = x * 255
    # x = x + 0.5
    # x = tf.clip_by_value(x, 0, 255)
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
    im = Image.new("RGB", (grid_rows, grid_cols))
    for row in range(rows):
        for col in range(cols):
            x = col * height + (1 + col) * spacing
            y = row * width + (1 + row) * spacing
            im.paste(tf.keras.preprocessing.image.array_to_img(images[col, row]), (x, y))
    plt.axis('off')
    plt.imshow(im)
    plt.savefig('./ncsn/assets/{}.png'.format(filename), bbox_inches="tight")
#%%
'''evaluation
1. generating (intermediate)
2. the nearest neighborhood (l2 dist)
'''

'''1. generating (intermediate)'''
B = 5
intermediate_images = []
x_init = tf.random.uniform(shape=(B, 32, 32, 3))
intermediate_images.append(x_init)
intermediate_images += annealed_langevin_dynamics(model, x_init, sigma_levels, T=PARAMS['T'], eps=PARAMS['epsilon'], intermediate=True)
images = tf.stack(intermediate_images)
save_as_grid(images, 'cifar10_intermediate')
#%%
'''2. the nearest neighborhood (l2 dist)'''
#%%