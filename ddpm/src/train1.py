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
os.chdir('/home1/prof/jeon/an/generative/ddpm')

from modules import models1
#%%
PARAMS = {
    "batch_size": 128,
    "epochs": 300000, 
    "learning_rate": 0.0001, 
    "data": "cifar10", 
    "embedding_dim": 48, 
    "T": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
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
step = 0
betas = tf.linspace(PARAMS['beta_start'],
                    PARAMS['beta_end'],
                    PARAMS['T'])
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))
#%%
'''q(x_t|x_0)'''
sigmas = np.sqrt(betas)
alphas = 1. - betas
alphas_sqrt = np.sqrt(alphas)
alphas_cumprod_sqrt = np.sqrt(np.cumprod(alphas, axis=0))
alphas_cumprod_one_minus_sqrt = np.sqrt(1 - np.cumprod(alphas, axis=0))
#%%
model = models1.build_unet(PARAMS, PARAMS['embedding_dim'], dropout=0., embedding_dim_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=(16, ), resamp_with_conv=True)
optimizer = K.optimizers.Adam(learning_rate=PARAMS['learning_rate'])

@tf.function
def train_one_step(optimizer, x_batch_perturbed, epsilon, timesteps):
    with tf.GradientTape() as tape:
        pred = model([x_batch_perturbed, timesteps])
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred - epsilon), axis=[1,2,3]))
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
#%%
'''training'''
loss_history = []
for _ in progress_bar:
    x_batch = next(iter(train_dataset))
    step += 1
    
    # sampling timestep
    timesteps = tf.random.uniform([x_batch.shape[0]], 
                                    minval=0,
                                    maxval=PARAMS['T'],
                                    dtype=tf.dtypes.int32)
    x0_weights = tf.gather(alphas_cumprod_sqrt, timesteps)
    epsilon_weights = tf.gather(alphas_cumprod_one_minus_sqrt, timesteps)
    epsilon = tf.random.normal(shape=x_batch.shape)
    x_batch_perturbed = x0_weights[:, tf.newaxis, tf.newaxis, tf.newaxis] * x_batch + epsilon_weights[:, tf.newaxis, tf.newaxis, tf.newaxis] * epsilon # reparmetrization trick
    
    current_loss = train_one_step(optimizer, x_batch_perturbed, epsilon, timesteps)
    loss_history.append(current_loss.numpy())
    
    progress_bar.set_description('setting: {} epochs:{} lr:{} dim:{} T:{} beta:{} to {} | iteration {}/{} | current loss {:.3f}'.format(
        PARAMS['data'], PARAMS['epochs'], PARAMS['learning_rate'], PARAMS['embedding_dim'], PARAMS['T'], PARAMS['beta_start'], PARAMS['beta_end'], 
        step, PARAMS['epochs'], current_loss
    ))

    if step == PARAMS['epochs']: break
#%%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(loss_history)
ax.set_title('loss')
plt.savefig('./assets/loss_{}_{}_{}_{}_{}_{}.png'.format(PARAMS['data'], 
                                                        PARAMS['learning_rate'], 
                                                        PARAMS['embedding_dim'],
                                                        PARAMS['T'],
                                                        PARAMS['beta_start'],
                                                        PARAMS['beta_end']))
# plt.show()
plt.close()
#%%
model.save_weights('./assets/{}/weights_{}_{}_{}_{}_{}_{}/weights'.format(PARAMS['data'],
                                                                        PARAMS['data'], 
                                                                        PARAMS['learning_rate'], 
                                                                        PARAMS['embedding_dim'],
                                                                        PARAMS['T'],
                                                                        PARAMS['beta_start'],
                                                                        PARAMS['beta_end']))
#%%
# model = models2.build_unet(PARAMS, PARAMS['embedding_dim'], dropout=0., embedding_dim_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=(16, ), resamp_with_conv=True)
# model.load_weights('./assets/{}/weights_{}_{}_{}_{}_{}_{}/weights'.format(PARAMS['data'], 
#                                                                         PARAMS['data'],
#                                                                         PARAMS['learning_rate'], 
#                                                                         PARAMS['embedding_dim'],
#                                                                         PARAMS['T'],
#                                                                         PARAMS['beta_start'],
#                                                                         PARAMS['beta_end']))
#%%
'''sampling'''
def reverse_process(model, PARAMS, B, T=None, intermediate=False):
    x = tf.random.normal(shape=[B, PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']], mean=0, stddev=1)
    if intermediate:
        x_list = [x]
        for t in tqdm(reversed(range(T))):
            epsilon = model([x, np.ones((B, )) * t])
            diff = (1 / alphas_sqrt[t]) * (x - (betas[t] / alphas_cumprod_one_minus_sqrt[t]) * epsilon)
            x = diff + sigmas[t] * tf.random.normal(shape=x.get_shape(), mean=0, stddev=1)
            if (t+1) % 50 == 0:
                x_list.append(x)
        x_list.append(x)
        return x_list
    else:
        for t in tqdm(reversed(range(T))):
            epsilon = model([x, np.ones((B, )) * t])
            diff = (1 / alphas_sqrt[t]) * (x - (betas[t] / alphas_cumprod_one_minus_sqrt[t]) * epsilon)
            x = diff + sigmas[t] * tf.random.normal(shape=x.get_shape(), mean=0, stddev=1)
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
    plt.figure(figsize = (20, 20))
    plt.axis('off')
    if PARAMS['channel'] == 1:
        plt.imshow(im, 'gray')
    else:
        plt.imshow(im)
    plt.savefig('./assets/{}.png'.format(filename), bbox_inches="tight")
    plt.close()
#%%
'''1. generating'''
tf.random.set_seed(11)
x = reverse_process(model, PARAMS, B=10, T=PARAMS['T'], intermediate=False)
save_as_grid(np.array(x), '{}_samples_{}_{}_{}_{}_{}'.format(PARAMS['data'], 
                                                            PARAMS['learning_rate'], 
                                                            PARAMS['embedding_dim'],
                                                            PARAMS['T'],
                                                            PARAMS['beta_start'],
                                                            PARAMS['beta_end']))
#%%