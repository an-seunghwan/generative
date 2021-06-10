'''
Langevin dynamics
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
import matplotlib.pyplot as plt
import os
# os.chdir(r'D:/generative/ncsn')
os.chdir('/Users/anseunghwan/Documents/GitHub/generative/ncsn')
#%%
PARAMS = {
    # "batch_size": 128,
    # "epochs": 10000, 
    # "learning_rate": 0.001,
    # "num_L": 100,
    # "sigma_low": 1.0,
    # "sigma_high": 20.0,
    "T": 1000,
    # "epsilon": 0.1,
    "num_epsilon": 19,
    "epsilon_low": 0.01, 
    "epsilon_high": 10.0,
}

key = 'langevin' 
#%%
'''true data distribution (Gaussian mixture)'''
import tensorflow_probability as tfp
tfd = tfp.distributions

def gmm(probs, loc, scale):
    gmm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=probs),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc,
            scale_identity_multiplier=scale))
    return gmm

def sample(gmm, sample_shape):
    s = tfd.Sample(gmm, sample_shape=sample_shape)
    return s.sample()

mixture_prob = [0.2, 0.8]
mean_parameters = [[-5, -5], [5, 5]]
variance_parameters = [1, 1]
gmm = gmm(mixture_prob, mean_parameters, variance_parameters)
#%%
# geometric sequence of epsilon
epsilon_levels = tf.math.exp(tf.linspace(tf.math.log(PARAMS['epsilon_high']),
                                        tf.math.log(PARAMS['epsilon_low']),
                                        PARAMS['num_epsilon']))
#%%
@tf.function
def analytic_log_gmm_prob_grad(x):
    '''
    ground truth gradient data score of gmm
    d/dx log gmm 
    = d/dx log (\pi_1 normal1 + \pi_2 normal2)
    = d/dx log exp (log \pi_1 + log normal1 + log \pi_2 + log normal2), where exp is element-wise
    '''
    x_tensor = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        normal1 = tfd.MultivariateNormalDiag(loc=mean_parameters[0],
                                            scale_diag=[variance_parameters])
        normal2 = tfd.MultivariateNormalDiag(loc=mean_parameters[1],
                                            scale_diag=[variance_parameters])
        probs = list()
        probs.append(tf.math.log(tf.convert_to_tensor(0.2)) + normal1.log_prob(x_tensor)) # log \pi_1 + log normal1
        probs.append(tf.math.log(tf.convert_to_tensor(0.8)) + normal2.log_prob(x_tensor)) # log \pi_2 + log normal2
        log_prob = tf.reduce_logsumexp(tf.stack(probs, axis=0), axis=0)
    true_gradients = tape.gradient(log_prob, x_tensor)
    return true_gradients

@tf.function
def langevin_dynamics(grad_function, x, epsilon=None, T=1000):
    for _ in tqdm(range(T), desc='Langevin dynamics'):
        score = grad_function(x)
        noise = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1)
        x = x + (epsilon / 2) * score + tf.sqrt(epsilon) * noise
    return x

def meshgrid(x):
    y = x
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)
#%%
'''plot density and generated samples'''
x_init = tf.random.uniform(shape=(1280, 2), minval=-8, maxval=8)

fig, axes = plt.subplots(5, int((PARAMS['num_epsilon'] + 1) / 5), figsize=(15, 15))
x = np.linspace(-8, 8, 500, dtype=np.float32)
axes.flatten()[0].imshow(gmm.prob(meshgrid(x)), cmap='inferno', extent=[-8, 8, -8, 8], origin='lower')
axes.flatten()[0].set_xlabel(r'$x$')
axes.flatten()[0].set_ylabel(r'$y$')
axes.flatten()[0].set_title('ground truth density')

for i in range(PARAMS['num_epsilon']):
    samples = langevin_dynamics(analytic_log_gmm_prob_grad, x_init, epsilon_levels[i], T=PARAMS['T'])
    axes.flatten()[i+1].scatter(samples.numpy()[:, 0], samples.numpy()[:, 1], s=7, alpha=0.3, color='black')
    axes.flatten()[i+1].set_title('$\epsilon$ = {}'.format(epsilon_levels.numpy()[i]))
# plt.savefig('./ncsn/assets/samples_{}_{}_{}_{}_{}_{}_{}.png'.format(key, 
#                                                             PARAMS['learning_rate'], 
#                                                             PARAMS['num_L'],
#                                                             PARAMS['sigma_high'],
#                                                             PARAMS['sigma_low'],
#                                                             PARAMS['T'],
#                                                             PARAMS['epsilon'],)
#             , bbox_inches="tight")
plt.show()
plt.close()
#%%