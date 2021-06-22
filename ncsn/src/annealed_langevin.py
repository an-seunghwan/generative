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
    "num_L": 10,
    "sigma_high": 20.0,
    "sigma_low": 1.0,
    "epsilon": 0.1,
}

key = 'annealed_langevin' 
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
# geometric sequence of sigma
sigma_levels = tf.math.exp(tf.linspace(tf.math.log(PARAMS['sigma_high']),
                                        tf.math.log(PARAMS['sigma_low']),
                                        PARAMS['num_L']))
#%%
@tf.function
def analytic_log_gmm_prob_grad(x, sigma):
    '''
    gradient of log(p * N(mu1, sigma_i) * (1 - p) * N(mu2, sigma_i))
    '''
    x_tensor = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        normal1 = tfd.MultivariateNormalDiag(loc=mean_parameters[0],
                                            scale_diag=[[sigma, sigma]])
        normal2 = tfd.MultivariateNormalDiag(loc=mean_parameters[1],
                                            scale_diag=[[sigma, sigma]])
        probs = list()
        probs.append(tf.math.log(tf.convert_to_tensor(0.2)) + normal1.log_prob(x_tensor)) # log \pi_1 + log normal1
        probs.append(tf.math.log(tf.convert_to_tensor(0.8)) + normal2.log_prob(x_tensor)) # log \pi_2 + log normal2
        log_prob = tf.reduce_logsumexp(tf.stack(probs, axis=0), axis=0)
    true_gradients = tape.gradient(log_prob, x_tensor)
    return true_gradients

@tf.function
def langevin_dynamics(grad_function, x, sigma, T, alpha):
    for _ in range(T):
        score = grad_function(x, sigma)
        noise = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1)
        x = x + (alpha / 2) * score + tf.sqrt(alpha) * noise
    return x

def annealed_langevin_dynamics(grad_function, x, sigma_levels, T, epsilon, intermediate=False):
    if intermediate:
        x_list = []
        for sigma_i in sigma_levels:
            alpha_i = epsilon * (sigma_i ** 2) / (sigma_levels[-1] ** 2) # step size
            x = langevin_dynamics(grad_function, x, sigma=sigma_i, T=T, alpha=alpha_i) # Langevin dynamics
            x_list.append(x)
        return x_list
    else:
        for sigma_i in sigma_levels:
            alpha_i = epsilon * (sigma_i ** 2) / (sigma_levels[-1] ** 2) # step size
            x = langevin_dynamics(grad_function, x, sigma=sigma_i, T=T, alpha=alpha_i) # Langevin dynamics
        return x

def meshgrid(x):
    y = x
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)
#%%
'''plot density and generated samples'''
x_init = tf.random.uniform(shape=(2560, 2), minval=-8, maxval=8)
annealed_samples = annealed_langevin_dynamics(analytic_log_gmm_prob_grad, x_init, sigma_levels, T=100, epsilon=PARAMS['epsilon'], intermediate=True)
samples = langevin_dynamics(analytic_log_gmm_prob_grad, x_init, variance_parameters[0], T=1000, alpha=PARAMS['epsilon'])
#%%
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
for i in range(PARAMS['num_L']):
    axes.flatten()[i].scatter(annealed_samples[i].numpy()[:, 0], annealed_samples[i].numpy()[:, 1], s=7, alpha=0.3, color='black')
    axes.flatten()[i].set_xlabel(r'$x$')
    axes.flatten()[i].set_ylabel(r'$y$')
    axes.flatten()[i].set_title('$\sigma_{}$: {}'.format(i+1, sigma_levels[i].numpy()))
plt.savefig('./assets/annealed_langevin_step.png', bbox_inches="tight")
plt.show()
#%%
x = np.linspace(-8, 8, 500, dtype=np.float32)
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes.flatten()[0].imshow(gmm.prob(meshgrid(x)), cmap='inferno', extent=[-8, 8, -8, 8], origin='lower')
axes.flatten()[0].set_xlabel(r'$x$')
axes.flatten()[0].set_ylabel(r'$y$')
axes.flatten()[0].set_title('ground truth density')
axes.flatten()[1].scatter(samples.numpy()[:, 0], samples.numpy()[:, 1], s=7, alpha=0.3, color='black')
axes.flatten()[1].set_xlabel(r'$x$')
axes.flatten()[1].set_ylabel(r'$y$')
axes.flatten()[1].set_title('Langevin dynamics samples')
axes.flatten()[2].scatter(annealed_samples[i].numpy()[:, 0], annealed_samples[i].numpy()[:, 1], s=7, alpha=0.3, color='black')
axes.flatten()[2].set_xlabel(r'$x$')
axes.flatten()[2].set_ylabel(r'$y$')
axes.flatten()[2].set_title('annealed Langevin dynamics samples')
plt.savefig('./assets/annealed_langevin.png', bbox_inches="tight")
plt.show()
#%%