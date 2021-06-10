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
    "T": 100,
    "epsilon": 0.01,
    # "num_epsilon": 8,
    # "epsilon_low": 0.01, 
    # "epsilon_high": 1.0,
}

key = 'langevin' 
#%%
'''true data distribution (Gaussian mixture)'''
import tensorflow_probability as tfp
tfd = tfp.distributions

def build_gaussian(loc, scale):
    return tfp.distributions.MultivariateNormalFullCovariance(
                loc=loc, covariance_matrix=scale
            )
    
def sample(gaussian, sample_shape):
    s = tfd.Sample(gaussian, sample_shape=sample_shape)
    return s.sample()
#%%
# # geometric sequence of epsilon
# epsilon_levels = tf.math.exp(tf.linspace(tf.math.log(PARAMS['epsilon_high']),
#                                         tf.math.log(PARAMS['epsilon_low']),
#                                         PARAMS['num_epsilon']))
#%%
# @tf.function
# def analytic_log_gmm_prob_grad(x):
#     '''
#     ground truth gradient data score of gmm
#     d/dx log gmm 
#     = d/dx log (\pi_1 normal1 + \pi_2 normal2)
#     = d/dx log exp (log \pi_1 + log normal1 + log \pi_2 + log normal2), where exp is element-wise
#     '''
#     x_tensor = tf.convert_to_tensor(x)
#     with tf.GradientTape() as tape:
#         tape.watch(x_tensor)
#         normal1 = tfd.MultivariateNormalDiag(loc=mean_parameters[0],
#                                             scale_diag=[variance_parameters])
#         normal2 = tfd.MultivariateNormalDiag(loc=mean_parameters[1],
#                                             scale_diag=[variance_parameters])
#         probs = list()
#         probs.append(tf.math.log(tf.convert_to_tensor(0.2)) + normal1.log_prob(x_tensor)) # log \pi_1 + log normal1
#         probs.append(tf.math.log(tf.convert_to_tensor(0.8)) + normal2.log_prob(x_tensor)) # log \pi_2 + log normal2
#         log_prob = tf.reduce_logsumexp(tf.stack(probs, axis=0), axis=0)
#     true_gradients = tape.gradient(log_prob, x_tensor)
#     return true_gradients

@tf.function
def analytic_log_gaussian_prob_grad(x, mean_parameters, variance_parameters):
    x_tensor = tf.cast(x, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        normal = tfd.MultivariateNormalFullCovariance(loc=mean_parameters,
                                                    covariance_matrix=variance_parameters)
        log_probs = normal.log_prob(x_tensor)
    true_gradients = tape.gradient(log_probs, x_tensor)
    return tf.cast(true_gradients, tf.float32)

@tf.function
def langevin_dynamics(grad_function, mean_parameters, variance_parameters, x, epsilon=None, T=1000):
    for _ in range(T):
        score = grad_function(x, mean_parameters, variance_parameters)
        noise = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1)
        x = x + (epsilon / 2) * score + tf.sqrt(epsilon) * noise
    return x

def meshgrid(x):
    y = x
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float64(gx), np.float64(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)
#%%
'''plot density and generated samples'''
x_init = tf.random.uniform(shape=(2560, 2), minval=-8, maxval=8)

rhos = np.linspace(0.91, 0.99, 9)
fig, axes = plt.subplots(3, int(len(rhos) / 3), figsize=(20, 20))
x = np.linspace(-8, 8, 500, dtype=np.float32)
for i in range(len(rhos)):
    rho = rhos[i]
    mean_parameters = [0, 0]
    variance_parameters = [[1, rho],
                           [rho, 1]]
    print('Langevin dymanics (rho = {})'.format(rho))
    samples = langevin_dynamics(analytic_log_gaussian_prob_grad, mean_parameters, variance_parameters, x_init, PARAMS['epsilon'], PARAMS['T'])
    axes.flatten()[i].scatter(samples.numpy()[:, 0], samples.numpy()[:, 1], s=7, alpha=0.2, color='black')
    axes.flatten()[i].set_title('rho = {}'.format(rho))
plt.savefig('./assets/langevin.png', bbox_inches="tight")
plt.show()
plt.close()
#%%
mean_parameters = [0, 0]
variance_parameters = [[1, 0.99],
                        [0.99, 1]]
# sample_ = np.random.multivariate_normal(mean_parameters, variance_parameters, size=(2560, ))
gaussian = build_gaussian(mean_parameters, variance_parameters)
sample_ = sample(gaussian, 2560)
plt.figure(figsize=(7, 7))
plt.scatter(sample_[:, 0], sample_[:, 1], s=7, alpha=0.2, color='black')
#%%
'''true score vs estimated score'''
x_for_grads = np.linspace(-8, 8, num=20)
grid = meshgrid(x_for_grads)
mean_parameters = [0, 0]
variance_parameters = [[1, 0.9],
                        [0.9, 1]]
@tf.function
def analytic_log_gaussian_prob_grad2(x, mean_parameters, variance_parameters):
    x_tensor = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        normal = tfd.MultivariateNormalFullCovariance(loc=mean_parameters,
                                                    covariance_matrix=variance_parameters)
        log_probs = normal.log_prob(x_tensor)
    true_gradients = tape.gradient(log_probs, x_tensor)
    return tf.cast(true_gradients, tf.float32)

true_grads = analytic_log_gaussian_prob_grad2(grid, mean_parameters, variance_parameters) # compute analytic gradients

U1, V1 = true_grads[:, :, 1], true_grads[:, :, 0]

fig, axes = plt.subplots(1, 1, figsize=(6, 6))
axes.quiver(x_for_grads, x_for_grads, U1, V1)
axes.set_xlabel(r'$x$')
axes.set_ylabel(r'$y$')
axes.set_title('data scores (rho=0.9)')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('./assets/analytic_score.png', bbox_inches="tight")
plt.show()
#%%