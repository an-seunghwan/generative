'''
Generative Modeling by Estimating Gradients of the Data Distribution
with toy example
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
os.chdir('/Users/anseunghwan/Documents/GitHub/generative')
#%%
PARAMS = {
    "batch_size": 128,
    "epochs": 10000, 
    "learning_rate": 0.001,
    "num_L": 10,
    "sigma_low": 1.0,
    "sigma_high": 20.0,
    "epsilon": 0.1
}

key = 'toy_ssm' # sliced score matching
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
'''scorenet'''
class ModelMLP(K.Model):
    def __init__(self, activation):
        super(ModelMLP, self).__init__()

        self.mlp1 = layers.Dense(128)  
        self.mlp2 = layers.Dense(128)
        self.mlp3 = layers.Dense(2) # dimension of the input, here 2D GMM samples
        self.activation = activation

    def call(self, inputs, **kwargs):
        x = self.mlp1(inputs)
        x = self.activation(x)
        x = self.mlp2(x)
        x = self.activation(x)
        x = self.mlp3(x)
        return x
#%%
'''loss of sliced score matching'''
@tf.function
def ssm_loss(scorenet, x_batch):
    sum_over = list(range(1, len(x_batch.shape)))
    v = tf.random.normal(x_batch.shape) # random vector follows Guassian
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        scores = scorenet(x_batch)
        v_times_scores = tf.reduce_sum(v * scores) # v^\top * s(x) (scalar)
    grad = tape.gradient(v_times_scores, x_batch) # d/dx (v^\top * s(x)) = v^\top * (d/dx s(x))
    loss_first = tf.reduce_sum(grad * v, axis=sum_over) # v^\top * (d/dx s(x)) * v
    loss_second = 0.5 * tf.reduce_sum(tf.square(scores), axis=sum_over)
    loss = tf.reduce_mean(loss_first + loss_second)
    return loss
#%%
model = ModelMLP(activation=tf.nn.softplus)
optimizer = tf.keras.optimizers.Adam(learning_rate=PARAMS['learning_rate'])

step = 0
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))
#%%
'''training'''
loss_history = []
for _ in progress_bar:
    x_batch = sample(gmm, PARAMS['batch_size'])
    step += 1

    with tf.GradientTape(persistent=True) as tape:
        current_loss = ssm_loss(model, x_batch)
        gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(current_loss.numpy())

    progress_bar.set_description('iteration {}/{} | current loss {:.3f}'.format(
        step, PARAMS['epochs'], current_loss
    ))

    if step == PARAMS['epochs']: break
#%%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(loss_history)
ax.set_title('loss')
plt.savefig('./ncsn/assets/loss_{}.png'.format(key), bbox_inches="tight")
plt.show()
#%%
@tf.function
def analytic_log_gmm_prob_grad(x, sigma_i):
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
                                            scale_diag=[[sigma_i, sigma_i]])
        normal2 = tfd.MultivariateNormalDiag(loc=mean_parameters[1],
                                            scale_diag=[[sigma_i, sigma_i]])
        probs = list()
        probs.append(tf.math.log(tf.convert_to_tensor(0.2)) + normal1.log_prob(x_tensor)) # log \pi_1 + log normal1
        probs.append(tf.math.log(tf.convert_to_tensor(0.8)) + normal2.log_prob(x_tensor)) # log \pi_2 + log normal2
        log_prob = tf.reduce_logsumexp(tf.stack(probs, axis=0), axis=0)
    true_gradients = tape.gradient(log_prob, x_tensor)
    return true_gradients

@tf.function
def langevin_dynamics(grad_function, x, sigma_i=None, alpha=0.1, T=1000):
    for _ in range(T):
        score = grad_function(x, sigma_i)
        noise = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1)
        x = x + (alpha / 2) * score + tf.sqrt(alpha) * noise
    return x

@tf.function
def annealed_langevin_dynamics(grad_function, x, sigma_levels, T=100, eps=0.1):
    for sigma_i in sigma_levels:
        alpha_i = eps * (sigma_i ** 2) / (sigma_levels[-1] ** 2) # step size
        x = langevin_dynamics(grad_function, x, sigma_i=sigma_i, alpha=alpha_i, T=T) # Langevin dynamics
    return x

def meshgrid(x):
    y = x
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)
#%%
'''geometric sequence of sigma'''
sigma_levels = tf.math.exp(tf.linspace(tf.math.log(PARAMS['sigma_high']),
                                        tf.math.log(PARAMS['sigma_low']),
                                        PARAMS['num_L']))

x_init = tf.random.uniform(shape=(1280, 2), minval=-8, maxval=8)
samples = annealed_langevin_dynamics(analytic_log_gmm_prob_grad, x_init, sigma_levels, T=100, eps=PARAMS['epsilon'])
#%%
'''plot density and generated samples'''
x = np.linspace(-8, 8, 500, dtype=np.float32)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes.flatten()[0].imshow(gmm.prob(meshgrid(x)), cmap='inferno', extent=[-8, 8, -8, 8], origin='lower')
axes.flatten()[0].set_xlabel(r'$x$')
axes.flatten()[0].set_ylabel(r'$y$')
axes.flatten()[0].set_title('ground truth density')
axes.flatten()[1].scatter(samples.numpy()[:, 0], samples.numpy()[:, 1], s=7, alpha=0.3, color='black')
axes.flatten()[1].set_title('annealed Langevin dynamics samples')
plt.savefig('./ncsn/assets/samples_{}.png'.format(key), bbox_inches="tight")
plt.show()
#%%
'''true score vs estimated score'''
x_for_grads = np.linspace(-8, 8, num=20)
grid = meshgrid(x_for_grads)
true_grads = analytic_log_gmm_prob_grad(grid, 1) # compute analytic gradients
estimated_grads = model(grid) # compute estimated gradients (scores)

U1, V1 = true_grads[:, :, 1], true_grads[:, :, 0]
U2, V2 = estimated_grads[:, :, 1], estimated_grads[:, :, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes.flatten()[0].quiver(x_for_grads, x_for_grads, U1, V1)
axes.flatten()[0].set_xlabel(r'$x$')
axes.flatten()[0].set_ylabel(r'$y$')
axes.flatten()[0].set_title('data scores')
axes.flatten()[1].quiver(x_for_grads, x_for_grads, U2, V2)
axes.flatten()[1].set_xlabel(r'$x$')
axes.flatten()[1].set_ylabel(r'$y$')
axes.flatten()[1].set_title('estimated scores')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('./ncsn/assets/scores_{}.png'.format(key), bbox_inches="tight")
plt.show()
#%%