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
    "num_L":10,
    "sigma_low":1.0,
    "sigma_high":20.0
}
#%%
import tensorflow_probability as tfp
tfd = tfp.distributions

def gmm(probs, loc, scale):
    gmm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=probs),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc,
            scale_identity_multiplier=scale))
    return gmm

mixture_prob = [0.2, 0.8]
mean_parameters = [[-5, -5], [5, 5]]
variance_parameters = [1, 1]
gmm = gmm(mixture_prob, mean_parameters, variance_parameters)
#%%
def sample(gmm, sample_shape):
    s = tfd.Sample(gmm, sample_shape=sample_shape)
    return s.sample()
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
'''sliced score matching'''
@tf.function
def ssm_loss(scorenet, x_batch):
    sum_over = list(range(1, len(x_batch.shape)))
    v = tf.random.normal(x_batch.shape)
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        scores = scorenet(x_batch)
        v_times_scores = tf.reduce_sum(v * scores) # v^\top * s(x) -> scalar
    grad = tape.gradient(v_times_scores, x_batch) # d/dx (v^\top * s(x)) = v^\top * (d/dx s(x))
    loss_first = tf.reduce_sum(grad * v, axis=sum_over) # v^\top * (d/dx s(x)) * v
    loss_second = 0.5 * tf.reduce_sum(tf.square(scores), axis=sum_over)
    loss = tf.reduce_mean(loss_first + loss_second)
    return loss
#%%
model = ModelMLP(activation=tf.nn.softplus)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

step = 0
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))
#%%
'''training'''
avg_loss = 0
for _ in progress_bar:
    x_batch = sample(gmm, PARAMS['batch_size'])
    step += 1

    with tf.GradientTape(persistent=True) as tape:
        current_loss = ssm_loss(model, x_batch)
        gradients = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    progress_bar.set_description('iteration {}/{} | current loss {:.3f}'.format(
        step, PARAMS['epochs'], current_loss
    ))

    avg_loss += current_loss
    
    if step == PARAMS['epochs']: break
#%%
@tf.function
def analytic_log_gmm_prob_grad(x, sigma_i):
    '''
    ground truth gradient of gmm
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
        noise = tf.sqrt(alpha) * tf.random.normal(shape=x.get_shape(), mean=0, stddev=1)
        x = x + (alpha / 2) * score + noise
    return x

@tf.function
def annealed_langevin_dynamics(grad_function, x, sigma_levels, T=100, eps=0.1):
    for sigma_i in sigma_levels:
        alpha_i = eps * (sigma_i ** 2) / (sigma_levels[-1] ** 2) # step size
        x = langevin_dynamics(grad_function, x, sigma_i=sigma_i, alpha=alpha_i, T=T)
    return x
#%%
# geometric sequence of sigma
sigma_levels = tf.math.exp(tf.linspace(tf.math.log(PARAMS['sigma_high']),
                                        tf.math.log(PARAMS['sigma_low']),
                                        PARAMS['num_L']))
#%%
epsilon = 0.1
x_init = tf.random.uniform(shape=(1280, 2), minval=-8, maxval=8)
'''
analytic_log_prob_grad: ground truth data score of gmm
'''
samples = annealed_langevin_dynamics(analytic_log_gmm_prob_grad, x_init, sigma_levels, T=100, eps=epsilon)
plt.scatter(samples.numpy()[:, 0], samples.numpy()[:, 1], s=0.5, marker='.', color='black')
#%%
def meshgrid(x):
    y = x
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)

def visualize_density(gmm, x):
    grid = meshgrid(x)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gmm.prob(grid), cmap='inferno', extent=[-8, 8, -8, 8], origin='lower')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    # plt.savefig("density.pdf", bbox_inches="tight")
    plt.show()
    return

def visualize_gradients(x, grads, filename="gradients"):
    U, V = grads[:, :, 1], grads[:, :, 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(x, x, U, V)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    # plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.show()
    return
#%%
# define grids
x = np.linspace(-8, 8, 500, dtype=np.float32)
x_for_grads = np.linspace(-8, 8, num=20)

# plot density
visualize_density(gmm, x)

# compute analytic gradients
grid = meshgrid(x_for_grads)
true_grads = analytic_log_gmm_prob_grad(grid, 1)

# compute estimated gradients (scores)
estimated_grads = model(grid)

# visualize gradients
visualize_gradients(x_for_grads, true_grads, "grad_analytic")
visualize_gradients(x_for_grads, estimated_grads, "grad_est")
#%%