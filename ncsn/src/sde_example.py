#%%
'''
reproduction of https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/

Stochastic differential equations (SDEs) model dynamical systems that are subject to noise. 
In this recipe, we simulate an Ornstein-Uhlenbeck process, which is a solution of the Langevin equation. 
The Ornstein-Uhlenbeck process is stationary, Gaussian, and Markov, which makes it a good candidate to represent stationary random noise.
We will simulate this process with a numerical method called the Euler-Maruyama method. 
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
b = 0.5
sigma = 1.
#%%
dt = .001  # Time step.
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.
#%%
x = np.zeros(n)
#%%
'''
discretized stochastic differential equation and process
x_{t+1} = x_t + (-b * x_t) * dt + sigma * (dt)^1/2 * N(0, 1)

where 
b = 1/2, sigma = 1
x_{t+1} = x_t + (-1/2 * x_t) * dt + 1 * (dt)^1/2 * N(0, 1)
and (-1/2 * x_t) = score function of N(0, 1)
and distribution of x_T converges to N(0, \sigma^2 / 2b) = N(0, 1)
'''
for i in range(n - 1):
    x[i + 1] = x[i] + (-b * x[i]) * dt + sigma * np.sqrt(dt) * np.random.randn()
#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)
ax.set_title('simulation of langevin equation')
#%%
ntrials = 10000
X = np.zeros(ntrials)
# We create bins for the histograms.
bins = np.linspace(-3., 3., 100)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for i in range(n):
    # We update the process independently for all trials
    X += (-b * X) * dt + sigma * np.sqrt(dt) * np.random.randn(ntrials)
    # We display the histogram for a few points in time (10000 points of process)
    if i in (5, 50, 900):
        hist, _ = np.histogram(X, bins=bins, density=True)
        ax.plot((bins[1:] + bins[:-1]) / 2, hist,
                {5: '-', 50: '-.', 900: '--', }[i],
                label=f"t={i * dt:.2f}")
hist, _ = np.histogram(np.random.randn(X.shape[0]), bins=bins, density=True)
ax.plot((bins[1:] + bins[:-1]) / 2, hist, ':', label='N(0,1)')
ax.legend()
#%%
# '''
# numerical solution
# '''
# np.random.seed(520)
# ntrials = 10000
# # true
# X = np.zeros(ntrials)
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# for i in range(n):
#     X += (-b * X) * dt + sigma * np.sqrt(dt) * np.random.randn(ntrials)
# hist, _ = np.histogram(np.random.randn(X.shape[0]), density=True)
# ax.plot(hist, '-.', label='N(0,1)')

# # solution
# X0 = np.random.normal(0, 1, (ntrials, ))
# X = np.exp(-b * T) * X0 + np.sum(sigma * np.exp(-b * (T - dt)) * np.random.normal(0, dt, (ntrials, ntrials)), axis=1)
# hist, _ = np.histogram(X, density=True)
# ax.plot(hist, '-', label='solution')
# ax.legend()
#%%
'''
The error of the Euler-Maruyama method is of order sqrt(dt). The Milstein method is a more precise numerical scheme, of order dt.
'''
#%%