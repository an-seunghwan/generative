#%%
'''
Technique 3. Noise conditioning is used
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
#%%
class CondInstanceNormPlusPlus2D(layers.Layer):
    def __init__(self, params, feature_num):
        super(CondInstanceNormPlusPlus2D, self).__init__()
        self.params = params
        self.L = self.params['num_L']
        self.feature_num = feature_num
        
        self.alpha = layers.Embedding(input_dim=self.L,
                                      output_dim=self.feature_num)
        self.beta = layers.Embedding(input_dim=self.L,
                                      output_dim=self.feature_num)
        self.gamma = layers.Embedding(input_dim=self.L,
                                      output_dim=self.feature_num)

    def call(self, x, idx_simgas, **kwargs):
        mu, s = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        m, v = tf.nn.moments(mu, axes=[-1], keepdims=True)

        first = self.gamma(idx_simgas)[:, tf.newaxis, tf.newaxis, :] * (x - mu) / tf.sqrt(s + 1e-6)
        second = self.beta(idx_simgas)[:, tf.newaxis, tf.newaxis, :]
        third = self.alpha(idx_simgas)[:, tf.newaxis, tf.newaxis, :] * (mu - m) / tf.sqrt(v + 1e-6)

        z = first + second + third

        return z
#%%
class ResidualConvUnit(layers.Layer):
    def __init__(self, params, activation, filters, kernel_size=3, n_block=2):
        super(ResidualConvUnit, self).__init__()

        self.params = params
        self.activation = activation
        self.kernel_size = kernel_size
        self.filters = filters
        self.n_block = n_block
        
        self.conv = []
        self.norm = []
        for _ in range(self.n_block):
            self.conv.append(layers.Conv2D(self.filters, self.kernel_size, 1, padding='same'))
            self.norm.append(CondInstanceNormPlusPlus2D(self.params, self.filters))
        
    def call(self, inputs, idx_simgas, **kwargs):
        path = inputs
        for i in range(self.n_block):
            path = self.activation(path)
            path = self.norm[i](path, idx_simgas)
            path = self.conv[i](path)

        return inputs + path
#%%
class MultiResolutionFusion(layers.Layer):
    def __init__(self, params, filters_high, filters_low, filters, kernel_size=3):
        super(MultiResolutionFusion, self).__init__()
        '''
        Fuses lower and higher resolution input feature maps.
        Lower inputs: coming from the previous RefineNet
        Higher Inputs: coming from the ResBlock of ResNet
        '''

        self.params = params
        self.kernel_size = kernel_size
        self.filters_high = filters_high
        self.filters_low = filters_low
        self.filters = filters
        
        self.norm_high = CondInstanceNormPlusPlus2D(self.params, self.filters_high)
        self.conv_high = layers.Conv2D(self.filters, self.kernel_size, 1, padding='same') 
        self.norm_low = CondInstanceNormPlusPlus2D(self.params, self.filters_low)
        self.conv_low = layers.Conv2D(self.filters, self.kernel_size, 1, padding='same')
        self.upsampling = layers.UpSampling2D(size=(2, 2))

    def call(self, high_inputs, low_inputs, idx_simgas, **kwargs):
        high_x = self.norm_high(high_inputs, idx_simgas)
        high_x = self.conv_high(high_x)
        
        low_x = self.norm_low(low_inputs, idx_simgas)
        low_x = self.conv_low(low_x)
        low_x = self.upsampling(low_x)
        
        return low_x + high_x
#%%
class ChainedResidualPooling(layers.Layer):
    def __init__(self, n_stage, params, activation, filters, kernel_size=3):
        super(ChainedResidualPooling, self).__init__()

        self.params = params
        self.activation = activation
        self.kernel_size = kernel_size
        self.filters = filters
        self.n_stage = n_stage
        
        self.pools = []
        self.norms = []
        self.convs = []
        for _ in range(self.n_stage):
            self.pools.append(layers.AveragePooling2D(pool_size=(5, 5), strides=(1, 1), padding="same"))
            self.norms.append(CondInstanceNormPlusPlus2D(self.params, self.filters))
            self.convs.append(layers.Conv2D(self.filters, self.kernel_size, 1, padding='same'))

    def call(self, inputs, idx_simgas, **kwargs):
        x = self.activation(inputs)
        path = x
        for i in range(self.n_stage):
            path = self.pools[i](path)
            path = self.norms[i](path, idx_simgas)
            path = self.convs[i](path)
            x = path + x

        return x
#%%