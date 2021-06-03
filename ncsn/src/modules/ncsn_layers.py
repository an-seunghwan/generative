#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
#%%
class InstanceNormPlusPlus2D(layers.Layer):
    def __init__(self, params, feature_num):
        super(InstanceNormPlusPlus2D, self).__init__()
        self.params = params
        self.feature_num = feature_num
        
        self.alpha = self.add_weight(name='alpha', 
                                    shape=(1, 1, 1, self.feature_num),
                                    initializer='random_normal',
                                    trainable=True)
        self.beta = self.add_weight(name='beta', 
                                    shape=(1, 1, 1, self.feature_num),
                                    initializer='random_normal',   
                                    trainable=True)
        self.gamma = self.add_weight(name='gamma', 
                                     shape=(1, 1, 1, self.feature_num),
                                    initializer='random_normal',   
                                    trainable=True)

    def call(self, x, **kwargs):
        mu, s = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        m, v = tf.nn.moments(mu, axes=[-1], keepdims=True)

        first = self.gamma * (x - mu) / tf.sqrt(s + 1e-6)
        second = self.beta
        third = self.alpha * (mu - m) / tf.sqrt(v + 1e-6)

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
        for _ in range(self.n_block):
            self.conv.append(layers.Conv2D(self.filters, self.kernel_size, 1, padding='same'))
        
    def call(self, inputs, **kwargs):
        path = inputs
        for i in range(self.n_block):
            path = self.activation(path)
            path = self.conv[i](path)

        return inputs + path
#%%
class MultiResolutionFusion(layers.Layer):
    def __init__(self, params, filters, kernel_size=3):
        super(MultiResolutionFusion, self).__init__()
        '''
        Fuses lower and higher resolution input feature maps.
        Lower inputs: coming from the previous RefineNet
        Higher Inputs: coming from the ResBlock of ResNet
        '''

        self.params = params
        self.kernel_size = kernel_size
        self.filters = filters
        
        self.conv_high = layers.Conv2D(self.filters, self.kernel_size, 1, padding='same') 
        self.conv_low = layers.Conv2D(self.filters, self.kernel_size, 1, padding='same')
        self.upsampling = layers.UpSampling2D(size=(2, 2))

    def call(self, high_inputs, low_inputs, **kwargs):
        high_x = self.conv_high(high_inputs)
        low_x = self.conv_low(low_inputs)
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
        self.convs = []
        for _ in range(self.n_stage):
            self.pools.append(layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="same"))
            self.convs.append(layers.Conv2D(self.filters, self.kernel_size, 1, padding='same'))

    def call(self, inputs, **kwargs):
        x = self.activation(inputs)
        path = x
        for i in range(self.n_stage):
            path = self.pools[i](path)
            path = self.convs[i](path)
            x = path + x

        return x
#%%