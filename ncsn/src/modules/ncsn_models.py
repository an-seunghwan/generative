#%%
'''
Technique 3. Noise conditioning is used
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers

from modules import ncsn_layers
#%%
class ResBlock(layers.Layer):
    def __init__(self, params, activation, filters_before, filters, kernel_size=3, down=False, dilation=None):
        super(ResBlock, self).__init__()
        '''
        pre-activation residual block
        '''

        self.params = params
        self.activation = activation
        self.kernel_size = kernel_size
        self.filters_before = filters_before
        self.filters = filters
        self.down = down # subsampling 
        self.dilation = dilation # works only with strides = 1
        
        self.norm1 = ncsn_layers.InstanceNormPlusPlus2D(self.params, self.filters_before)
        self.norm2 = ncsn_layers.InstanceNormPlusPlus2D(self.params, self.filters)        
        if down:
            if dilation == None:
                self.conv1 = layers.Conv2D(self.filters, self.kernel_size, strides=2, padding='same') # subsampling
                self.conv2 = layers.Conv2D(self.filters, self.kernel_size, strides=1, padding='same')
            else:
                self.conv1 = layers.Conv2D(self.filters, self.kernel_size, strides=2, padding='same') # subsampling
                self.conv2 = layers.Conv2D(self.filters, self.kernel_size, strides=1, dilation_rate=self.dilation, padding='same')
            
            self.conv_shortcut = layers.Conv2D(self.filters, self.kernel_size, strides=2, padding='same')                 
        else:
            if dilation == None:
                self.conv1 = layers.Conv2D(self.filters, self.kernel_size, strides=1, padding='same') 
                self.conv2 = layers.Conv2D(self.filters, self.kernel_size, strides=1, padding='same')
            else:
                self.conv1 = layers.Conv2D(self.filters, self.kernel_size, strides=1, padding='same') 
                self.conv2 = layers.Conv2D(self.filters, self.kernel_size, strides=1, dilation_rate=self.dilation, padding='same')

    def call(self, inputs, **kwargs):
        x = self.norm1(inputs)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.down:
            skip_x = self.conv_shortcut(inputs)
        else:
            skip_x = inputs

        return skip_x + x
#%%
class RefineBlock(layers.Layer):
    def __init__(self, params, activation, filters_high, filters_low, filters, kernel_size=3, n_rcu_block=2, n_crp_stage=3):
        super(RefineBlock, self).__init__()

        self.params = params
        self.activation = activation
        self.kernel_size = kernel_size
        self.filters_high = filters_high
        self.filters_low = filters_low
        self.filters = filters
        self.n_rcu_block = n_rcu_block
        self.n_crp_stage = n_crp_stage
        
        self.RCUBlock_high = ncsn_layers.ResidualConvUnit(self.params, self.activation, self.filters_high, self.kernel_size, self.n_rcu_block)
        if self.filters_low is not None: # first block in RefineNet
            self.RCUBlock_low = ncsn_layers.ResidualConvUnit(self.params, self.activation, self.filters_low, self.kernel_size, self.n_rcu_block)
        self.MRFBlock = ncsn_layers.MultiResolutionFusion(self.params, self.filters, self.kernel_size)
        self.CRPBlock = ncsn_layers.ChainedResidualPooling(self.n_crp_stage, self.params, self.activation, self.filters, self.kernel_size)
        self.RCUBlock_end = ncsn_layers.ResidualConvUnit(self.params, self.activation, self.filters, self.kernel_size, 1)

    def call(self, high_inputs, low_inputs, **kwargs):
        if low_inputs == None:
            high_x = self.RCUBlock_high(high_inputs)
            x = high_x
        else:
            high_x = self.RCUBlock_high(high_inputs)
            low_x = self.RCUBlock_low(low_inputs)
            x = self.MRFBlock(high_x, low_x)
            
        x = self.CRPBlock(x)
        x = self.RCUBlock_end(x)
        
        return x
#%%
def build_refinenet(PARAMS, activation):
    inputs = layers.Input((PARAMS["data_dim"], PARAMS["data_dim"], PARAMS["channel"]))
    
    instancenorm_start = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, PARAMS['channel'])
    conv_start = layers.Conv2D(128, 3, 1, padding='same') 
    x = conv_start(instancenorm_start(inputs))
    
    ResBlock1 = ResBlock(PARAMS, activation, x.shape[-1], 128, 3, False, None)
    x1 = ResBlock1(x)
    ResBlock2 = ResBlock(PARAMS, activation, x1.shape[-1], 128, 3, False, None)
    x2 = ResBlock2(x1)
    
    ResBlock3 = ResBlock(PARAMS, activation, x2.shape[-1], 256, 3, True, None)
    x3 = ResBlock3(x2)
    ResBlock4 = ResBlock(PARAMS, activation, x3.shape[-1], 256, 3, False, None)
    x4 = ResBlock4(x3)
    
    ResBlock5 = ResBlock(PARAMS, activation, x4.shape[-1], 256, 3, True, 2)
    x5 = ResBlock5(x4)
    ResBlock6 = ResBlock(PARAMS, activation, x5.shape[-1], 256, 3, False, 2)
    x6 = ResBlock6(x5)
    
    ResBlock7 = ResBlock(PARAMS, activation, x6.shape[-1], 256, 3, True, 4)
    x7 = ResBlock7(x6)
    ResBlock8 = ResBlock(PARAMS, activation, x7.shape[-1], 256, 3, False, 4)
    x8 = ResBlock8(x7)
    
    RefineBlock1 = RefineBlock(PARAMS, activation, x8.shape[-1], None, 256, 3, 2)
    y1 = RefineBlock1(x8, None)
    
    RefineBlock2 = RefineBlock(PARAMS, activation, x6.shape[-1], y1.shape[-1], 256, 3, 2)
    y2 = RefineBlock2(x6, y1)
    
    RefineBlock3 = RefineBlock(PARAMS, activation, x4.shape[-1], y2.shape[-1], 128, 3, 2)
    y3 = RefineBlock3(x4, y2)
    
    RefineBlock4 = RefineBlock(PARAMS, activation, x2.shape[-1], y3.shape[-1], 128, 3, 2)
    y4 = RefineBlock4(x2, y3)
    
    instancenorm_end = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, y4.shape[-1])
    conv_end = layers.Conv2D(PARAMS['channel'], 3, 1, padding='same') 
    outputs = conv_end(instancenorm_end(y4))
    
    model = K.models.Model(inputs, outputs)
    
    model.summary()
    
    return model
#%%
def build_unet(PARAMS):
    input_size = (PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel'])

    '''contracting path'''
    inputs = layers.Input(input_size)
    inputs_conv = layers.Conv2D(filters = 128, kernel_size = 3, activation = 'elu', padding = 'same')(inputs)
    
    norm1 = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, 128)(inputs_conv)
    conv1 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same')(norm1)
    conv1 = inputs_conv + conv1
    norm1 = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, 128)(conv1)
    conv1 = layers.Conv2D(128, 3, strides=2, activation = 'elu', padding = 'same')(norm1) 

    norm2 = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, 128)(conv1)
    conv2 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same')(norm2)
    conv2 = conv1 + conv2
    norm2 = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, 128)(conv2)
    conv2 = layers.Conv2D(256, 3, 2, activation = 'elu', padding = 'same')(norm2) 

    norm3 = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, 256)(conv2)
    conv3 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same')(norm3)
    conv3 = conv2 + conv3
    norm3 = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, 256)(conv3)
    conv3 = layers.Conv2D(256, 3, 2, activation = 'elu', padding = 'same')(norm3)

    norm4 = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, 256)(conv3)
    conv4 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same')(norm4)
    conv4 = conv3 + conv4
    norm4 = ncsn_layers.InstanceNormPlusPlus2D(PARAMS, 256)(conv4)
    conv4 = layers.Conv2D(256, 3, 2, activation = 'elu', padding = 'same')(norm4) 

    '''expanding path'''
    upconv4 = layers.UpSampling2D(size = (2, 2))(conv4) 
    conv5 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same')(upconv4)
    conv5 = layers.concatenate([conv5, conv3], axis = 3)
    conv5 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same')(conv5)
    conv5 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same')(conv5)

    upconv5 = layers.UpSampling2D(size = (2, 2))(conv5) 
    conv6 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same')(upconv5) 
    conv6 = layers.concatenate([conv6, conv2], axis = 3)
    conv6 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same')(conv6)
    conv6 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same')(conv6)

    upconv6 = layers.UpSampling2D(size = (2, 2))(conv6) 
    conv7 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same')(upconv6) 
    conv7 = layers.concatenate([conv7, conv1], axis = 3)
    conv7 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same')(conv7)
    conv7 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same')(conv7)
    
    upconv7 = layers.UpSampling2D(size = (2, 2))(conv7) 
    conv8 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same')(upconv7)
    conv8 = layers.concatenate([conv8, inputs_conv], axis = 3)
    conv8 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same')(conv8) 
    conv8 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same')(conv8) 

    '''output layer'''
    conv9 = layers.Conv2D(PARAMS['channel'], 1)(conv8)

    model = K.models.Model(inputs, conv9)

    model.summary()

    return model
#%%