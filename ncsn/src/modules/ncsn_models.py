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
    conv1 = layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1) # 256x256x64
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1) # 128x128x64

    conv2 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2) # 128x128x128
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2) # 64x64x128

    conv3 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3) # 64x64x256
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3) # 32x32x256

    conv4 = layers.Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4) # 32x32x512
    drop4 = layers.Dropout(0.5)(conv4) # 32x32x512, implicit augmentation
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4) # 16x16x512

    '''bottle-neck'''
    conv5 = layers.Conv2D(1024, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5) # 16x16x1024, implicit augmentation

    '''expanding path'''
    updrop5 = layers.UpSampling2D(size = (2, 2))(drop5) # 32x32x1024
    up6 = layers.Conv2D(512, 2, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(updrop5) # 32x32x512
    merge6 = layers.concatenate([drop4, up6], axis = 3) # skip connection
    conv6 = layers.Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    upconv6 = layers.UpSampling2D(size = (2, 2))(conv6) # 64x64x512
    up7 = layers.Conv2D(256, 2, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(upconv6) #64x64x256
    merge7 = layers.concatenate([conv3, up7], axis = 3)
    conv7 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    upconv7 = layers.UpSampling2D(size = (2, 2))(conv7) # 128x128x256
    up8 = layers.Conv2D(128, 2, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(upconv7) # 128x128x128
    merge8 = layers.concatenate([conv2, up8], axis = 3)
    conv8 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    upconv8 = layers.UpSampling2D(size = (2, 2))(conv8) # 256x256x128
    up9 = layers.Conv2D(64, 2, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(upconv8) # 256x256x64
    merge9 = layers.concatenate([conv1, up9], axis = 3)
    conv9 = layers.Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = layers.Conv2D(32, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = layers.Conv2D(16, 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9) # 256x256x2, final feature map

    '''output layer'''
    conv10 = layers.Conv2D(PARAMS['channel'], 1)(conv9)

    model = K.models.Model(inputs, conv10)

    model.summary()

    return model
#%%