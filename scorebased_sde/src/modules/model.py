#%%
import tensorflow.keras as K
from tensorflow.keras.layers import *
#%%
def BuildUnet(input_size = (256, 256, 1)):
    # input_size = (256, 256, 1)

    '''contracting path'''
    inputs = Input(input_size)
    conv1 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1) # 256x256x64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 128x128x64

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) # 128x128x128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 64x64x128

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) # 64x64x256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 32x32x256

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) # 32x32x512
    drop4 = Dropout(0.5)(conv4) # 32x32x512, implicit augmentation
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # 16x16x512

    '''bottle-neck'''
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5) # 16x16x1024, implicit augmentation

    '''expanding path'''
    updrop5 = UpSampling2D(size = (2, 2))(drop5) # 32x32x1024
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(updrop5) # 32x32x512
    merge6 = concatenate([drop4, up6], axis = 3) # skip connection
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    upconv6 = UpSampling2D(size = (2, 2))(conv6) # 64x64x512
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upconv6) #64x64x256
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    upconv7 = UpSampling2D(size = (2, 2))(conv7) # 128x128x256
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upconv7) # 128x128x128
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    upconv8 = UpSampling2D(size = (2, 2))(conv8) # 256x256x128
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upconv8) # 256x256x64
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) # 256x256x2, final feature map

    '''output layer'''
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = K.models.Model(inputs, conv10)

    return model
#%%