#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers

import tensorflow_addons as tfa
#%%
@tf.function
def nonlinearity(x):
      return tf.nn.swish(x)
#%%
@tf.function
def normalize(x):
  return tfa.layers.GroupNormalization(1)(x)
#%%
@tf.function
def upsample(x, with_conv):
    B, H, W, C = tf.shape(x)
    x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    assert x.shape == [B, H * 2, W * 2, C]
    if with_conv:
        x = layers.Conv2D(filters=C, kernel_size=3, strides=1, padding='same', name='conv')(x)
        assert x.shape == [B, H * 2, W * 2, C]
    return x
#%%
@tf.function
def downsample(x, with_conv):
    B, H, W, C = x.shape
    if with_conv:
        x = layers.Conv2D(filters=C, kernel_size=3, strides=2, padding='same', name='conv')(x)
    else:
        x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    assert x.shape == [B, H // 2, W // 2, C]
    return x
#%%
@tf.function
def get_timestep_embedding(timesteps, embedding_dim):
    """ 
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    
    half_dim = embedding_dim // 2
    emb = tf.math.log(10000.0) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == [tf.shape(timesteps)[0], embedding_dim]
    return emb
#%%
@tf.function
def resnet_block(x, temb, dropout, out_ch=None, conv_shortcut=False):
    B, H, W, C = tf.shape(x)

    if out_ch is None:
        out_ch = C

    h = x

    h = nonlinearity(normalize(h))
    h = layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='same', name='conv1')(h)

    # add in timestep embedding
    h += layers.Dense(out_ch, name='temb_proj')(nonlinearity(temb))[:, None, None, :]

    h = nonlinearity(normalize(h))
    h = layers.Dropout(rate=dropout)(h)
    h = layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='same', name='conv2')(h)

    if C != out_ch:
        if conv_shortcut:
            x = layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='same', name='conv_shortcut')(x)
        else:
            x = layers.Dense(out_ch, name='nin_shortcut')(x)

    assert x.shape == h.shape
    return x + h
#%%
@tf.function
def attn_block(x):
    B, H, W, C = tf.shape(x)
    h = normalize(x)
    q = layers.Dense(C, name='q')(h)
    k = layers.Dense(C, name='k')(h)
    v = layers.Dense(C, name='v')(h)

    w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = tf.reshape(w, [B, H, W, H * W])
    w = tf.nn.softmax(w, -1)
    w = tf.reshape(w, [B, H, W, H, W])

    h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
    h = layers.Dense(C, name='proj_out')(h)

    assert h.shape == x.shape
    return x + h
#%%
class Unet(K.models.Model):
    def __init__(self, params, embedding_dim, out_ch, dropout=0., embedding_dim_mult=(1, 2, 4, 8), num_res_blocks=3, attn_resolutions=(16, ), resamp_with_conv=True):
        super(Unet, self).__init__()
        
        self.params = params
        self.embedding_dim = embedding_dim
        self.out_ch = out_ch
        self.dropout = dropout
        self.embedding_dim_mult = embedding_dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.resamp_with_conv = resamp_with_conv
        
        self.nonlinearity = nonlinearity
        self.normalize = normalize
        self.upsample = upsample
        self.downsample = downsample
        self.get_timestep_embedding = get_timestep_embedding
        self.attn_block = attn_block
        self.resnet_block = resnet_block
        
        self.dense0 = layers.Dense(self.embedding_dim * 4, name='dense0')
        self.dense1 = layers.Dense(self.embedding_dim * 4, name='dense1')
        self.conv_in = layers.Conv2D(filters=self.embedding_dim, kernel_size=3, strides=1, padding='same', name='conv_in')
        self.conv_out = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv_out')

    def call(self, x, timesteps, **kwargs):
        # B, _, _, _ = tf.shape(x)
        B = self.params['batch_size']
        num_resolutions = len(self.embedding_dim_mult)

        '''Timestep embedding'''
        temb = get_timestep_embedding(timesteps, self.embedding_dim)
        temb = self.dense0(temb)
        temb = self.dense1(nonlinearity(temb))
        # assert temb.shape == [B, self.embedding_dim * 4]

        '''Downsampling'''
        hs = [self.conv_in(x)]
        # hs = [nn.conv2d(x, name='conv_in', num_units=ch)]
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = resnet_block(hs[-1], temb=temb, out_ch=self.embedding_dim * self.embedding_dim_mult[i_level], dropout=self.dropout)
                if h.shape[1] in self.attn_resolutions:
                    h = attn_block(h)
                hs.append(h)
                # Downsample
            if i_level != num_resolutions - 1:
                hs.append(downsample(hs[-1], with_conv=self.resamp_with_conv))

        '''Middle'''
        h = hs[-1]
        h = resnet_block(h, temb=temb, dropout=self.dropout)
        h = attn_block(h)
        h = resnet_block(h, temb=temb, dropout=self.dropout)

        '''Upsampling'''
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks + 1):
                h = resnet_block(tf.concat([h, hs.pop()], axis=-1), temb=temb, out_ch=self.embedding_dim * self.embedding_dim_mult[i_level], dropout=self.dropout)
                if h.shape[1] in self.attn_resolutions:
                    h = attn_block(h)
            # Upsample
            if i_level != 0:
                h = upsample(h, with_conv=self.resamp_with_conv)

        '''End'''
        h = nonlinearity(normalize(h))
        h = self.conv_out(h)
        # assert h.shape == x.shape[:3] + [self.out_ch]
        
        return h
#%%