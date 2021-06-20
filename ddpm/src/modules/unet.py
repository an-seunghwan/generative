#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers

import tensorflow_addons as tfa
#%%
def nonlinearity(x):
  return tf.nn.swish(x)
#%%
def normalize(x):
  return tfa.layers.GroupNormalization(1)(x)
#%%
def upsample(x, with_conv):
    B, H, W, C = tf.shape(x)
    x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    assert x.shape == [B, H * 2, W * 2, C]
    if with_conv:
        x = layers.Conv2D(filters=C, kernel_size=3, strides=1, padding='same', name='conv')(x)
        assert x.shape == [B, H * 2, W * 2, C]
    return x
#%%
def downsample(x, with_conv):
    B, H, W, C = x.shape
    if with_conv:
        x = layers.Conv2D(filters=C, kernel_size=3, strides=2, padding='same', name='conv')(x)
        # x = tf.nn.conv2d(x, name='conv', num_units=C, filter_size=3, stride=2)
    else:
        x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
        # x = tf.nn.avg_pool(x, 2, 2, 'SAME')
    assert x.shape == [B, H // 2, W // 2, C]
    return x
#%%
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
def resnet_block(x, temb, dropout, out_ch=None, conv_shortcut=False):
    B, H, W, C = tf.shape(x)

    if out_ch is None:
        out_ch = C

    h = x

    h = nonlinearity(normalize(h))
    h = layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='same', name='conv1')(h)
    # h = tf.nn.conv2d(h, name='conv1', num_units=out_ch)

    # add in timestep embedding
    h += layers.Dense(out_ch, name='temb_proj')(nonlinearity(temb))[:, None, None, :]
    # h += tf.nn.dense(nonlinearity(temb), name='temb_proj', num_units=out_ch)[:, None, None, :]

    h = nonlinearity(normalize(h))
    h = layers.Dropout(rate=dropout)(h)
    h = layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='same', name='conv2')(h)
    # h = nn.conv2d(h, name='conv2', num_units=out_ch, init_scale=0.)

    if C != out_ch:
        if conv_shortcut:
            x = layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='same', name='conv_shortcut')(x)
            # x = nn.conv2d(x, name='conv_shortcut', num_units=out_ch)
        else:
            x = layers.Dense(out_ch, name='nin_shortcut')(x)
            # x = nn.nin(x, name='nin_shortcut', num_units=out_ch)

    assert x.shape == h.shape
    return x + h
#%%
def attn_block(x):
    B, H, W, C = tf.shape(x)
    h = normalize(x)
    q = layers.Dense(C, name='q')(h)
    k = layers.Dense(C, name='k')(h)
    v = layers.Dense(C, name='v')(h)
    # q = nn.nin(h, name='q', num_units=C)
    # k = nn.nin(h, name='k', num_units=C)
    # v = nn.nin(h, name='v', num_units=C)

    w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = tf.reshape(w, [B, H, W, H * W])
    w = tf.nn.softmax(w, -1)
    w = tf.reshape(w, [B, H, W, H, W])

    h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
    h = layers.Dense(C, name='proj_out')(h)
    # h = nn.nin(h, name='proj_out', num_units=C, init_scale=0.)

    assert h.shape == x.shape
    return x + h
#%%
def model(x, timesteps, embedding_dim, out_ch, 
        dropout=0., embedding_dim_mult=(1, 2, 4, 8), num_res_blocks=3, attn_resolutions=(16, ), resamp_with_conv=True):
    
    # x = layers.Input((32, 32, 3))
    # timesteps = layers.Input(())
    B, _, _, _ = tf.shape(x)
    num_resolutions = len(embedding_dim_mult)

    '''Timestep embedding'''
    temb = get_timestep_embedding(timesteps, embedding_dim)
    temb = layers.Dense(embedding_dim * 4, name='dense0')(temb)
    # temb = nn.dense(temb, name='dense0', num_units=ch * 4)
    temb = layers.Dense(embedding_dim * 4, name='dense1')(nonlinearity(temb))
    # temb = nn.dense(nonlinearity(temb), name='dense1', num_units=ch * 4)
    assert temb.shape == [B, embedding_dim * 4]

    '''Downsampling'''
    hs = [layers.Conv2D(filters=embedding_dim, kernel_size=3, strides=1, padding='same', name='conv_in')(x)]
    # hs = [nn.conv2d(x, name='conv_in', num_units=ch)]
    for i_level in range(num_resolutions):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks):
            h = resnet_block(hs[-1], temb=temb, out_ch=embedding_dim * embedding_dim_mult[i_level], dropout=dropout)
            # h = resnet_block(hs[-1], name='block_{}'.format(i_block), temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
            if tf.shape(h)[1] in attn_resolutions:
                h = attn_block(h)
            hs.append(h)
            # Downsample
        if i_level != num_resolutions - 1:
            hs.append(downsample(hs[-1], with_conv=resamp_with_conv))

    '''Middle'''
    h = hs[-1]
    h = resnet_block(h, temb=temb, dropout=dropout)
    # h = resnet_block(h, temb=temb, name='block_1', dropout=dropout)
    h = attn_block(h)
    # h = attn_block(h, name='attn_1'.format(i_block), temb=temb)
    h = resnet_block(h, temb=temb, dropout=dropout)
    # h = resnet_block(h, temb=temb, name='block_2', dropout=dropout)

    '''Upsampling'''
    for i_level in reversed(range(num_resolutions)):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks + 1):
            h = resnet_block(tf.concat([h, hs.pop()], axis=-1), temb=temb, out_ch=embedding_dim * embedding_dim_mult[i_level], dropout=dropout)
            if tf.shape(h)[1] in attn_resolutions:
                h = attn_block(h)
        # Upsample
        if i_level != 0:
            h = upsample(h, with_conv=resamp_with_conv)

    '''End'''
    h = nonlinearity(normalize(h))
    h = layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='same', name='conv_out')(h)
    # h = nn.conv2d(h, name='conv_out', num_units=out_ch, init_scale=0.)
    assert h.shape == x.shape[:3] + [out_ch]
    
    return h
    # return K.models.Model([x, timesteps], h)
#%%