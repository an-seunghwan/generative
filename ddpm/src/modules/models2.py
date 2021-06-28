#%%
'''
Fixed:
tfa.layers.GroupNormalization(1) -> layers.LayerNormalization()
'''
#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np

# import tensorflow_addons as tfa
#%%
# def nonlinearity(x):
#   return tf.nn.swish(x)
#%%
# # FIXME
# def normalize(x):
#   return tfa.layers.GroupNormalization(1)(x)
#   return layers.LayerNormalization()(x)
#%%
class Upsampling(layers.Layer):
    def __init__(self, in_ch, with_conv):
        super(Upsampling, self).__init__()
        
        self.in_ch = in_ch
        self.with_conv = with_conv
        self.conv = layers.Conv2D(filters=self.in_ch, kernel_size=3, strides=1, padding='same', name='conv_up')

    def call(self, x, **kwargs):
        B, H, W, C = x.shape
        x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # assert x.shape == [B, H * 2, W * 2, C]
        if self.with_conv:
            x = self.conv(x)
            # assert x.shape == [B, H * 2, W * 2, C]
        return x    
#%%
class Downsampling(layers.Layer):
    def __init__(self, in_ch, with_conv):
        super(Downsampling, self).__init__()
        
        self.in_ch = in_ch
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = layers.Conv2D(filters=self.in_ch, kernel_size=3, strides=2, padding='same', name='conv_down')
        else:
            self.avgpool = layers.AveragePooling2D(pool_size=(2, 2), strides=2)

    def call(self, x, **kwargs):
        # B, H, W, C = x.shape
        if self.with_conv:
            x = self.conv(x)
        else:
            x = self.avgpool(x)
        # assert x.shape == [B, H // 2, W // 2, C]
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
    # assert emb.shape == [timesteps.shape[0], embedding_dim]
    return emb
#%%
class ResnetBlock(layers.Layer):
    def __init__(self, dropout, in_ch, out_ch=None):
        super(ResnetBlock, self).__init__()
        
        self.dropout = dropout
        self.in_ch = in_ch
        self.out_ch = out_ch
        if self.out_ch is None:
            self.out_ch = self.in_ch
        
        if self.out_ch != self.in_ch:
            self.shortcut = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv_shortcut')
        
        # self.nonlinearity = nonlinearity
        # self.normalize1 = tfa.layers.GroupNormalization(1)
        # self.normalize2 = tfa.layers.GroupNormalization(1)
        self.normalize1 = layers.LayerNormalization()
        self.normalize2 = layers.LayerNormalization()
        self.conv1 = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv1')
        self.temb_proj = layers.Dense(self.out_ch, name='temb_proj')
        self.dropout_layer = layers.Dropout(rate=self.dropout)
        self.conv2 = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv2')

    def call(self, x, temb, **kwargs):
        h = x
        h = tf.nn.swish(self.normalize1(h))
        h = self.conv1(h)

        # add in timestep embedding
        h += self.temb_proj(tf.nn.swish(temb))[:, None, None, :]

        h = tf.nn.swish(self.normalize2(h))
        h = self.dropout_layer(h)
        h = self.conv2(h)
        
        if self.out_ch != self.in_ch:
            x = self.shortcut(x)

        # assert x.shape == h.shape
        return x + h
#%%
class AttentionBlock(layers.Layer):
    def __init__(self, in_ch):
        super(AttentionBlock, self).__init__()
        
        self.in_ch = in_ch
        # self.normalize = tfa.layers.GroupNormalization(1)
        self.normalize = layers.LayerNormalization()
        self.q_layer = layers.Dense(self.in_ch, name='q')
        self.k_layer = layers.Dense(self.in_ch, name='k')
        self.v_layer = layers.Dense(self.in_ch, name='v')
        
        self.proj_out = layers.Dense(self.in_ch, name='proj_out')

    def call(self, x, **kwargs):
        B, H, W, C = x.shape
        h = self.normalize(x)
        q = self.q_layer(h)
        k = self.k_layer(h)
        v = self.v_layer(h)
        
        w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(self.in_ch) ** (-0.5))
        w = tf.reshape(w, [-1, H, W, H * W])
        w = tf.nn.softmax(w, -1)
        w = tf.reshape(w, [-1, H, W, H, W])
        
        h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
        h = self.proj_out(h)
        
        # assert h.shape == x.shape
        return x + h
#%%
def build_unet(PARAMS, embedding_dim, dropout=0., embedding_dim_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=(16, ), resamp_with_conv=True):
    x = layers.Input((PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']))
    timesteps = layers.Input(())

    num_resolutions = len(embedding_dim_mult)

    '''Timestep embedding'''
    temb = get_timestep_embedding(timesteps, embedding_dim)
    temb = layers.Dense(embedding_dim * 4, name='dense0')(temb)
    temb = layers.Dense(embedding_dim * 4, name='dense1')(tf.nn.swish(temb))
    # assert temb.shape == [B, self.embedding_dim * 4]
    
    '''Downsampling'''
    hs = [layers.Conv2D(filters=embedding_dim, kernel_size=3, strides=1, padding='same', name='conv_in')(x)]
    for i_level in range(num_resolutions):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks):
            h = ResnetBlock(dropout=dropout, in_ch=hs[-1].shape[-1], out_ch=embedding_dim * embedding_dim_mult[i_level])(hs[-1], temb=temb)
            if h.shape[1] in attn_resolutions:
                h = AttentionBlock(in_ch=h.shape[-1])(h)
            hs.append(h)
        # Downsample
        if i_level != num_resolutions - 1:
            hs.append(Downsampling(in_ch=hs[-1].shape[-1], with_conv=resamp_with_conv)(hs[-1]))

    '''Middle'''
    h = hs[-1]
    h = ResnetBlock(dropout=dropout, in_ch=embedding_dim * embedding_dim_mult[-1], out_ch=None)(h, temb=temb)
    h = AttentionBlock(in_ch=h.shape[-1])(h)
    h = ResnetBlock(dropout=dropout, in_ch=embedding_dim * embedding_dim_mult[-1], out_ch=None)(h, temb=temb)
    
    '''Upsampling'''
    for i_level in reversed(range(num_resolutions)):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks + 1):
            h = tf.concat([h, hs.pop()], axis=-1)
            h = ResnetBlock(dropout=dropout, in_ch=h.shape[-1], out_ch=embedding_dim * embedding_dim_mult[i_level])(h, temb=temb)
            if h.shape[1] in attn_resolutions:
                h = AttentionBlock(in_ch=h.shape[-1])(h)
        # Upsample
        if i_level != 0:
            h = Upsampling(in_ch=h.shape[-1], with_conv=resamp_with_conv)(h)
            
    '''End'''
    # h = tf.nn.swish(tfa.layers.GroupNormalization(1)(h))
    h = tf.nn.swish(layers.LayerNormalization()(h))
    h = layers.Conv2D(filters=PARAMS['channel'], kernel_size=3, strides=1, padding='same', name='conv_out')(h)
    # assert h.shape == x.shape[:3] + [self.out_ch]
    
    model = K.models.Model([x, timesteps], h)

    model.summary()

    return model
#%%
# class Unet(K.models.Model):
#     def __init__(self, params, embedding_dim, out_ch, dropout=0., embedding_dim_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=(16, ), resamp_with_conv=True):
#         super(Unet, self).__init__()
        
#         self.params = params
#         self.embedding_dim = embedding_dim
#         self.out_ch = out_ch
#         self.dropout = dropout
#         self.embedding_dim_mult = embedding_dim_mult
#         self.num_res_blocks = num_res_blocks
#         self.attn_resolutions = attn_resolutions
#         self.resamp_with_conv = resamp_with_conv
#         self.num_resolutions = len(self.embedding_dim_mult)
        
#         self.nonlinearity = nonlinearity
#         self.normalize = normalize
#         self.get_timestep_embedding = get_timestep_embedding
        
#         self.channel_mult = [embedding_dim * m for m in self.embedding_dim_mult]
        
#         '''Downsampling'''
#         self.resblocks_down = [ResnetBlock(dropout=self.dropout, C=self.channel_mult[i], out_ch=self.embedding_dim * self.embedding_dim_mult[i])
#                                 for i in range(self.num_resolutions)]
#         self.attnblocks_down = AttentionBlock(C=self.attn_resolutions[0])
#         self.downsamples_down = [Downsampling(C=self.channel_mult[i], with_conv=self.resamp_with_conv)
#                                 for i in range(self.num_resolutions-1)]
        
#         '''Middle'''
#         self.resnet_middle1 = ResnetBlock(dropout=self.dropout, C=self.channel_mult[-1], out_ch=None)
#         self.attnblock_middle = AttentionBlock(C=int(self.params['data_dim'] / (2 ** (self.num_resolutions - 1))))
#         self.resnet_middle2 = ResnetBlock(dropout=self.dropout, C=self.channel_mult[-1], out_ch=None)
        
#         '''Upsampling'''
#         self.resblocks_up = [ResnetBlock(dropout=self.dropout, C=self.channel_mult[::-1][i], out_ch=self.embedding_dim * self.embedding_dim_mult[i])
#                             for i in range(self.num_resolutions)]
#         self.attnblocks_up = AttentionBlock(C=self.attn_resolutions[0])
#         self.Upsamples_up = [Upsampling(C=self.channel_mult[::-1][i], with_conv=self.resamp_with_conv)
#                             for i in range(self.num_resolutions-1)]
        
#         self.dense0 = layers.Dense(self.embedding_dim * 4, name='dense0')
#         self.dense1 = layers.Dense(self.embedding_dim * 4, name='dense1')
#         self.conv_in = layers.Conv2D(filters=self.embedding_dim, kernel_size=3, strides=1, padding='same', name='conv_in')
#         self.conv_out = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv_out')

#     def call(self, x, timesteps, **kwargs):
#         # B, _, _, _ = tf.shape(x)
#         B = self.params['batch_size']

#         '''Timestep embedding'''
#         temb = self.get_timestep_embedding(timesteps, self.embedding_dim)
#         temb = self.dense0(temb)
#         temb = self.dense1(nonlinearity(temb))
#         # assert temb.shape == [B, self.embedding_dim * 4]

#         '''Downsampling'''
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             # Residual blocks for this resolution
#             for i_block in range(self.num_res_blocks):
#                 h = self.resblocks_down[i_block](hs[-1], temb=temb)
#                 if h.shape[1] in self.attn_resolutions:
#                     h = self.attnblocks_down(h)
#                 hs.append(h)
#             # Downsample
#             if i_level != self.num_resolutions - 1:
#                 hs.append(self.downsamples_down[i_level](hs[-1]))

#         '''Middle'''
#         h = hs[-1]
#         h = self.resnet_middle1(h, temb=temb)
#         h = self.attnblock_middle(h)
#         h = self.resnet_middle2(h, temb=temb)

#         '''Upsampling'''
#         for i_level in reversed(range(self.num_resolutions)):
#             # Residual blocks for this resolution
#             for i_block in range(self.num_res_blocks + 1):
#                 h = self.resblocks_up[i_block](tf.concat([h, hs.pop()], axis=-1), temb=temb)
#                 if h.shape[1] in self.attn_resolutions:
#                     h = self.attnblocks_up(h)
#             # Upsample
#             if i_level != 0:
#                 h = self.Upsamples_up[i_block](h)

#         '''End'''
#         h = self.nonlinearity(self.normalize(h))
#         h = self.conv_out(h)
#         # h = nn.conv2d(h, name='conv_out', num_units=out_ch, init_scale=0.)
#         # assert h.shape == x.shape[:3] + [self.out_ch]
        
#         return h
#%%
# def model(x, timesteps, embedding_dim, out_ch, 
#         dropout=0., embedding_dim_mult=(1, 2, 4, 8), num_res_blocks=3, attn_resolutions=(16, ), resamp_with_conv=True):
    
#     # x = layers.Input((32, 32, 3))
#     # timesteps = layers.Input(())
#     B, _, _, _ = tf.shape(x)
#     num_resolutions = len(embedding_dim_mult)

#     '''Timestep embedding'''
#     temb = get_timestep_embedding(timesteps, embedding_dim)
#     temb = layers.Dense(embedding_dim * 4, name='dense0')(temb)
#     # temb = nn.dense(temb, name='dense0', num_units=ch * 4)
#     temb = layers.Dense(embedding_dim * 4, name='dense1')(nonlinearity(temb))
#     # temb = nn.dense(nonlinearity(temb), name='dense1', num_units=ch * 4)
#     assert temb.shape == [B, embedding_dim * 4]

#     '''Downsampling'''
#     hs = [layers.Conv2D(filters=embedding_dim, kernel_size=3, strides=1, padding='same', name='conv_in')(x)]
#     # hs = [nn.conv2d(x, name='conv_in', num_units=ch)]
#     for i_level in range(num_resolutions):
#         # Residual blocks for this resolution
#         for i_block in range(num_res_blocks):
#             h = resnet_block(hs[-1], temb=temb, out_ch=embedding_dim * embedding_dim_mult[i_level], dropout=dropout)
#             # h = resnet_block(hs[-1], name='block_{}'.format(i_block), temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
#             if tf.shape(h)[1] in attn_resolutions:
#                 h = attn_block(h)
#             hs.append(h)
#             # Downsample
#         if i_level != num_resolutions - 1:
#             hs.append(downsample(hs[-1], with_conv=resamp_with_conv))

#     '''Middle'''
#     h = hs[-1]
#     h = resnet_block(h, temb=temb, dropout=dropout)
#     # h = resnet_block(h, temb=temb, name='block_1', dropout=dropout)
#     h = attn_block(h)
#     # h = attn_block(h, name='attn_1'.format(i_block), temb=temb)
#     h = resnet_block(h, temb=temb, dropout=dropout)
#     # h = resnet_block(h, temb=temb, name='block_2', dropout=dropout)

#     '''Upsampling'''
#     for i_level in reversed(range(num_resolutions)):
#         # Residual blocks for this resolution
#         for i_block in range(num_res_blocks + 1):
#             h = resnet_block(tf.concat([h, hs.pop()], axis=-1), temb=temb, out_ch=embedding_dim * embedding_dim_mult[i_level], dropout=dropout)
#             if tf.shape(h)[1] in attn_resolutions:
#                 h = attn_block(h)
#         # Upsample
#         if i_level != 0:
#             h = upsample(h, with_conv=resamp_with_conv)

#     '''End'''
#     h = nonlinearity(normalize(h))
#     h = layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding='same', name='conv_out')(h)
#     # h = nn.conv2d(h, name='conv_out', num_units=out_ch, init_scale=0.)
#     assert h.shape == x.shape[:3] + [out_ch]
    
#     return h
#     # return K.models.Model([x, timesteps], h)
#%%