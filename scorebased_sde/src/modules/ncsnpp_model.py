#%%
# FIXME: layer normalization -> group normalization
#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class Upsampling(layers.Layer):
    def __init__(self, with_conv, in_ch=None):
        super(Upsampling, self).__init__()
        
        self.with_conv = with_conv
        if self.with_conv:
            self.in_ch = in_ch
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
    def __init__(self, with_conv, in_ch=None):
        super(Downsampling, self).__init__()
        
        self.with_conv = with_conv
        if self.with_conv:
            self.in_ch = in_ch
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
# positional 
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
# FIXME: Add Finite Impulse Response (FIR) Up or Down sampling 
class ResnetBlock(layers.Layer):
    def __init__(self, dropout, skip_rescale, in_ch, out_ch=None, up=False, down=False, scale_num=1):
        """[summary]

        Args:
            dropout ([type]): [description]
            skip_rescale ([type]): [description]
            in_ch ([type]): [description]
            out_ch ([type], optional): [description]. Defaults to None.
            up (bool, optional): [description]. Defaults to False.
            down (bool, optional): [description]. Defaults to False.
            scale_num (int, optional): [description]. Defaults to 1.
        """
        super(ResnetBlock, self).__init__()
        
        self.up = up
        self.down = down
        self.dropout = dropout
        self.skip_rescale = skip_rescale
        self.in_ch = in_ch
        self.out_ch = out_ch
        if self.out_ch is None:
            self.out_ch = self.in_ch
        self.scale_num = scale_num
        
        if self.up:
            self.shortcut = layers.Conv2DTranspose(filters=self.out_ch, kernel_size=1, strides=2 ** self.scale_num, padding='same', name='conv_shortcut')    
        elif self.down:
            self.shortcut = layers.Conv2D(filters=self.out_ch, kernel_size=1, strides=2 ** self.scale_num, padding='same', name='conv_shortcut')
        elif self.out_ch != self.in_ch:
            self.shortcut = layers.Conv2D(filters=self.out_ch, kernel_size=1, strides=1, padding='same', name='conv_shortcut')
        
        self.normalize1 = layers.LayerNormalization()
        self.normalize2 = layers.LayerNormalization()
        self.conv1 = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv1')
        self.temb_proj = layers.Dense(self.out_ch, name='temb_proj')
        self.dropout_layer = layers.Dropout(rate=self.dropout)
        self.conv2 = layers.Conv2D(filters=self.out_ch, kernel_size=3, strides=1, padding='same', name='conv2')

    def call(self, x, temb, **kwargs):
        B, H, W, C = x.shape
        h = tf.nn.swish(self.normalize1(x))
        
        if self.up:
            for _ in range(self.scale_num):
                B, H, W, C = h.shape
                h = tf.image.resize(h, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif self.down:
            for _ in range(self.scale_num):
                B, H, W, C = h.shape
                h = tf.image.resize(h, size=[H // 2, W // 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
        h = self.conv1(h)

        # add timestep embedding
        h += self.temb_proj(tf.nn.swish(temb))[:, None, None, :]

        h = tf.nn.swish(self.normalize2(h))
        h = self.dropout_layer(h)
        h = self.conv2(h)
        
        if self.up:
            x = self.shortcut(x)
        elif self.down:
            x = self.shortcut(x)
        elif self.out_ch != self.in_ch:
            x = self.shortcut(x)
        
        # assert x.shape == h.shape
        if self.skip_rescale:
            return (x + h) / np.sqrt(2.)
        else:
            return x + h    
#%%
class AttentionBlock(layers.Layer):
    def __init__(self, in_ch):
        super(AttentionBlock, self).__init__()
        
        self.in_ch = in_ch
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
def build_unet(PARAMS):
    
    embedding_dim = PARAMS['embedding_dim']
    skip_rescale = PARAMS['skip_rescale']
    scale_num = PARAMS['scale_num']
    dropout = PARAMS['dropout']
    embedding_dim_mult = PARAMS['embedding_dim_mult']
    num_res_blocks = PARAMS['num_res_blocks']
    attn_resolutions = PARAMS['attn_resolutions']
    progressive_input = PARAMS['progressive_input']
    progressive_output = PARAMS['progressive_output']
    progressive_combine = PARAMS['progressive_combine']
    # attention_type = PARAMS['attention_type']
    
    x = layers.Input((PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel']))
    timesteps = layers.Input(())

    num_resolutions = len(embedding_dim_mult)

    '''Timestep embedding'''
    temb = get_timestep_embedding(timesteps, embedding_dim)
    temb = layers.Dense(embedding_dim * 4, name='dense0')(temb)
    temb = layers.Dense(embedding_dim * 4, name='dense1')(tf.nn.swish(temb))
    # assert temb.shape == [B, self.embedding_dim * 4]
    
    '''Downsampling'''
    if progressive_input is not None:
        input_pyramid = x
        
    hs = [layers.Conv2D(filters=embedding_dim, kernel_size=3, strides=1, padding='same', name='conv_in')(x)]
    for i_level in range(num_resolutions):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks):
            h = ResnetBlock(dropout=dropout, skip_rescale=skip_rescale, 
                            in_ch=hs[-1].shape[-1], out_ch=embedding_dim * embedding_dim_mult[i_level], 
                            up=False, down=False,
                            scale_num=scale_num)(hs[-1], temb=temb)
            
            if h.shape[1] in attn_resolutions:
                h = AttentionBlock(in_ch=h.shape[-1])(h)
            hs.append(h)
        # Downsample
        if i_level != num_resolutions - 1:
            h = ResnetBlock(dropout=dropout, skip_rescale=skip_rescale, 
                            in_ch=hs[-1].shape[-1], out_ch=embedding_dim * embedding_dim_mult[i_level], 
                            up=False, down=True,
                            scale_num=scale_num)(hs[-1], temb=temb)
            
            if progressive_input == 'input_skip':
                input_pyramid = Downsampling(with_conv=False, in_ch=None)(input_pyramid)
                if progressive_combine == 'concat':
                    h = layers.Concatenate(axis=-1)([input_pyramid, h])
                elif progressive_combine == 'sum':
                    h = input_pyramid = h

            elif progressive_input == 'residual':
                input_pyramid = Downsampling(with_conv=True, in_ch=h.shape[-1])(input_pyramid)
                if skip_rescale:
                    input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                else:
                    input_pyramid = input_pyramid + h
                h = input_pyramid
            
            hs.append(h)

    '''Middle'''
    h = hs[-1]
    h = ResnetBlock(dropout=dropout, skip_rescale=skip_rescale, 
                    in_ch=embedding_dim * embedding_dim_mult[-1], out_ch=None, 
                    up=False, down=False,
                    scale_num=scale_num)(h, temb=temb)
    h = AttentionBlock(in_ch=h.shape[-1])(h)
    h = ResnetBlock(dropout=dropout, skip_rescale=skip_rescale, 
                    in_ch=embedding_dim * embedding_dim_mult[-1], out_ch=None, 
                    up=False, down=False,
                    scale_num=scale_num)(h, temb=temb)
    
    '''Upsampling'''
    for i_level in reversed(range(num_resolutions)):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks + 1):
            h = tf.concat([h, hs.pop()], axis=-1)
            h = ResnetBlock(dropout=dropout, skip_rescale=skip_rescale, 
                            in_ch=h.shape[-1], out_ch=embedding_dim * embedding_dim_mult[i_level], 
                            up=False, down=False,
                            scale_num=scale_num)(h, temb=temb)
            
            if h.shape[1] in attn_resolutions:
                h = AttentionBlock(in_ch=h.shape[-1])(h)
                
        if progressive_output is not None:
            if i_level == num_resolutions - 1:
                if progressive_output == 'output_skip':
                    pyramid = tf.nn.swish(layers.LayerNormalization()(h))
                    pyramid = layers.Conv2D(filters=x.shape[-1], kernel_size=3, strides=1, padding='same')(pyramid)
                    
                elif progressive_output == 'residual':
                    pyramid = tf.nn.swish(layers.LayerNormalization()(h))
                    pyramid = layers.Conv2D(filters=h.shape[-1], kernel_size=3, strides=1, padding='same')(pyramid)
            else:
                if progressive_output == 'output_skip':
                    pyramid = Upsampling(with_conv=False, in_ch=None)(pyramid)
                    pyramid = tf.nn.swish(layers.LayerNormalization()(h))
                    pyramid = pyramid + layers.Conv2D(filters=x.shape[-1], kernel_size=3, strides=1, padding='same')(pyramid)
                    
                elif progressive_output == 'residual':
                    pyramid = Upsampling(with_conv=True, in_ch=h.shape[-1])(pyramid)
                    if skip_rescale:
                        pyramid = (pyramid + h) / np.sqrt(2.)
                    else:
                        pyramid = pyramid + h
                    h = pyramid
                        
        # Upsample
        if i_level != 0:
            h = ResnetBlock(dropout=dropout, skip_rescale=skip_rescale, 
                            in_ch=h.shape[-1], out_ch=None, 
                            up=True, down=False,
                            scale_num=scale_num)(h, temb=temb)
            
    '''End'''
    if progressive_output == 'output_skip':
        h = pyramid
    else:
        h = tf.nn.swish(layers.LayerNormalization()(h))
        h = layers.Conv2D(filters=PARAMS['channel'], kernel_size=3, strides=1, padding='same', name='conv_out')(h)
        
    # assert h.shape == x.shape[:3] + [self.out_ch]
    
    model = K.models.Model([x, timesteps], h)

    model.summary()

    return model
#%%