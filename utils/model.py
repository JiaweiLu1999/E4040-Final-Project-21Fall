import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# define the model
class SSI_RES_UNET():
    def __init__(self, in_ch=28, out_ch=28):
        
        self.in_ch = in_ch                        # define input channels
        self.out_ch = out_ch                      # define output channels
        
        self.n_resblocks_part1 = 4                # number of res blocks for first part
        self.n_resblocks_part2 = 8                # number of res blocks for second part
        self.n_resblocks_part3 = 4                # number of res blocks for third part
        self.n_feats = 64                         # number of features(channels) through main structure of the model
        self.kernel_size = 3                      # conv kernel size
        self.scale = 2                            # upsample scale
        self.data_format = 'channels_first'       # data format
        self.data_format_nn = 'NCHW'              # tf.nn receives another type of data format
        
    def _ResBlock(self, input_tensor, n_feats, kernel_size, bias=True, bn=False, res_scale=1, name=None):
        
        x = layers.Conv2D(n_feats, kernel_size, padding='same', 
                          use_bias=bias, name=name+'_conv1', data_format = self.data_format)(input_tensor)
        if bn:
            x = layers.BatchNormalization(1, name=name+'_bn1')(x)
        x = layers.ReLU(name=name+'_relu1')(x)
        x = layers.Conv2D(n_feats, kernel_size, padding='same', 
                          use_bias=bias, name=name+'_conv2', data_format = self.data_format)(x)
        if bn:
            x = layers.BatchNormalization(1, name=name+'_bn2')(x)
        
        return res_scale * x + input_tensor
        
        
    def create_model(self, input_shape):
        
        ## Input layer
        input_tensor = layers.Input(shape = input_shape, name='input_layer')
        
        ## head part
        x = layers.Conv2D(self.n_feats, self.kernel_size, padding='same', name='input_conv', data_format = self.data_format)(input_tensor)
        x = layers.ZeroPadding2D(padding=1, data_format ='channels_first')(x)
        x = layers.Conv2D(self.n_feats, self.kernel_size, strides=2, padding='valid', name='downsampler_1', data_format = self.data_format)(x)
        
        head_temp = x
        
        ## body part
        # resblocks part1
        for i in range(self.n_resblocks_part1):
            x = self._ResBlock(x, self.n_feats, self.kernel_size, name='resblock_part1_' + str(i+1))
        # downsample
        x = layers.ZeroPadding2D(padding=1, data_format ='channels_first')(x)
        x = layers.Conv2D(self.n_feats, self.kernel_size, strides=2, padding='valid', name='downsampler_2', data_format = self.data_format)(x)
        
        # resblocks part2
        for i in range(self.n_resblocks_part2):
            x = self._ResBlock(x, self.n_feats, self.kernel_size, name='resblock_part2_' + str(i+1))
        
        # upsample - using pixelshuffle method instead of conv2dtranspose
        x = layers.Conv2D(self.n_feats * 4, self.kernel_size, padding='same', name='upsampler_1', data_format = self.data_format)(x)
        x = tf.nn.depth_to_space(x, self.scale, data_format = self.data_format_nn)
        
        # resblocks part3
        for i in range(self.n_resblocks_part3):
            x = self._ResBlock(x, self.n_feats, self.kernel_size, name='resblock_part3_' + str(i+1))
        
        x = layers.Conv2D(self.n_feats, self.kernel_size, padding='same', name='extra_conv', data_format = self.data_format)(x)
        
        x = x + head_temp
        ## tail part
        # upsample 
        x = layers.Conv2D(self.n_feats * 4, self.kernel_size, padding='same', name='upsampler_2', data_format = self.data_format)(x)
        x = tf.nn.depth_to_space(x, self.scale, data_format = self.data_format_nn)
        
        # output conv
        x = layers.Conv2D(self.out_ch, self.kernel_size, padding='same', name='output_conv', data_format = self.data_format)(x)
        
        # assemble a model
        model = keras.Model(inputs=input_tensor, outputs=x, name='ssi_res_unet')
        
        return model