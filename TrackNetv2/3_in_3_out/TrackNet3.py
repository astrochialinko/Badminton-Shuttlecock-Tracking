from keras.models import *
from keras.layers import *
from keras.activations import *

def TrackNet3( input_height, input_width ): #input_height = 288, input_width = 512

    #imgs_input = Input(shape=(9,input_height,input_width))
    imgs_input = Input(shape=(input_height,input_width,9))

    #Layer1
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last' )(imgs_input)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer2
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last' )(x)
    x = ( Activation('relu'))(x)
    x1 = ( BatchNormalization(axis=-2))(x)

    #Layer3
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last' )(x1)

    #Layer4
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last' )(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer5
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last' )(x)
    x = ( Activation('relu'))(x)
    x2 = ( BatchNormalization(axis=-2))(x)
    #x2 = (Dropout(0.5))(x2) 

    #Layer6
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last' )(x2)

    #Layer7
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last' )(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer8
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last' )(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer9
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last' )(x)
    x = ( Activation('relu'))(x)
    x3 = ( BatchNormalization(axis=-2))(x)
    #x3 = (Dropout(0.5))(x3)

    #Layer10
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last' )(x3)

    #Layer11
    x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer12
    x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer13
    x = ( Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)
    #x = (Dropout(0.5))(x)

    #Layer14
    #x = UpSampling2D( (2,2), data_format='channels_last')(x)
    x = concatenate( [UpSampling2D( (2,2), data_format='channels_last')(x), x3], axis=3)

    #Layer15
    x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer16
    x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer17
    x = ( Conv2D( 256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_last'))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)
    
    #Layer18
    #x = UpSampling2D( (2,2), data_format='channels_last')(x)
    x = concatenate( [UpSampling2D( (2,2), data_format='channels_last')(x), x2], axis=3)

    #Layer19
    x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_last' ))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer20
    x = ( Conv2D( 128 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_last' ))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer21
    #x = UpSampling2D( (2,2), data_format='channels_last')(x)
    x = concatenate( [UpSampling2D( (2,2), data_format='channels_last')(x), x1], axis=3)

    #Layer22
    x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_last' ))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer23
    x = ( Conv2D( 64 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_last' ))(x)
    x = ( Activation('relu'))(x)
    x = ( BatchNormalization(axis=-2))(x)

    #Layer24
    x =  Conv2D( 3 , (1, 1) , kernel_initializer='random_uniform', padding='same', data_format='channels_last' )(x)
    x = ( Activation('sigmoid'))(x)
        

    o_shape = Model(imgs_input , x ).output_shape

    #print ("layer24 output shape:", o_shape[1],o_shape[2],o_shape[3])
    #Layer24 output shape: (3, 288, 512)

    OutputHeight = o_shape[1]
    OutputWidth = o_shape[2]

    output = x

    model = Model( imgs_input , output)
    #model input unit:9*288*512, output unit:3*288*512
    model.outputWidth = OutputWidth
    model.outputHeight = OutputHeight

    #Show model's details
    #model.summary()

    return model




