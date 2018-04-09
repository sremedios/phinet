from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D,\
                         GlobalMaxPooling3D, AveragePooling3D, Dense, Flatten,\
                         Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Activation
from keras.layers.merge import Concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K

def phinet(n_classes, n_channels=1, learning_rate=1e-3):

    #inputs = Input(shape=input_shape)
    #reshape = Reshape(input_shape+(1,), input_shape=input_shape)(inputs)

    inputs = Input(shape=(None,None,None,n_channels))

    x = Conv3D(8, (3,3,3), strides=(2,2,2), padding='same')(inputs)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), padding='same')(x)

    for _ in range(3):
        x = Conv3D(8, (3,3,3), strides=(1,1,1), padding='same')(x)
        x = BatchNormalization()(x)
        y = Activation('relu')(x)
        x = Conv3D(8, (3,3,3), strides=(1,1,1), padding='same')(y)
        x = BatchNormalization()(x)
        x = add([x, y])
        x = Activation('relu')(x)

    # this block will pool a handful of times to get the "big picture" 
    y = AveragePooling3D(pool_size=(3,3,3), strides=(2,2,2), padding='same')(inputs)
    y = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), padding='same')(y)
    y = Conv3D(8, (3,3,3), strides=(1,1,1), padding='same')(y)

    # this layer will preserve original signal
    z = Conv3D(8, (3,3,3), strides=(2,2,2), padding='same')(inputs)
    z = Conv3D(8, (3,3,3), strides=(2,2,2), padding='same')(z)
    z = Conv3D(8, (3,3,3), strides=(1,1,1), padding='same')(z)

    x = Concatenate(axis=4)([x, y, z])

    # global avg pooling before FC
    x = GlobalAveragePooling3D()(x)
    x = Dense(n_classes)(x)

    pred = Activation('softmax')(x)
    
    model = Model(inputs=inputs, outputs=pred)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    return model
