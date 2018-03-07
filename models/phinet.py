from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Activation
from keras.layers.merge import Concatenate, add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

def phinet(input_shape, n_inputs=1, learning_rate=0.001, load_weights=False, weights=""):

    inputs = Input(shape=input_shape)
    reshape = Reshape(input_shape+(1,), input_shape=input_shape)(inputs)

    x = Conv3D(16, (3,3,3), strides=(2,2,2), padding='same')(reshape)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), padding='same')(x)

    x = Conv3D(16, (3,3,3), strides=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    y = Activation('relu')(x)
    x = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same')(y)
    x = BatchNormalization()(y)
    x = add([x, y])
    x = Activation('relu')(x)

    for _ in range(10):
        x = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same')(x)
        x = BatchNormalization()(x)
        y = Activation('relu')(x)
        x = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same')(y)
        x = BatchNormalization()(x)
        x = add([x, y])
        x = Activation('relu')(x)

    # this block will pool a handful of times to get the "big picture" 
    y = AveragePooling3D(pool_size=(7,7,7), strides=(4,4,4), padding='same')(reshape)
    y = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), padding='same')(y)

    # this layer will preserve original signal
    z = Conv3D(16, (3,3,3), strides=(2,2,2), padding='same')(reshape) 
    z = Conv3D(16, (3,3,3), strides=(2,2,2), padding='same')(z) 
    z = Conv3D(16, (3,3,3), strides=(2,2,2), padding='same')(z)

    x = Concatenate(axis=4)([x, y, z])

    # global avg pooling before FC
    x = AveragePooling3D(pool_size=(2,2,2), strides=(1,1,1), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    pred = Dense(n_inputs, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=pred)

    if load_weights:
        model.load_weights(weights)

    model.compile(optimizer=Adam(lr=learning_rate), \
                loss='binary_crossentropy',metrics=['accuracy'])
    return model
