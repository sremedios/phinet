from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    AveragePooling3D,
    GlobalAveragePooling3D,
    GlobalMaxPooling3D,
    UpSampling3D,
    BatchNormalization,
    add,
    Dropout,
    Activation,
    Dense,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def residual_block(prev_layer, repetitions, num_filters):
    block = prev_layer
    for i in range(repetitions):
        x = Conv3D(
            filters=num_filters,
            kernel_size=3,
            kernel_regularizer=regularizers.l2(1e-2),
            bias_regularizer=regularizers.l2(1e-2),
            strides=1,
            padding='same'
        )(block)
        y = Activation('relu')(x)
        x = Conv3D(
            filters=num_filters,
            kernel_size=3,
            kernel_regularizer=regularizers.l2(1e-2),
            bias_regularizer=regularizers.l2(1e-2),
            strides=1,
            padding='same'
        )(x)
        x = add([x, y])
        x = Activation('relu')(x)
        block = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    return block

def pooling_block(prev_layer, repetitions, num_filters):
    block = prev_layer
    for i in range(repetitions):
        block = MaxPooling3D(
            pool_size=5,
            strides=2,
            padding='same'
        )(block)


    block = Conv3D(
        filters=num_filters,
        kernel_size=3,
        kernel_regularizer=regularizers.l2(1e-2),
        bias_regularizer=regularizers.l2(1e-2),
        strides=1,
        padding='same'
    )(block)

    block = Activation('relu')(block)

    return block

def linear_block(prev_layer, repetitions, num_filters):
    block = prev_layer
    for i in range(repetitions):
        block = Conv3D(
            filters=num_filters * (2**i),
            kernel_size=3,
            kernel_regularizer=regularizers.l2(1e-2),
            bias_regularizer=regularizers.l2(1e-2),
            strides=2,
            padding='same'
        )(block)

    return block



def phinet(num_classes, ds):
    inputs = Input(shape=(None, None, None, 1))
    x = Conv3D(
        filters=64//ds,
        kernel_size=7,
        kernel_regularizer=regularizers.l2(1e-2),
        bias_regularizer=regularizers.l2(1e-2),
        strides=2,
        padding='same',
        activation='relu',
    )(inputs)
    block_0 = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    repetitions = 2
    num_filters = 64//ds

    # Residual branch
    x = residual_block(
        prev_layer=block_0, 
        repetitions=repetitions, 
        num_filters=num_filters)

    # Pooling branch
    y = pooling_block(
        prev_layer=block_0, 
        repetitions=repetitions, 
        num_filters=num_filters
    )

    # Linear branch
    z = linear_block(
        prev_layer=block_0, 
        repetitions=repetitions, 
        num_filters=num_filters
    )

    x = concatenate([x, y, z], axis=-1)

    # global avg pooling before FC
    x = GlobalMaxPooling3D()(x)
    outputs = Dense(num_classes, dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    return model
