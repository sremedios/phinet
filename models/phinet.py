from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    UpSampling2D,
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
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            kernel_regularizer=regularizers.l2(1e-2),
            bias_regularizer=regularizers.l2(1e-2),
            strides=1,
            padding='same'
        )(block)
        y = Activation('relu')(x)
        x = Conv2D(
            filters=num_filters,
            kernel_size=3,
            kernel_regularizer=regularizers.l2(1e-2),
            bias_regularizer=regularizers.l2(1e-2),
            strides=1,
            padding='same'
        )(x)
        x = add([x, y])
        x = Activation('relu')(x)
        block = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    return block

def pooling_block(prev_layer, repetitions, num_filters):
    block = prev_layer
    for i in range(repetitions):
        block = MaxPooling2D(
            pool_size=5,
            strides=2,
            padding='same'
        )(block)


    block = Conv2D(
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
        block = Conv2D(
            filters=num_filters * (2**i),
            kernel_size=3,
            kernel_regularizer=regularizers.l2(1e-2),
            bias_regularizer=regularizers.l2(1e-2),
            strides=2,
            padding='same'
        )(block)

    return block



def phinet(num_classes, ds):
    inputs = Input(shape=(None, None, 1))
    x = Conv2D(
        filters=64//ds,
        kernel_size=7,
        kernel_regularizer=regularizers.l2(1e-2),
        bias_regularizer=regularizers.l2(1e-2),
        strides=2,
        padding='same',
        activation='relu',
    )(inputs)
    block_0 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

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
    x = GlobalMaxPooling2D()(x)
    outputs = Dense(num_classes, dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=outputs)

    return model
