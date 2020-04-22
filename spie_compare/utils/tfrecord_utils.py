import tensorflow as tf

def volume_image_example(X, Y, x_shape, y_shape):
    '''
    Creates an image example.
    X: numpy ndarray: the input image data
    Y: numpy ndarray: corresponding label information, can be an ndarray, integer, float, etc

    Returns: tf.train.Example with the following features:
        dim0, dim1, dim2, ..., dimN, X, Y, X_dtype, Y_dtype

    '''
    feature = {}
    feature['X'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[X.tobytes()]))
    feature['Y'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[Y.tobytes()]))
    feature['x_shape'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[x_shape.tobytes()]))
    feature['y_shape'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[y_shape.tobytes()]))

    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_into_volume(record):
    features = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'Y': tf.io.FixedLenFeature([], tf.string),
        'x_shape': tf.io.FixedLenFeature([], tf.string),
        'y_shape': tf.io.FixedLenFeature([], tf.string),
    }

    image_features = tf.io.parse_single_example(record, features=features)

    x_shape = tf.io.decode_raw(image_features.get('x_shape'), tf.uint16)
    x_shape = tf.cast(x_shape, tf.int32)
    y_shape = tf.io.decode_raw(image_features.get('y_shape'), tf.uint16)
    y_shape = tf.cast(y_shape, tf.int32)

    x = tf.io.decode_raw(image_features.get('X'), tf.float32)
    x = tf.reshape(x, x_shape)

    y = tf.io.decode_raw(image_features.get('Y'), tf.float32)
    y = tf.reshape(y, y_shape)

    return x[..., tf.newaxis], y
