import tensorflow as tf

def slice_image_example(X, Y,):
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

    return tf.train.Example(features=tf.train.Features(feature=feature))

def volume_image_example(X, Y, num_instances):
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
    feature['num_instances'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[num_instances]))

    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_into_slice(record, instance_size, num_labels):
    features = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'Y': tf.io.FixedLenFeature([], tf.string),
    }

    image_features = tf.io.parse_single_example(record, features=features)

    x = tf.io.decode_raw(image_features.get('X'), tf.float32)
    x = tf.reshape(x, (*instance_size, 1))
    #x = tf.cast(x, tf.float32)

    y = tf.io.decode_raw(image_features.get('Y'), tf.float32)
    y = tf.reshape(y, (num_labels, ))
    #y = tf.cast(y, tf.float32)

    return x, y


def parse_into_volume(record, instance_size, num_labels):
    features = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'Y': tf.io.FixedLenFeature([], tf.string),
        'num_instances': tf.io.FixedLenFeature([], tf.int64)
    }

    image_features = tf.io.parse_single_example(record, features=features)


    x = tf.io.decode_raw(image_features.get('X'), tf.float32)
    x = tf.reshape(x, (image_features.get('num_instances'), *instance_size, 1))
    #x = tf.cast(x, tf.float32)

    y = tf.io.decode_raw(image_features.get('Y'), tf.float32)
    y = tf.reshape(y, (num_labels, ))
    #y = tf.cast(y, tf.float32)

    return x, y
