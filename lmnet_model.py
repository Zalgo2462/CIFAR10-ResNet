from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(prev, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    prev = tf.layers.batch_normalization(
        inputs=prev, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    prev = tf.nn.relu(prev)
    return prev


def building_block(prev2, prev, filters, is_training, projection_shortcut_2, projection_shortcut_1, strides,
                   data_format):
    """Standard building block for residual networks with BN before convolutions.

    Args:
      prev2: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      prev: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      is_training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut_2: The function to use for projection shortcuts (typically
        a 1x1 convolution when downsampling the input).
      projection_shortcut_1: The function to use for projection shortcuts (typically
        a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      (prev, The output tensor of the block.)
    """

    prev_link = prev
    pre_activation_1 = batch_norm_relu(prev, is_training, data_format)

    if projection_shortcut_1 is not None:
        prev = projection_shortcut_1(pre_activation_1)

    if projection_shortcut_2 is not None:
        pre_activation_2 = batch_norm_relu(prev2, is_training, data_format)
        prev2 = projection_shortcut_2(pre_activation_2)

    k = tf.get_variable("k", shape=[], initializer=tf.initializers.random_uniform(-0.1, 0))
    prev = (1-k) * prev
    prev2 = k * prev2

    conv = tf.layers.conv2d(
        inputs=pre_activation_1, filters=filters, kernel_size=3, strides=strides,
        padding='SAME', use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format
    )

    conv = batch_norm_relu(conv, is_training, data_format)

    conv = tf.layers.conv2d(
        inputs=conv, filters=filters, kernel_size=3, strides=1,
        padding='SAME', use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format
    )

    return prev_link, prev2 + prev + conv


def block_layer(prev2, prev, filters, blocks, strides, is_training, name,
                data_format):
    """Creates one layer of blocks for the ResNet model.

    Args:
      prev2: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      prev: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      is_training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      (prev, The output tensor of the block layer.)
    """

    with tf.variable_scope(name_or_scope=None, default_name="LMNet-Layer-Projector"):
        def projection_shortcut(inputs):
            out = tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=1, strides=strides,
                padding='SAME', use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
                data_format=data_format, name="Projector", reuse=tf.AUTO_REUSE)
            return out

        with tf.variable_scope(name_or_scope=None, default_name="LMNet-Layer"):
            prev2, prev = building_block(prev2, prev, filters, is_training,
                                         projection_shortcut, projection_shortcut,
                                         strides, data_format)

        with tf.variable_scope(name_or_scope=None, default_name="LMNet-Layer"):
            prev2, prev = building_block(prev2, prev, filters, is_training,
                                         projection_shortcut, None,
                                         1, data_format)

    for _ in range(2, blocks):
        with tf.variable_scope(name_or_scope=None, default_name="LMNet-Layer"):
            prev2, prev = building_block(prev2, prev, filters, is_training, None, None, 1, data_format)

    return prev2, tf.identity(prev, name)


def cifar10_lmnet_v2_generator(resnet_size, num_classes, data_format=None):
    """Generator for CIFAR-10 ResNet v2 models.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      num_classes: The number of possible classes for image classification.
      data_format: The input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.

    Returns:
      The model function that takes in `prev` and `is_training` and
      returns the output tensor of the ResNet model.

    Raises:
      ValueError: If `resnet_size` is invalid.
    """
    if resnet_size % 6 != 2:
        raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    if data_format is None:
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    # Asumes prev is NHWC format
    def model(prev, is_training):
        """Constructs the ResNet model given the prev."""
        if data_format == 'channels_first':
            # Convert the prev from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            prev = tf.transpose(prev, [0, 3, 1, 2])

        prev = tf.layers.conv2d(
            inputs=prev, filters=16, kernel_size=3, strides=1,
            padding='SAME', use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)
        prev = tf.identity(prev, 'initial_conv')
        prev2 = prev

        prev2, prev = block_layer(
            prev2=prev2, prev=prev, filters=16, blocks=num_blocks,
            strides=1, is_training=is_training, name='block_layer1',
            data_format=data_format)
        prev2, prev = block_layer(
            prev2=prev2, prev=prev, filters=32, blocks=num_blocks,
            strides=2, is_training=is_training, name='block_layer2',
            data_format=data_format)
        prev2, prev = block_layer(
            prev2=prev2, prev=prev, filters=64, blocks=num_blocks,
            strides=2, is_training=is_training, name='block_layer3',
            data_format=data_format)

        prev = batch_norm_relu(prev, is_training, data_format)
        prev = tf.layers.average_pooling2d(
            inputs=prev, pool_size=8, strides=1, padding='VALID',
            data_format=data_format)
        prev = tf.identity(prev, 'final_avg_pool')
        prev = tf.reshape(prev, [-1, 64])
        prev = tf.layers.dense(inputs=prev, units=num_classes)
        prev = tf.identity(prev, 'final_dense')
        return prev

    return model
