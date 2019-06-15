"""
code borrow from:

https://github.com/mnicnc404/CartoonGan-tensorflow
https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/shufflenet_v2.py
https://github.com/zengarden/light_head_rcnn
https://github.com/geonseoks/Light_head_R_CNN_xception
https://github.com/Stick-To/light-head-rcnn-tensorflow

"""

import tensorflow as tf #todo: remove this line
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, InputSpec

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, Activation

CEM_FILTER=245

@tf.function
def channle_shuffle(inputs, group):
    """Shuffle the channel
    Args:
        inputs: 4D Tensor
        group: int, number of groups
    Returns:
        Shuffled 4D Tensor
    """
    #in_shape = inputs.get_shape().as_list()
    h, w, in_channel  = K.int_shape(inputs)[1:]
    #h, w, in_channel = in_shape[1:]
    assert(in_channel % group == 0)
    l = K.reshape(inputs, [-1, h, w, in_channel // group, group])
    l = K.permute_dimensions(l, [0, 1, 2, 4, 3])
    l = K.reshape(l, [-1, h, w, in_channel])

    return l


class Conv2D_BN(Model):
    """Conv2D -> BN """
    def __init__(self, channel, kernel_size=1, stride=1):
        super(Conv2D_BN, self).__init__()

        self.conv = Conv2D(channel, kernel_size, strides=stride,
                            padding="SAME", use_bias=False)
        self.bn = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)

    def build(self, input_shape):
        super(Conv2D_BN, self).build(input_shape)
    @tf.function
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)

        return x


class Conv2D_BN_ReLU(Model):
    """Conv2D -> BN -> ReLU"""
    def __init__(self, channel, kernel_size=1, stride=1):
        super(Conv2D_BN_ReLU, self).__init__(name="Conv2D_BN_ReLU")

        self.conv = Conv2D(channel, kernel_size, strides=stride,
                            padding="SAME", use_bias=False)
        self.bn_ = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = Activation("relu")
        self.model = tf.keras.models.Sequential()
        self.model.add(self.conv)
        self.model.add(self.bn_ )
        self.model.add(self.relu)



    def build(self, input_shape):
        super(Conv2D_BN_ReLU, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        #x = self.conv(inputs)
        #x = self.bn_(x, training=training)
        #x = self.relu(x)
        x=self.model(inputs, training=training)
        return x


class DepthwiseConv2D_BN(Model):
    """DepthwiseConv2D -> BN"""
    def __init__(self, kernel_size=3, stride=1):
        super(DepthwiseConv2D_BN, self).__init__()

        self.dconv = DepthwiseConv2D(kernel_size, strides=stride,
                                     depth_multiplier=1,
                                     padding="SAME", use_bias=False)
        self.bn = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)
    @tf.function
    def call(self, inputs, training=False):
        x = self.dconv(inputs)
        x = self.bn(x, training=training)
        return x

class DepthwiseConv2D_BN_POINT(Model):
    def __init__(self, kernel_size=3, stride=1, out_channel=256):
        super(DepthwiseConv2D_BN_POINT, self).__init__()
        self.dconv = DepthwiseConv2D(kernel_size, strides=stride,
                                     depth_multiplier=1,
                                     padding="SAME", use_bias=False)
        self.bn = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)
        self.conv = Conv2D(out_channel, 1, strides=1,
                            padding="SAME", use_bias=True)
    @tf.function
    def call(self, inputs, training=False):
        x = self.dconv(inputs)
        x = self.bn(x, training=training)
        x = self.conv(x)
        return x




class ShufflenetUnit1(Model):
    def __init__(self, out_channel):
        """The unit of shufflenetv2 for stride=1
        Args:
            out_channel: int, number of channels
        """
        super(ShufflenetUnit1, self).__init__()

        assert out_channel % 2 == 0
        self.out_channel = out_channel

        self.conv1_bn_relu = Conv2D_BN_ReLU(out_channel // 2, 1, 1)
        self.dconv_bn = DepthwiseConv2D_BN(5, 1)
        self.conv2_bn_relu = Conv2D_BN_ReLU(out_channel // 2, 1, 1)

    def build(self, input_shape):
        super(ShufflenetUnit1, self).build(input_shape)
    @tf.function
    def call(self, inputs, training=False):
        #print(K.int_shape(inputs))
        # split the channel
        shortcut, x = tf.split(inputs, 2, axis=3)

        x = self.conv1_bn_relu(x, training=training)
        x = self.dconv_bn(x, training=training)
        x = self.conv2_bn_relu(x, training=training)

        x = tf.concat([shortcut, x], axis=3)
        #print(K.int_shape(x))
        x = channle_shuffle(x, 2)
        return x

class ShufflenetUnit2(tf.keras.Model):
    """The unit of shufflenetv2 for stride=2"""
    def __init__(self, in_channel, out_channel):
        super(ShufflenetUnit2, self).__init__()

        assert out_channel % 2 == 0
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1_bn_relu = Conv2D_BN_ReLU(out_channel // 2, 1, 1)
        self.dconv_bn = DepthwiseConv2D_BN(5, 2)
        self.conv2_bn = Conv2D_BN(out_channel - in_channel, 1, 1)

        # for shortcut
        self.shortcut_dconv_bn = DepthwiseConv2D_BN(3, 2)
        self.shortcut_conv_bn = Conv2D_BN(in_channel, 1, 1)
    @tf.function
    def call(self, inputs, training=False):
        shortcut, x = inputs, inputs

        x = self.conv1_bn_relu(x, training=training)
        x = self.dconv_bn(x, training=training)
        x = self.conv2_bn(x, training=training)

        shortcut = self.shortcut_dconv_bn(shortcut, training=training)
        shortcut = self.shortcut_conv_bn(shortcut, training=training)

        x = tf.concat([shortcut, x], axis=3)
        x = ReLU()(x)
        x = channle_shuffle(x, 2)
        return x

class ShufflenetStage(Model):
    """The stage of shufflenet"""
    def __init__(self, in_channel, out_channel, num_blocks):
        super(ShufflenetStage, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.ops = []
        for i in range(num_blocks):
            if i == 0:
                op = ShufflenetUnit2(in_channel, out_channel)
            else:
                op = ShufflenetUnit1(out_channel)
            self.ops.append(op)
    @tf.function
    def call(self, inputs, training=False):
        x = inputs
        for op in self.ops:
            x = op(x, training=training)
        return x


class CEM(Model):
    """Context Enhancement Module"""
    def __init__(self):
        super(CEM, self).__init__()
        self.conv4 = Conv2D(CEM_FILTER, 1, strides=1,
                        padding="SAME", use_bias=True)
        self.conv5 = Conv2D(CEM_FILTER, 1, strides=1,
                        padding="SAME", use_bias=True)
        #self.b = K.reshape(inputs, [-1, h, w, in_channel // group, group])
    @tf.function
    def call(self, inputs, training=False):
        C4_lat = self.conv4(inputs[0])
        C5_lat = self.conv5(inputs[1])
        C5_lat = tf.keras.backend.resize_images(C5_lat, 2, 2, "channels_last", "bilinear")
        Cglb_lat = K.reshape(inputs[2], [-1, 1, 1, CEM_FILTER])
        return C4_lat + C5_lat + Cglb_lat


class ShuffleNetv2(Model):
    """Shufflenetv2"""
    def __init__(self, num_classes, first_channel=24, channels_per_stage=(132, 264, 528)):
        super(ShuffleNetv2, self).__init__(name="ShuffleNetv2")

        self.num_classes = num_classes

        self.conv1_bn_relu = Conv2D_BN_ReLU(first_channel, 3, 2)
        self.pool1 = MaxPool2D(3, strides=2, padding="SAME")
        self.stage2 = ShufflenetStage(first_channel, channels_per_stage[0], 4)
        self.stage3 = ShufflenetStage(channels_per_stage[0], channels_per_stage[1], 8)
        self.stage4 = ShufflenetStage(channels_per_stage[1], channels_per_stage[2], 4)
        #self.conv5_bn_relu = Conv2D_BN_ReLU(1024, 1, 1)
        self.gap = GlobalAveragePooling2D()
        self.linear = Dense(num_classes)

    def build(self, input_shape):
        super(ShuffleNetv2, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=False):
        x = self.conv1_bn_relu(inputs, training=training)
        x = self.pool1(x)
        x = self.stage2(x, training=training)
        C4 = self.stage3(x, training=training)
        C5 = self.stage4(C4, training=training)
        #print("C5: ", K.int_shape(C5))
        #x = self.conv5_bn_relu(x, training=training)
        Cglb = self.gap(C5)
        print(K.int_shape(Cglb))
        x = self.linear(Cglb)
        #print(K.int_shape(x))

        return x, C4, C5, Cglb


class RPNROI(Model):
    """roi"""
    def __init__(self):
        super(RPNROI, self).__init()

    @tf.function
    def call(self, inputs, training=False):
        """inputs=[SAM, rpn_conf, rpn_pbbox] """
        pass #TODO

class RPN(Model):
    """region proposal network"""

    def __init__(self, filter=256):
        super(RPN, self).__init__()
        self.num_anchors = 5*5
        self.rpn = DepthwiseConv2D_BN_POINT(kernel_size=6, stride=1, out_channel=filter)
        self.rpn_cls_score = Conv2D(2*self.num_anchors, 1, strides=1,
                                padding="VALID", use_bias=True)
        self.rpn_cls_pred = Conv2D(4*self.num_anchors, 1, strides=1,
                                padding="VALID", use_bias=True)
    @tf.function
    def call(self, inputs, training=False):
        rpn = self.rpn(inputs, training=training)
        rpn_conf  = self.rpn_cls_score(rpn)
        #cls_pred  = tf.reshape(rpn_cls_score, [tf.shape(rpn_cls_score)[0], -1, 2]
        rpn_pbbox  = self.rpn_cls_pred(rpn)


        return rpn, rpn_conf, rpn_pbbox




class SAM(Model):
    """spatial attention module"""
    def __init__(self):
        super(SAM, self).__init__()
        self.point =  Conv2D(CEM_FILTER, 1, strides=1,
                                padding="VALID", use_bias=False)
        self.bn = BatchNormalization()

    @tf.function
    def call(self, inputs, training=False):
        """[RPN, CEM] """
        x = self.point(inputs[0])
        x = self.bn(x)
        x = tf.keras.activations.softmax(x, axis=-1)
        x = tf.math.multiply(x, inputs[1])
        return x



import numpy as np

if __name__ == "__main__":



    s = (10, 320, 320, 12)
    nx = np.random.rand(*s).astype(np.float32)

    custom_layers = [
        ShufflenetUnit1(out_channel = s[-1]),
        ShufflenetUnit2(in_channel = 24, out_channel = 116),
        ShufflenetStage(in_channel = 24, out_channel = 116, num_blocks = 5)
    ]

    for layer in custom_layers:
        tf.keras.backend.clear_session()
        out = layer(nx)
        layer.summary()
        print(f"Input  Shape: {nx.shape}")
        print(f"Output Shape: {out.shape}")
        print("\n" * 2)

    tf.keras.backend.clear_session()
    g = ShuffleNetv2(CEM_FILTER)





    shape = (10, 320, 320, 3)
    nx = np.random.rand(*shape).astype(np.float32)
    t = tf.keras.Input(shape=nx.shape[1:], batch_size=nx.shape[0])

    x,C4,C5,Cglb = g(nx, training=False)

    cem = CEM()
    sam = SAM()
    rpn = RPN()
    re = cem([C4, C5, x], training=False)

    rpn_result, rpn_cls_score, rpn_cls_pred = rpn(re, training =False)

    sam_result = sam([rpn_result, re], training=False)


    print('cem rsult: ', K.int_shape(re))


    g.summary()
    sam.summary()
    cem.summary()
    rpn.summary()

