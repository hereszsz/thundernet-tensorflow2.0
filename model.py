from layers import ShuffleNetv2, CEM, RPN, SAM, CEM_FILTER
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras as keras

import tensorflow as tf
import numpy as np

class ThunderNet_bb(Model):
    """ thundernet without roi part"""
    def __init__(self):
        super(ThunderNet_bb, self).__init__(name="ThunderNet_bb")
        self.bb=ShuffleNetv2(CEM_FILTER)
        self.cem=CEM()
        self.rpn=RPN()
        self.sam=SAM()

    def build(self, input_shape):
        super(ThunderNet_bb, self).build(input_shape)
    @tf.function
    def call(self, inputs, training=False):
        x,C4,C5,_ = self.bb(inputs, training=training)
        re = self.cem([C4, C5, x], training=training)
        rpn_result, rpn_cls_score, rpn_cls_pred = self.rpn(re, training =training)
        sam_result = self.sam([rpn_result, re], training=training)
        return sam_result, rpn_result, rpn_cls_score, rpn_cls_pred

    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape([10, 20, 20, 245])


from tensorflow.keras import layers


if __name__ == "__main__":
    print("summary for ThunderNet_bb")
    #K.clear_session()

    shape = (10, 320, 320, 3)
    nx = np.random.rand(*shape).astype(np.float32)

    g = ThunderNet_bb()
    t = keras.Input(shape=nx.shape[1:], batch_size=nx.shape[0])
    sam_result, rpn_result, rpn_cls_score, rpn_cls_pred = g(nx, training=False)
    #sam_result = g(t, training=False)

    g.summary()

    print('thundernet_backbone result: \nsam_result, rpn_result, rpn_cls_score, rpn_cls_pred: ', list(map(K.int_shape, [sam_result, rpn_result, rpn_cls_score, rpn_cls_pred])))

