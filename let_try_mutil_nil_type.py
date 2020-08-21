import os
os.environ['TF_KERAS'] = '1'

from bert4keras.layers import *
from bert4keras.backend import K, keras, batch_gather
from bert4keras.models import *
from bert4keras.optimizers import *
import tensorflow as tf

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint


import matplotlib.pyplot as plt
from collections import Counter

config = 'roberta_zh_l12/bert_config.json'
ckpt = 'roberta_zh_l12/bert_model.ckpt'

def my_metrics(y_true, y_pred):
    y_pred = tf.where(y_pred>0, x=tf.ones_like(y_pred), y=tf.zeros_like(y_pred))
    o = tf.math.equal(y_pred, y_true)
    o = tf.cast(o, dtype=tf.float32)
    o = tf.math.reduce_sum(o, axis=-1)
    all_sum = tf.math.count_nonzero(o)
    all_sum = tf.cast(all_sum, dtype=tf.float32)
    finial = tf.where(o==24, x=tf.ones_like(o), y=tf.zeros_like(o))
    finial = tf.math.reduce_sum(finial)
    accuracy = finial/all_sum
    return accuracy

def extract_subject(inputs):

    output, entity_ids = inputs
    entity_ids = K.cast(entity_ids, 'int32')
    start = batch_gather(output, entity_ids[:, :1])
    end = batch_gather(output, entity_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    # log(1+\sum{e^{s_i}})
    # 目的添加1
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)

    neg_loss = tf.math.reduce_logsumexp(y_pred_neg, axis=-1)
    pos_loss = tf.math.reduce_logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss

def nil_type_model():

    entity_id = Input(shape=(2, ), name='entity_ids')

    bert = build_transformer_model(config_path=config,
                                   ckpt=ckpt,
                                   return_keras_model=False)
    output = bert.output

    cls = Lambda(lambda x: x[:, 0], name='CLS')(output)

    entity_vec = Lambda(extract_subject, name='Entity-vec')([output, entity_id])

    output = concatenate([cls, entity_vec], axis=-1, name='cls-entity-vec')

    output = Dropout(rate=bert.dropout_rate)(output)

    logits = Dense(units=24,
                   kernel_initializer=bert.initializer)(output)

    model = Model(bert.model.inputs + [entity_id], logits)

    model.compile(optimizer=Adam(5e-6),
                  loss=[multilabel_categorical_crossentropy],
                  metrics=[my_metrics])

    model.summary()

    return model

def get_input(input_file=r'./ccks2020_el_data/multi_nil_train_input.pkl'):
    data = pd.read_pickle(input_file)
    inputs = [data['ids'], data['seg'], data['entity_id'], data['label']]
    for i in inputs:
        print(i.shape, i)
    return inputs

def train():
    dataset = get_input()
    train = dataset

    model = nil_type_model()
    # model_path = './model/nil_loss.h5_multi'
    # model.load_weights(model_path)
    # print(model.summary())
    loss_path = './model/nil_loss.h5_multi_1'
    checkpoint = ModelCheckpoint(filepath=loss_path, monitor='val_my_metrics', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    lr = keras.callbacks.LearningRateScheduler(step_decy, verbose=2)
    model.fit(train[:-1], train[-1], batch_size=100, epochs=15, validation_split=0.2, verbose=1,shuffle=True, callbacks=[checkpoint, lr])


def step_decy(epoch):
    if epoch < 4:
        lr = 2e-5
    elif epoch < 15:
        lr = 5e-6
    else:
        lr = 3e-7
    return lr


if __name__ == '__main__':
    # data = get_input()
    # model = nil_type_model()
    #
    # model.load_weights('./model/nil_loss.h5_other')
    # ids, seg, entity, label = data
    # for i in range(10):
    #     pred = model.predict([np.array([ids[i]]), np.array([seg[i]]), np.array([entity[i]])])
    #     print(pred)
    #     print(np.argmax(pred, axis=-1), label[i])
    # train()
    train()