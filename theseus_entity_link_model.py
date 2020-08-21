#! -*- coding:utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'

import tensorflow as tf
from bert4keras.backend import K, batch_gather, keras
import pandas as pd
from bert4keras.layers import *
from bert4keras.models import *
from bert4keras.optimizers import *
from bert4keras.snippets import sequence_padding, DataGenerator

config = 'roberta_zh_l12/bert_config.json'
ckpt = 'roberta_zh_l12/bert_model.ckpt'

train_input_path = './ccks2020_el_data/train_iput.pkl'
dev_input_path = './ccks2020_el_data/dev_iput.pkl'




def get_input(path, small_sample=False, sample_num=100):

    data = pd.read_pickle(path)
    data['ids'] = sequence_padding(data['ids'])
    data['seg'] = sequence_padding(data['seg'])
    inputs = [data['ids'], data['seg'], data['entity_id'], np.expand_dims(data['labels'], axis=-1)]
    if small_sample:
        inputs = [inputs[0][:sample_num], inputs[1][:sample_num], inputs[2][:sample_num], inputs[3][:sample_num]]
    return inputs

def metrics_f1(y_true, y_pred):
    # tf.where(condition, true返回x, False返回y)

    y_pred = tf.where(y_pred < 0.5, x = tf.zeros_like(y_pred), y = tf.ones_like(y_pred))
    equal_num = tf.math.count_nonzero(tf.multiply(y_true, y_pred))
    true_sum = tf.math.count_nonzero(y_true)
    pred_sum = tf.math.count_nonzero(y_pred)
    precesion = equal_num / pred_sum
    recall = equal_num / true_sum
    f1 = (2 * precesion * recall) / (precesion + recall + 1e-10)
    return f1

def evaluate(y_true, y_pred):
    threshold_value = 0.5
    y_true = np.reshape(y_true, newshape=(-1))
    y_pred = np.reshape(y_pred, newshape=(-1))
    y_pred = np.where(y_pred>threshold_value, np.ones_like(y_pred), np.zeros_like(y_pred))
    equal_num = np.count_nonzero(np.multiply(y_true, y_pred))
    true_sum = np.count_nonzero(y_true)
    pred_sum = np.count_nonzero(y_pred)
    precesion = equal_num / pred_sum
    recall = equal_num / true_sum
    f1 = (2 * precesion * recall) / (precesion + recall)
    return precesion, recall, f1

class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, filepath):

        self.val = val_dataset[:-1]
        self.label = val_dataset[-1]
        self.filepath = filepath
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.val)
        precesion, recall, f1 = evaluate(self.label, y_pred)
        if f1 > self.best:
            self.best = f1
            self.model.save_weights(self.filepath)
            print('\n f1_score improved and saved')
        else:
            print('\n not improved not saved')
        print('\n 精确率：{}，\n 召回率：{} \n F1：{} \n best F1: {}\n'.format(
            precesion,
            recall,
            f1,
            self.best
        ))


def lr_schedual(epoch):
    if epoch < 4:
        return 2e-5
    else:
        return 3e-6


class BinaryRandomChoice(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BinaryRandomChoice, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[1]

    # 训练的时候，随机选择前辈和后辈的输出
    # 测试的时候，选择后辈的输出
    def call(self, inputs, **kwargs):
        source, target = inputs
        mask = K.random_binomial(shape=[1], p=0.5)
        # 1 选择前辈
        # 0 选择后辈
        output = mask * source + (1- mask) * target
        return K.in_train_phase(output, target)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def extract_subject(inputs):

    output, entity_ids = inputs
    entity_ids = K.cast(entity_ids, 'int32')
    start = batch_gather(output, entity_ids[:, :1])
    end = batch_gather(output, entity_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]

# 判别模型
def Classfier():

    x_in = Input(shape=(None, 768))

    entity = Input(shape=(2,), name='entity_loc')

    cls = Lambda(lambda x: x[:, 0], name='CLS')(x_in)

    entity_vec = Lambda(extract_subject, name='extract_entity_vec')([x_in, entity])

    output = concatenate([cls, entity_vec], axis=-1, name='cls-entity')

    x = Dense(units=1,
              activation='sigmoid',
              )(output)

    classfier = tf.keras.Model([x_in, entity], x, name='classfier')

    classfier.summary()

    return classfier, entity


def Bert_layer(num_hidden_layers, prefix):
    bert = build_transformer_model(
                                   config_path=config,
                                   checkpoint_path=ckpt,
                                   return_keras_model=False,
                                   num_hidden_layers=num_hidden_layers,
                                   prefix=prefix
                                   )
    return bert


def bert_of_theseus(predecessor, successor, classfier, entity):
    inputs = predecessor.inputs
    for layer in predecessor.model.layers:
        layer.trainable = False
    classfier.trainable = False

    # 替换embedding层
    predecessor_outputs = predecessor.apply_embeddings(inputs)
    successor_outputs = successor.apply_embeddings(inputs)

    outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])

    # 替换Transformer层

    layer_per_module = predecessor.num_hidden_layers // successor.num_hidden_layers

    for index in range(successor.num_hidden_layers):
        predecessor_outputs = outputs
        for sub_index in range(layer_per_module):
            predecessor_outputs = predecessor.apply_main_layers(
                predecessor_outputs, layer_per_module * index + sub_index
            )
        successor_outputs = successor.apply_main_layers(outputs, index)
        outputs = BinaryRandomChoice()([predecessor_outputs, successor_outputs])

    outputs = classfier([outputs, entity])
    model = Model(inputs+[entity], outputs, name='theseus')
    return model



def All_Models():
    classfier, entity = Classfier()

    # -------------------------前辈-----------------------------#
    predecessor_bert = Bert_layer(num_hidden_layers=12, prefix='Predecessor-')

    predecessor = tf.keras.Model(
        predecessor_bert.inputs+[entity],
        classfier([predecessor_bert.output, entity]),
        name='Predecessor'
    )

    predecessor.compile(
        loss='binary_crossentropy',
        optimizer=Adam(2e-5),
        metrics=[tf.keras.metrics.BinaryAccuracy(), metrics_f1]
    )

    predecessor.summary()

    # ---------------------------后辈 ------------------------- #
    successor_bert = Bert_layer(num_hidden_layers=3, prefix='Successor-')

    successor = tf.keras.Model(
        successor_bert.inputs + [entity],
        classfier([successor_bert.output, entity]),
        name='successor'
    )
    successor.compile(
        loss='binary_crossentropy',
        optimizer=Adam(2e-5),
        metrics=[tf.keras.metrics.BinaryAccuracy(), metrics_f1]
    )

    successor.summary()

    #---------------------------忒休斯--------------------------#

    theseus = bert_of_theseus(predecessor_bert, successor_bert, classfier, entity)
    theseus.summary()
    theseus.compile(
        loss='binary_crossentropy',
        optimizer=Adam(2e-5),
        metrics=[tf.keras.metrics.BinaryAccuracy(), metrics_f1]
    )

    return predecessor, successor, theseus

if __name__ == "__main__":
    # 加载模型
    predecessor, successor, theseus = All_Models()

    # 训练测试，数据
    train_data = get_input(train_input_path)
    dev_data = get_input(dev_input_path)

    lr = tf.keras.callbacks.LearningRateScheduler(lr_schedual, verbose=2)

    # 训练先辈
    print('----------------------------训练先辈--------------------------')
    predecessor_evaluator = Evaluate(dev_data, 'best_predecessor.weights')
    predecessor.fit(
        train_data[:-1],
        train_data[-1],
        batch_size=30,
        epochs=5,
        callbacks=[predecessor_evaluator, lr]
    )

    # 训练忒休斯，后辈的bert效果接近先辈的bert
    print('----------------------------训练忒休斯--------------------------')
    theseus_evaluate = Evaluate(dev_data, 'best_theseus.weights')
    theseus.fit(
        train_data[:-1],
        train_data[-1],
        batch_size=30,
        epochs=10,
        callbacks=[theseus_evaluate]
    )

    # 微调后辈
    print('----------------------------微调后辈--------------------------')
    successor_evaluate = Evaluate(dev_data, 'best_successor.weights')
    successor.fit(
        train_data[:-1],
        train_data[-1],
        batch_size=30,
        epochs=5,
        callbacks=[successor_evaluate, lr]
    )