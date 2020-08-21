import os
os.environ['TF_KERAS'] = '1'

import tensorflow as tf
from bert4keras.backend import K, batch_gather, keras
import pandas as pd
from bert4keras.layers import *
from bert4keras.models import *
from bert4keras.optimizers import *
from bert4keras.snippets import sequence_padding
import logging


def metrics_f1(y_true, y_pred):
    # tf.where(condition, true返回x, False返回y)

    y_pred = tf.where(y_pred < 0.5, x = tf.zeros_like(y_pred), y = tf.ones_like(y_pred))
    equal_num = tf.math.count_nonzero(tf.multiply(y_true, y_pred))
    true_sum = tf.math.count_nonzero(y_true)
    pred_sum = tf.math.count_nonzero(y_pred)
    precesion = equal_num / pred_sum
    recall = equal_num / true_sum
    f1 = (2 * precesion * recall) / (precesion + recall)
    return f1


class Evaluate(keras.callbacks.Callback):
    def __init__(self, validation_data, filepath, stop_patience=2, verbose=1):
        val_data, val_label = validation_data
        self.val = val_data
        self.label = val_label
        self.best = 0
        self.f1_raise = 1
        self.stop_patience = stop_patience
        self.filepath = filepath
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        print('Evaluate:')
        precision, recall, f1, = self.evaluate()
        if f1 > self.best:
            self.best = f1
            self.model.save_weights(self.filepath)

        print(' precision: %.6f, recall: %.6f,f1: %.6f, best f1: %.6f\n' % (
            float(precision), float(recall), float(f1), float(self.best)))

        logging.debug(str(precision) + ' ' + str(recall) + ' ' + str(f1))

    def evaluate(self):
        pred = self.model.predict(self.val)
        return self.link_f1(self.label, pred)

    def stop_train(self, F1, best_f1, stop_patience):
        stop = True
        for f in F1[-stop_patience:]:
            if f >= best_f1:
                stop = False
        if stop == True:
            print('EarlyStopping!!!')
            self.model.stop_training = True

    def link_f1(self, y_true, y_pred):
        threshold_valud = 0.5
        y_true = np.reshape(y_true, (-1))
        y_pred = [1 if p > threshold_valud else 0 for p in np.reshape(y_pred, (-1))]
        equal_num = np.sum([1 for t, p in zip(y_true, y_pred) if t == p and t == 1 and p == 1])
        true_sum = np.sum(y_true)
        pred_sum = np.sum(y_pred)
        precision = equal_num / pred_sum
        recall = equal_num / true_sum
        f1 = (2 * precision * recall) / (precision + recall)

        print('equal_num:', equal_num)
        print('true_sum:', true_sum)
        print('pred_sum:', pred_sum)
        print('precision:', precision)
        print('recall:', recall)
        print('f1:', f1)

        return precision, recall, f1

def get_input():
    data = pd.read_pickle('./ccks2020_el_data/train_dev_iput.pkl')
    data['ids'] = sequence_padding(data['ids'])
    data['seg'] = sequence_padding(data['seg'])
    inputs = [data['ids'], data['seg'], data['entity_id'], np.expand_dims(data['labels'], axis=-1)]
    print(inputs[-1])
    return inputs

def step_decay(epoch):
    if epoch < 3:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr

def extract_subject(inputs):

    output, entity_ids = inputs
    entity_ids = K.cast(entity_ids, 'int32')
    start = batch_gather(output, entity_ids[:, :1])
    end = batch_gather(output, entity_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]

def bert_model(config, ckpt, pred=False):

    entity_id = Input(shape=(2, ), name='entity_ids')

    bert = build_transformer_model(config_path=config,
                                   checkpoint_path=ckpt,
                                   return_keras_model=False)
    if pred:
        bert.dropout_rate = 0

    output = bert.model.output
    cls = Lambda(lambda x: x[:, 0], name='CLS-token')(output)

    entity_vec = Lambda(extract_subject, name='Entity-vec')([output, entity_id])


    output = concatenate([cls, entity_vec], axis=-1, name='cls-entity-vec')

    output = Dropout(rate=bert.dropout_rate)(output)

    logits = Dense(units=24,
                   activation='sigmoid',
                   kernel_initializer=bert.initializer)(output)

    model = Model(bert.model.inputs + [entity_id], logits)

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=[metrics_f1])

    model.summary()
    return model

def train(config, ckpt):
    data = get_input()
    data_ids, data_seg, data_entity_id, data_y = data
    split = -70000
    input_train = [data_ids[:split], data_seg[:split], data_entity_id[:split], data_y[:split]]
    input_val = [data_ids[split:], data_seg[split:], data_entity_id[split:], data_y[split:]]

    # filepath_loss = 'model/ED_binary_model_bert_loss.h5_' + str(i)
    filepath_f1 = 'model/ED_binary_model_bert_f1.h5_' + 'new'
    evaluate = Evaluate((input_val[:-1], input_val[-1]), filepath_f1)
    # checkpoint = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=2, save_weights_only=True, save_best_only=True, mode='min')
    # reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    lrate = keras.callbacks.LearningRateScheduler(step_decay, verbose=2)
    callbacks = [evaluate, lrate, earlystopping]
    model = bert_model(config, ckpt)

    model.load_weights('./model/ED_binary_model_bert_f1.h5_new')

    model.fit(x=input_train[:-1],
              y=input_train[-1],
              batch_size=30,
              epochs=4,
              validation_data=(input_val[:-1],
                               input_val[-1]),
              verbose=1,
              callbacks=callbacks)

    print('one over')

if __name__ == "__main__":
    config = 'roberta_zh_l12/bert_config.json'
    ckpt = 'roberta_zh_l12/bert_model.ckpt'
    # model, bert = bert_model(config, ckpt)
    train(config, ckpt)
