import os
os.environ['TF_KERAS'] = '1'

from bert4keras.layers import *
from bert4keras.backend import K, keras, batch_gather
from bert4keras.models import *
from bert4keras.optimizers import *
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from collections import Counter

config = 'roberta_zh_l12/bert_config.json'
ckpt = 'roberta_zh_l12/bert_model.ckpt'

def extract_subject(inputs):

    output, entity_ids = inputs
    entity_ids = K.cast(entity_ids, 'int32')
    start = batch_gather(output, entity_ids[:, :1])
    end = batch_gather(output, entity_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]

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

    logits = Dense(units=46,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)

    model = Model(bert.model.inputs + [entity_id], logits)

    model.compile(optimizer=Adam(5e-6),
                  loss=sparse_categorical_crossentropy,
                  metrics=[sparse_categorical_accuracy])

    model.summary()

    return model

def get_input(input_file=r'./ccks2020_el_data/nil_train_input.pkl'):
    data = pd.read_pickle(input_file)
    inputs = [data['ids'], data['seg'], data['entity_id'], data['label']]
    inputs[-1] = np.expand_dims(inputs[-1], -1)
    return inputs

def train():
    dataset = get_input()
    train = dataset

    model = nil_type_model()
    model_path = './model/nil_loss.h5_other'
    model.load_weights(model_path)
    # print(model.summary())
    loss_path = './model/nil_loss.h5_other'
    checkpoint = ModelCheckpoint(filepath=loss_path, monitor='val_loss', verbose=2, save_weights_only=True, save_best_only=True, mode='min')
    model.fit(train[:-1], train[-1], batch_size=50, epochs=20, validation_split=0.2, verbose=1, callbacks=[checkpoint],shuffle=True)


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
    train()