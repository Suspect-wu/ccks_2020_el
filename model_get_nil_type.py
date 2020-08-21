import pandas as pd
from keras.layers import *
from keras.models import *
from keras_layers import *
from keras_bert import load_trained_model_from_checkpoint
from keras_layers import CLSOut, StateMixOne
from keras.optimizers import *
from keras.callbacks import *
from keras.losses import *
from keras.utils import to_categorical
from keras.metrics import *
import tensorflow as tf
from sklearn.model_selection import KFold
import json
import numpy as np



config_path = './chinese_wwm_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_wwm_L-12_H-768_A-12/vocab.txt'


def nil_model():
    input_begin = Input(shape=(1, ), name='men_sen')
    input_end = Input(shape=(1, ), name='men_pos')
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=True, seq_len=55)
    x = bert_model.output

    cls = CLSOut()(x)
    entity_em = StateMixOne()([input_begin, input_end, x, x])
    x = concatenate([cls, entity_em], axis=-1)

    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(units=32, activation='softmax')(x)

    model = Model(bert_model.inputs + [input_begin, input_end], x)
    model.compile(optimizer=adam(1e-6), loss=categorical_crossentropy, metrics=['accuracy'])
    return model

def get_input(input_file=r'./ccks2020_el_data/nil_train_input.pkl', mode = 'train'):
    if mode == 'test':
        # data = input_file
        return
    else:
        data = pd.read_pickle(input_file)
        inputs = [data['ids'], data['seg'], data['begin'], data['end'], to_categorical(data['label'], num_classes=32)]
        return inputs



def train():
    dataset = get_input()
    train = dataset

    model = nil_model()
    model_path = './model/nil_loss.h5'
    model.load_weights(model_path)
    print(model.summary())
    loss_path = './model/nil_loss.h5_other'
    checkpoint = ModelCheckpoint(filepath=loss_path, monitor='val_loss', verbose=2, save_weights_only=True, save_best_only=True, mode='min')
    model.fit(train[:-1], train[-1], batch_size=40, epochs=5, validation_split=0.2, verbose=1, callbacks=[checkpoint],shuffle=True)
if __name__ == '__main__':
    train()




