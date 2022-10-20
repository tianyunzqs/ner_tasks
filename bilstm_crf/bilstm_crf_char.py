# -*- coding: utf-8 -*-
# @Time        : 2022/10/17 16:27
# @Author      : tianyunzqs
# @Description : 用BiLSTM+CRF做中文命名实体识别

import os
import json
import collections
import jieba
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
char_embedding_size = 128
seg_embedding_size = 20
maxlen = 256
epochs = 30
batch_size = 64
learning_rate = 1e-3
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
K.set_session(session)


def load_data_lines(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    categories = list()
    char_counter = collections.Counter()
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for i, l in enumerate(f.split('\n\n')):
            # if i > 5000:
            #     break
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                char_counter.update([char])
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    if flag[2:] not in categories:
                        categories.append(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D, categories, char_counter


def load_data_json(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    categories = set()
    char_counter = collections.Counter()
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = [l['text']]
            char_counter.update(list(l['text']))
            for k, v in l['label'].items():
                categories.add(k)
                for spans in v.values():
                    for start, end in spans:
                        d.append((start, end, k))
            D.append(d)
    return D, categories, char_counter


# 标注数据
train_data, categories, char_counter = load_data_lines(os.path.join(project_path, 'data/example.train'))
valid_data, _, _ = load_data_lines(os.path.join(project_path, 'data/example.dev'))
categories, _, _ = list(sorted(categories))

char_counter = {k: v for k, v in char_counter.items() if v >= 5}
char2id = {"<PAD>": 0}
char2id.update({k: i + 1 for i, (k, v) in enumerate(char_counter.items())})
char2id['<UNK>'] = len(char2id) + 1
id2char = {v: k for k, v in char2id.items()}


class TyDataGenerator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_sentence_ids, batch_labels = [], []
        for is_end, d in self.sample(random):
            sentence_ids = [char2id[w if w in char2id else '<UNK>'] for w in d[0]]
            batch_sentence_ids.append(sentence_ids)

            labels = np.zeros(len(sentence_ids))
            for start, end, label in d[1:]:
                labels[start] = categories.index(label) * 2 + 1
                labels[start + 1:end + 1] = categories.index(label) * 2 + 2
            batch_labels.append(labels)

            if len(batch_sentence_ids) == self.batch_size or is_end:
                batch_sentence_ids = sequence_padding(batch_sentence_ids)
                batch_labels = sequence_padding(batch_labels)
                yield batch_sentence_ids, batch_labels
                batch_sentence_ids, batch_labels = [], []


char_input = keras.layers.Input(shape=(None,), name='char_input')
char_embedding = keras.layers.Embedding(len(char2id) + 1, char_embedding_size, embeddings_initializer='zeros')
t = char_embedding(char_input)
t = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(t)
t = keras.layers.Dropout(0.5)(t)
output = Dense(len(categories) * 2 + 1)(t)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(char_input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        sentence_ids = [char2id[w if w in char2id else '<UNK>'] for w in text]

        token_ids = to_array([sentence_ids])
        nodes = model.predict([token_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(w[0], w[-1], l) for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('best.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = TyDataGenerator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best.weights')
    NER.trans = K.eval(CRF.trans)
