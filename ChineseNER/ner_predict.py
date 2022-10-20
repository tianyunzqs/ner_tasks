# encoding=utf8

import os
import sys
import pickle
import jieba.posseg as pseg
import tensorflow as tf
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from ChineseNER.model import Model
from ChineseNER.ner_utils import load_config
from ChineseNER.data_utils import full_to_half, replace_html


class Predictor(object):
    def __init__(self):
        tf.reset_default_graph()
        pseg.initialize()
        config = load_config(os.path.join(project_path, "ChineseNER/ner_model/config_file"))
        with open(os.path.join(project_path, "ChineseNER/ner_model/maps.pkl"), "rb") as f:
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(f)

        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        self.Session = tf.Session(graph=self.graph)  # 创建新的sess
        with self.Session.as_default():
            with self.graph.as_default():
                self.model = Model(config)  # 模型必须在新建的graph中创建，否则会出错
                self.model.saver.restore(self.Session, os.path.join(project_path, "ChineseNER/ner_model/ner.ckpt"))

    @staticmethod
    def is_dict_in_list(maybe_error_dict, _list):
        for tmp_dict in _list:
            if maybe_error_dict['start'] >= tmp_dict['start'] \
                    and maybe_error_dict['end'] <= tmp_dict['end'] \
                    and maybe_error_dict['word'] in tmp_dict['word']:
                return True
        return False

    @staticmethod
    def entities_combine(entities):
        if len(entities) <= 1:
            return entities
        _list_or_set_tmp = sorted(entities, key=lambda x: len(x['word']))
        result = list()
        for i in range(len(_list_or_set_tmp)):
            for j in range(i + 1, len(_list_or_set_tmp)):
                if _list_or_set_tmp[i]['word'] in _list_or_set_tmp[j]['word'] \
                        and _list_or_set_tmp[i]['start'] >= _list_or_set_tmp[j]['start'] \
                        and _list_or_set_tmp[i]['end'] <= _list_or_set_tmp[j]['end']:
                    break
            else:
                result.append(_list_or_set_tmp[i])
        return result

    def input_from_line(self, line, seg_feature):
        line = full_to_half(line)
        line = replace_html(line)
        inputs = list()
        inputs.append([line])
        line.replace(" ", "$")
        inputs.append(
            [[self.char_to_id[char] if char in self.char_to_id else self.char_to_id["<UNK>"] for char in line]])
        inputs.append([seg_feature])
        inputs.append([[]])
        return inputs

    def model_predict(self, line):
        if not isinstance(line, str) or not line.strip():
            return []
        result = []
        seg_feature = []
        idx = 0
        for word in pseg.cut(line):
            if word.flag == 'company':
                result.append({'word': word.word, 'start': idx, 'end': idx+len(word.word), 'type': 'COMPANY'})
            if len(word.word) == 1:
                seg_feature.append(0)
            else:
                tmp = [2] * len(word.word)
                tmp[0] = 1
                tmp[-1] = 3
                seg_feature.extend(tmp)

            idx += len(word.word)

        inputs = self.input_from_line(line, seg_feature)
        res = self.model.evaluate_line(self.Session, inputs, self.id_to_tag)
        return [(w['word'], 0) for w in self.entities_combine(res["entities"])]


if __name__ == "__main__":
    text = "根据督导情况，建议对权健公司在违法广告、传销等方面可能涉嫌刑事犯罪的行为，由公安机关介入侦查"
    # text = "南方财经全媒体记者江月 报道历经一次打断，商汤赴港发行IPO终将成行。12月29日清晨，该公司公布了招股结果，最终募得57.75亿港元。该股将于29日下午港股收市后进行暗盘交易，30日早晨将登陆港交所正式挂牌交易。"
    text = "但是机器人企业今年的三季报表现却不乐观，国内一些代表企业如埃斯顿、拓斯达、哈工智能(6.010, -0.03, -0.50%)等出现了营收大涨，利润暴跌的情形。"
    predictor_PER = Predictor()
    while True:
        text = input("text: ")
        print(predictor_PER.model_predict(text))
