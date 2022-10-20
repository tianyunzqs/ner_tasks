# -*- coding: utf-8 -*-
# @Time        : 2018/10/17 18:53
# @Author      : tianyunzqs
# @Description :

# import re
#
#
# def gen_trie_names(f_name, trie):
#     """
#     构建词典树
#     :param f_name: 词典路径
#     :param trie: 词典树
#     :return:
#     """
#     with open(f_name, "r", encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in lines:
#             parts = re.split(r"\s+", line.strip())
#             if len(parts) == 3:
#                 word, freq, tag = parts
#                 if tag != "nr":
#                     continue
#                 p = trie
#                 for w in word:
#                     if w not in p:
#                         p[w] = {}
#                     p = p[w]
#                 p[""] = ""
#             elif len(parts) == 1:
#                 word = parts[0]
#                 p = trie
#                 for w in word:
#                     if w not in p:
#                         p[w] = {}
#                     p = p[w]
#                 p[""] = ""
#
#
# def is_word_in_trie(word, trie):
#     """
#     判断word是否在词典树trie中
#     :param word: 待判定的词
#     :param trie: 词典树
#     :return: 在，返回True；不在，返回False
#     """
#     p = trie
#     for w in word:
#         if w not in p:
#             return False
#         else:
#             p = p[w]
#     if "" in p:
#         return True
#     else:
#         return False
#
#
# # 人名词典树
# trie_names = {}
# gen_trie_names('E:/pyworkspace/ImageQ_GAMind/nlp_semantic_analysis/person_recognition/pangu/name_dict/nr_dict.txt', trie_names)
#
#
# loc_list = set()
# org_list = set()
# per_list = set()
#
# with open("./data/example.train", "r", encoding="utf8") as f:
#     lines = f.readlines()
#     text, flag = "", ""
#     for line in lines:
#         line = line.strip()
#         parts = line.split()
#         if len(parts) == 2:
#             if parts[1] == "B-LOC":
#                 text = parts[0]
#                 flag = "LOC"
#             elif parts[1] == "I-LOC":
#                 text += parts[0]
#             elif parts[1] == "B-ORG":
#                 text = parts[0]
#                 flag = "ORG"
#             elif parts[1] == "I-ORG":
#                 text += parts[0]
#             elif parts[1] == "B-PER":
#                 text = parts[0]
#                 flag = "PER"
#             elif parts[1] == "I-PER":
#                 text += parts[0]
#             elif parts[1] == "O":
#                 if text:
#                     if flag == "LOC" and is_word_in_trie(text, trie_names):
#                         loc_list.add(text)
#                     elif flag == "ORG" and is_word_in_trie(text, trie_names):
#                         org_list.add(text)
#                     elif flag == "PER" and is_word_in_trie(text, trie_names):
#                         per_list.add(text)
#
# print(loc_list)
# print(org_list)
# print(per_list)
