#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from random import shuffle
import jieba
import argparse
import logging
import synonyms

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eda")


class EDA(object):
    __doc__ = """eda class"""

    _baidu_stop_words_path = "stopwords/baidu_stopwords.txt"
    _cn_stop_words_path = "stopwords/cn_stopwords.txt"
    _hit_stop_words_path = "stopwords/hit_stopwords.txt"
    _scu_stop_words_path = "stopwords/scu_stopwords.txt"
    random.seed(2021)

    def __init__(self, num_aug=2, stop_words=None, stop_words_type="hit"):
        """
        EDA初始化
        :param num_aug: 增强的数据量
        :param stop_words:自定义停用词表
        :param stop_words_type: 默认提供四种停用词表:"hit", "cn", "baidu", "scu"，默认是hit
        """
        self.num_aug = num_aug
        self.stop_words = stop_words
        self.stop_words_type = stop_words_type
        if self.stop_words is None:
            self.stop_words = self._load_stop_words()

    def _load_stop_words(self):
        """
        load stop words
        :return:
        """
        if self.stop_words_type == "hit":
            file_path = self._hit_stop_words_path
        elif self.stop_words_type == "cn":
            file_path = self._cn_stop_words_path
        elif self.stop_words_type == "scu":
            file_path = self._scu_stop_words_path
        elif self.stop_words_type == "baidu":
            file_path = self._baidu_stop_words_path
        else:
            raise Exception(f"you select stop words type is {self.stop_words_type}, not in ['hit', 'baidu', 'scu', "
                            f"'cn'], please check again")
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        logger.debug(f"loading stop words with file:{file_path}")
        stop_words = list()
        with open(file_path, 'r', encoding='utf8') as reader:
            for stop_word in reader:
                stop_words.append(stop_word[:-1])
        return stop_words

    @staticmethod
    def get_synonyms(word):
        """
        获取word最相近的词语
        :param word: 待获取的词语
        :return:
        """
        return synonyms.nearby(word)[0]

    def synonym_replacement(self, words, n):
        """同义词替换
        替换一个语句中的n个单词为其同义词
        :param words: 待处理的一个文本
        :param n: 随机替换的词语数
        :return:
        """
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms_words = self.get_synonyms(random_word)
            if len(synonyms_words) >= 1:
                synonym = random.choice(synonyms_words)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words

    def random_insertion(self, words, n):
        """
        随机在语句中插入n个词
        :param words: 语句
        :param n: 随机插入词语的个数据
        :return:
        """
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        """
        找一个词语的相似词语，随机在语句中插入
        :param new_words:
        :return:
        """
        synonyms_word = []
        counter = 0
        while len(synonyms_word) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms_word = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms_word)
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)

    def random_swap(self, words, n):
        """
        Randomly swap two words in the sentence n times
        :param words: 语句
        :param n: 交换n个词语
        :return:
        """
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    @staticmethod
    def swap_word(new_words):
        """
        词语交换
        :param new_words:
        :return:
        """
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    @staticmethod
    def random_deletion(words, p):
        """
        以概率p删除语句中的词
        :param words: 语句
        :param p: 概率
        :return:
        """
        if len(words) == 1:
            return words
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]
        return new_words

    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
        """
        eda main function
        :param sentence: 待处理的语句
        :param alpha_sr: 同义词替换词语比例
        :param alpha_ri: 随机插入比例
        :param alpha_rs: 随机交换比例
        :param p_rd: 随机删除概率
        :return:
        """

        def enhance(method_inner, param_inner):
            logger.debug(f"use method:{method_inner.__name__}")
            tmp_result = []
            for _ in range(num_new_per_technique):
                a_words = method_inner(*param_inner)
                tmp_result.append(' '.join(a_words))
            return tmp_result

        seg_list = jieba.cut(sentence)
        seg_list = " ".join(seg_list)
        words = list(seg_list.split())
        num_words = len(words)
        augmented_sentences = []
        num_new_per_technique = int(self.num_aug / 4) + 1  # every method generate sentence number
        n_sr = max(1, int(alpha_sr * num_words))  # cal synonym replace number
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        # 同义词替换sr，随机插入ri，随机交换rs，随机删除rd
        methods = [self.synonym_replacement, self.random_insertion, self.random_swap, self.random_deletion]
        params = [(words, n_sr), (words, n_ri), (words, n_rs), (words, p_rd)]
        for method, param in zip(methods, params):
            augmented_sentences.extend(enhance(method, param))

        shuffle(augmented_sentences)

        if self.num_aug >= 1:
            augmented_sentences = augmented_sentences[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
        augmented_sentences.append(seg_list)
        return augmented_sentences


class SimpleEDAEnhance(object):

    @staticmethod
    def simple_eda_enhance():
        """
        通过控制台传入相关参数即可完成数据增强的方法
        :return:
        """
        ap = argparse.ArgumentParser()
        ap.add_argument("--input", required=True, type=str, help="原始数据的输入文件")
        ap.add_argument("--output", required=False, type=str, help="增强数据后的输出文件")
        ap.add_argument("--num_aug", required=False, type=int, default=9, help="每条原始语句增强的语句数")
        ap.add_argument("--alpha", required=False, type=float, default=0.1, help="每条语句中将会被改变的单词数占比")
        args = ap.parse_args()
        if args.output:
            output = args.output
        else:
            from os.path import dirname, basename, join
            output = join(dirname(args.input), 'eda_' + basename(args.input))
        eda = EDA(num_aug=args.num_aug)
        enhance_result = []
        with open(args.input, 'r', encoding='utf8') as reader:
            for index, line in enumerate(reader):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                label = parts[0]
                sentence = parts[1]
                aug_sentences = eda.eda(sentence, alpha_sr=args.alpha, alpha_ri=args.alpha, alpha_rs=args.alpha,
                                        p_rd=args.alpha)
                for aug_sentence in aug_sentences:
                    enhance_result.append(f"{label}\t{aug_sentence}")
        with open(output, 'w', encoding='utf8') as writer:
            writer.write("\n".join(enhance_result))
        logger.debug(f"enhance data has been generated, and save to {output}!")
