# coding:utf-8

import logging.config
import ConfigParser
import numpy as np
import random
import codecs
import os
from collections import OrderedDict

# 获取当前路径
path = os.getcwd()

# 导入日志配置文件
logging.config.fileConfig("logging.conf")

# 创建日志对象
logger = logging.getLogger()
# loggerInfo = logging.getLogger("TimeInfoLogger")
# Console_logger = logging.getLogger("ConsoleLogger")

# 导入配置文件
conf = ConfigParser.ConfigParser()
conf.read("setting.conf")

# 文件路径
train_file = os.path.join(path, os.path.normpath(conf.get("filepath", "trainfile")))
word_id_map_file = os.path.join(path,  os.path.normpath(conf.get("filepath", "wordidmapfile")))
theta_file = os.path.join(path, os.path.normpath(conf.get("filepath", "thetafile")))
phi_file = os.path.join(path, os.path.normpath(conf.get("filepath", "phifile")))
param_file = os.path.join(path, os.path.normpath(conf.get("filepath", "paramfile")))
top_N_file = os.path.join(path, os.path.normpath(conf.get("filepath", "topNfile")))
tassgin_file = os.path.join(path, os.path.normpath(conf.get("filepath", "tassginfile")))

# 模型初始参数
K = int(conf.get("model_args", "K"))
alpha = float(conf.get("model_args", "alpha"))
beta = float(conf.get("model_args", "beta"))
iter_times = int(conf.get("model_args", "iter_times"))
top_words_num = int(conf.get("model_args", "top_words_num"))


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0


class DataPreProcessing(object):

    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()

    def cache_word_id_map(self):
        with codecs.open(word_id_map_file, 'w', 'utf-8') as f:
            for word, id in self.word2id.items():
                f.write(word + "\t" + str(id) + "\n")


class LDAModel(object):
    
    def __init__(self, data_pre):

        # 获取预处理参数
        self.data_pre = data_pre

        # 模型参数
        # 聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta)
        self.K = K
        self.beta = beta
        self.alpha = alpha
        self.iter_times = iter_times
        self.top_words_num = top_words_num

        # 文件变量
        # 分好词的文件train_file
        # 词对应id文件word_id_map_file
        # 文章-主题分布文件theta_file
        # 词-主题分布文件phi_file
        # 每个主题topN词文件top_N_file
        # 最后分派结果文件tassgin_file
        # 模型训练选择的参数文件param_file
        self.word_id_map_file = word_id_map_file
        self.train_file = train_file
        self.theta_file = theta_file
        self.phi_file = phi_file
        self.top_N_file = top_N_file
        self.tassgin_file = tassgin_file
        self.param_file = param_file

        # p,概率向量 double类型，存储采样的临时变量
        # nw,词word在主题topic上的分布
        # nw_sum,每各topic的词的总数
        # nd,每个doc中各个topic的词的总数
        # nd_sum,每各doc中词的总数
        self.p = np.zeros(self.K)        
        self.nw = np.zeros((self.data_pre.words_count, self.K), dtype="int")
        self.nw_sum = np.zeros(self.K, dtype="int")
        self.nd = np.zeros((self.data_pre.docs_count, self.K), dtype="int")
        self.nd_sum = np.zeros(data_pre.docs_count, dtype="int")

        # M*doc.size()，文档中词的主题分布
        self.Z = np.array([[0 for y in xrange(data_pre.docs[x].length)] for x in xrange(data_pre.docs_count)])

        # 随机先分配类型
        for x in xrange(len(self.Z)):
            self.nd_sum[x] = self.data_pre.docs[x].length
            for y in xrange(self.data_pre.docs[x].length):
                topic = random.randint(0,self.K-1)
                self.Z[x][y] = topic
                self.nw[self.data_pre.docs[x].words[y]][topic] += 1
                self.nd[x][topic] += 1
                self.nw_sum[topic] += 1

        self.theta = np.array([[0.0 for y in xrange(self.K)] for x in xrange(self.data_pre.docs_count)])
        self.phi = np.array([[0.0 for y in xrange(self.data_pre.words_count)] for x in xrange(self.K)])

    def sampling(self, i, j):
        topic = self.Z[i][j]
        word = self.data_pre.docs[i].words[j]
        self.nw[word][topic] -= 1
        self.nd[i][topic] -= 1
        self.nw_sum[topic] -= 1
        self.nd_sum[i] -= 1

        v_beta = self.data_pre.words_count * self.beta
        k_alpha = self.K * self.alpha
        self.p = (self.nw[word] + self.beta) / (self.nw_sum + v_beta) * \
                 (self.nd[i] + self.alpha) / (self.nd_sum[i] + k_alpha)
        for k in xrange(1, self.K):
            self.p[k] += self.p[k-1]

        u = random.uniform(0, self.p[self.K-1])
        for topic in xrange(self.K):
            if self.p[topic] > u:
                break

        self.nw[word][topic] += 1
        self.nw_sum[topic] += 1
        self.nd[i][topic] += 1
        self.nd_sum[i] += 1

        return topic

    def est(self):

        # Console_logger.info(u"迭代次数为%s 次" % self.iter_times)
        for x in xrange(self.iter_times):
            for i in xrange(self.data_pre.docs_count):
                for j in xrange(self.data_pre.docs[i].length):
                    topic = self.sampling(i, j)
                    self.Z[i][j] = topic
        logger.info(u"迭代完成。")
        logger.debug(u"计算文章-主题分布")
        self._theta()
        logger.debug(u"计算词-主题分布")
        self._phi()
        logger.debug(u"保存模型")
        self.save()

    def _theta(self):
        for i in xrange(self.data_pre.docs_count):
            self.theta[i] = (self.nd[i]+self.alpha)/(self.nd_sum[i]+self.K * self.alpha)

    def _phi(self):
        for i in xrange(self.K):
            self.phi[i] = (self.nw.T[i] + self.beta)/(self.nw_sum[i]+self.data_pre.words_count * self.beta)

    def save(self):

        # 保存theta文章-主题分布
        logger.info(u"文章-主题分布已保存到%s" % self.theta_file)
        with codecs.open(self.theta_file, 'w') as f:
            for x in xrange(self.data_pre.docs_count):
                for y in xrange(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')

        # 保存phi词-主题分布
        logger.info(u"词-主题分布已保存到%s" % self.phi_file)
        with codecs.open(self.phi_file, 'w') as f:
            for x in xrange(self.K):
                for y in xrange(self.data_pre.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')

        # 保存参数设置
        logger.info(u"参数设置已保存到%s" % self.param_file)
        with codecs.open(self.param_file, 'w','utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')

        # 保存每个主题topic的词
        logger.info(u"主题topN词已保存到%s" % self.top_N_file)

        with codecs.open(self.top_N_file, 'w', 'utf-8') as f:
            self.top_words_num = min(self.top_words_num, self.data_pre.words_count)
            for x in xrange(self.K):
                f.write(u'第' + str(x) + u'类：' + '\n')
                t_words = [(n, self.phi[x][n]) for n in xrange(self.data_pre.words_count)]
                t_words.sort(key=lambda i: i[1], reverse=True)
                for y in xrange(self.top_words_num):
                    word = OrderedDict({value:key for key, value in self.data_pre.word2id.items()})[t_words[y][0]]
                    f.write('\t' * 2 + word + '\t' + str(t_words[y][1]) + '\n')

        # 保存最后退出时，文章的词分派的主题的结果
        logger.info(u"文章-词-主题分派结果已保存到%s" % self.tassgin_file)
        with codecs.open(self.tassgin_file, 'w') as f:
            for x in xrange(self.data_pre.docs_count):
                for y in xrange(self.data_pre.docs[x].length):
                    f.write(str(self.data_pre.docs[x].words[y])+':'+str(self.Z[x][y]) + '\t')
                f.write('\n')
        logger.info(u"模型训练完成。")


def pre_processing():
    logger.info(u'载入数据......')
    with codecs.open(train_file, 'r','utf-8') as f:
        docs = f.readlines()
    logger.debug(u"载入完成,准备生成字典对象和统计文本数据...")
    data_pre = DataPreProcessing()
    items_idx = 0
    for line in docs:
        if line != "":
            tmp = line.strip().split()

            # 生成一个文档对象
            doc = Document()
            for item in tmp:
                if data_pre.word2id.has_key(item):
                    doc.words.append(data_pre.word2id[item])
                else:
                    data_pre.word2id[item] = items_idx
                    doc.words.append(items_idx)
                    items_idx += 1
            doc.length = len(tmp)
            data_pre.docs.append(doc)
        else:
            pass

    data_pre.docs_count = len(data_pre.docs)
    data_pre.words_count = len(data_pre.word2id)
    logger.info(u"共有%s个文档" % data_pre.docs_count)
    data_pre.cache_word_id_map()
    logger.info(u"词与序号对应关系已保存到%s" % word_id_map_file)
    return data_pre


def run():
    data_pre = pre_processing()
    lda = LDAModel(data_pre)
    lda.est()
    

if __name__ == '__main__':
    run()
