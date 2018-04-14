# create by fanfan on 2018/3/18 0018
from math import log ,exp
class LapplaceEstimate(object):
    """
        拉普拉斯平滑处理的贝叶斯估计
        """
    def __init__(self):
        self.d = {}
        self.total = 0.0
        self.none = 1

    def exists(self,key):
        return key in self.d

    def getsum(self):
        return self.total

    def get(self,key):
        if not self.exists(key):
            return False,self.none
        return True,self.d[key]

    def getprob(self,key):
        """
                估计先验概率
                :param key: 词
                :return: 概率
                """
        return float(self.get(key)[1]) / self.total
    def samples(self):
        """
                获取全部样本
                :return:
                """
        return self.d.keys()

    def add(self,key,value):
        self.total += value
        if not self.exists(key):
            self.d[key] = 1
            self.total +=1
        self.d[key] +=value


class Bayes(object):
    def __init__(self):
        self.d = {} # [标签, 概率] map
        self.total = 0

    def train(self,data):
        for d in data:
            c = d[1]
            if c not in self.d:
                self.d[c] = LapplaceEstimate()
            for word in d[0]:
                self.d[c].add(word,1)

        self.total = sum(map(lambda  x:self.d[x].getsum(),self.d.keys()))

    def classify(self,x):
        tmp = {}
        for c in self.d:
            tmp[c] = log(self.d[c].getsum()) - log(self.total) # P(Y=ck)
            for word in x:
                tmp[c] += log(self.d[c].getprob(word)) # P(Xj=xj | Y=ck)
        ret,prob = 0,0
        for c in self.d:
            now = 0
            try:
                for otherc in self.d:
                    now +=exp(tmp[otherc] - tmp[c])
                now = 1/now
            except OverflowError:
                now =0
            if now > prob:
                ret,prob = c,now
        return ret,prob

class Sentiment(object):
    def __init__(self):
        self.classifier = Bayes()

    def segment(self,sent):
        words = sent.split(" ")
        return  words

    def train(self,neg_docs,pos_docs):
        data = []
        for sent in neg_docs:
            data.append([self.segment(sent),u'neg'])
        for sent in pos_docs:
            data.append([self.segment(sent),u'pos'])
        self.classifier.train(data)

    def classify(self,sent):
        return self.classifier.classify(self.segment(sent))



s = Sentiment()
s.train([u'糟糕', u'好 差劲'], [u'优秀', u'很 好'])
print(s.classify(u"好 优秀"))