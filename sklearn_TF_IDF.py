#!/usr/bin/python  
# -*- coding: utf-8 -*-
import numpy
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
reload(sys)
#sys.setdefaultencoding('utf8')了 网易 杭研 大厦","小明 硕士 毕业 与 中国 科学院","我 爱 北京 天安门"]
trainfile = open("C:\\Users\\hd\\Desktop\\docs.txt","r") #不同的documents用换行符隔开
traincorpus = trainfile.readlines()
 
#corpus=["我 来到 北京 清华大学","我 他 来到
trainfile.close()
corpus = traincorpus;
     
vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')
'''
min_df:的含义 
min_df is used for removing terms that appear too infrequently. For example: 
•min_df = 0.01 means "ignore terms that appear in less than 1% of the documents". 
•min_df = 5 means "ignore terms that appear in less than 5 documents". 
 
The default min_df is 1, which means "ignore terms that appear in less than 1 document".Thus, the default setting does not ignore any terms. 
 
max_df:的含义 
max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example: 
•max_df = 0.50 means "ignore terms that appear in more than 50% of the documents". 
•max_df = 25 means "ignore terms that appear in more than 25 documents". 
The default max_df is 1.0, which means "ignore terms that appear in more than 100% of the documents". Thus, the default setting does not ignore any terms. '''


transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tfidf=transformer.fit_transform(tfidf_vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word=tfidf_vectorizer.get_feature_names()#获取词袋模型中的所有词语
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
f = open("C:\\Users\\hd\\Desktop\\tif.txt","w+")
for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#         print("-------这里输出第",i,u"类文本的词语tf-idf权重------")
    f.write(str(i+1)+"\t")
    for j in range(len(word)):
        if(weight[i][j]>0): f.write(str(j+1) + ":" + str(weight[i][j]) + " ")
    f.write("\n")
    print i
f.close()
f = open("C:\\Users\\hd\\Desktop\\dictionary.txt","w+")
for i in range(len(word)):
    f.write(str(i) + "\t" + word[i].encode("utf-8") + "\n")
f.close()
 
SimMatrix = (tfidf * tfidf.T).A
print(SimMatrix[1,3]) #"第一篇与第4篇的相似度"
 
numpy.savetxt("C:\\Users\\hd\\Desktop\\SimMatrix.csv", SimMatrix, delimiter=",") #保存相似度矩阵