
# coding: utf-8

# In[ ]:

import csv
import string
import numpy as np
import math
from nltk.corpus import stopwords


# In[ ]:

words = {}
wordIndex = 0
punctuations = string.punctuation + '\n'


# In[ ]:

stop = set(stopwords.words('english'))


# In[ ]:

def getProperSentence(sentence):
    returnSentence = sentence.lower()
    replace_punctuation = string.maketrans(punctuations, ' '*len(punctuations))
    returnSentence = returnSentence.translate(replace_punctuation)
    return returnSentence


# In[ ]:

messageList = []
cleanedMessageList = []


# In[ ]:

def populateMessageList():
    with open('train.csv','rb') as train:
        reader = csv.reader(train)
        for row in reader:
            message =row[4]
            messageList.append(message)
            
populateMessageList()

def populateCleanedMessageList():
    with open('train.csv','rb') as train:
        reader = csv.reader(train)
        for row in reader:
            message =row[4]
            cleanedMessageList.append(getProperSentence(message))
            
populateCleanedMessageList()


# In[ ]:

wordsIndex = {}
wordCount = 0
wordCountForIdf = {}


# In[ ]:




# In[ ]:

def populateWordsIndex():
    global wordCount
    for message in cleanedMessageList:
        for word in message.split(' '):
            if not word.isdigit():
                if word not in wordsIndex:
                    wordsIndex[word] = wordCount
                    wordCount += 1

populateWordsIndex()


# In[ ]:

def populateWordCountForIdf():
    for word in wordsIndex:
        count = 0
        for message in cleanedMessageList:
            if word in message:
                count += 1
        wordCountForIdf[word] = count
        
populateWordCountForIdf()


# In[ ]:

stopWordsRemovedCount = 0
stopWordsRemovedIndex = {}
for word in wordsIndex:
    if word not in stop:
        stopWordsRemovedIndex[word] = stopWordsRemovedCount
        stopWordsRemovedCount += 1


# In[ ]:




# In[ ]:

def tf(givenWord, message): #Assuming cleaned message
    count = 0
    for word in message.split(' '):
        if givenWord == word:
            count += 1.0 
    return count

def idf(givenWord):
    N = len(messageList)
    n = wordCountForIdf[givenWord]
    return math.log(N*1.0/(1+n))


# In[ ]:

def makeFeatureVector(sentence):
    tempSen = sentence
    retVal = np.zeros(wordCount)
    for word in tempSen.split(' '):
        if word.isdigit():
            continue
        indexOfWord = wordsIndex[word]
        retVal[indexOfWord] = tf(word,tempSen)*idf(word)
    return retVal

def makeFeatureVectorWithoutStopwords(sentence):
    tempSen = sentence
    retVal = np.zeros(stopWordsRemovedCount)
    for word in tempSen.split(' '):
        if word.isdigit() or word in stop:
            continue
        indexOfWord = stopWordsRemovedIndex[word]
        retVal[indexOfWord] = tf(word,tempSen)*idf(word)
    return retVal
    


# In[ ]:

def returnFeatureSet(makeFeatureFunction):
    index = 0
    ret = np.array([])
    for message in cleanedMessageList:        
        featureVector = makeFeatureFunction(message)
#             print featureVector
        if index == 0:
            ret = np.hstack((ret,featureVector))
            index = index + 1
        else:
            ret = np.vstack((ret, featureVector))
    return ret


# In[ ]:

featureVectorSet = returnFeatureSet(makeFeatureVector)
stopwordLessFeatureVectorSet = returnFeatureSet(makeFeatureVectorWithoutStopwords)


# In[ ]:

from sklearn.cluster import KMeans


# In[ ]:

kmeans = KMeans(n_clusters=20, random_state=0).fit(stopwordLessFeatureVectorSet)


# In[ ]:

def getClusters(clusterIndices, messageList, requiredIndex):
    retVal = []
    for i in range(len(clusterIndices)):
        if clusterIndices[i] == requiredIndex:
            retVal.append(messageList[i])
    return retVal


# In[ ]:

getClusters(kmeans.labels_,cleanedMessageList,3)


# In[ ]:

[(i,len(getClusters(kmeans.labels_,messageList,i))) for i in range(20)]


# In[ ]:



