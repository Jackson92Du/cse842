# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:52:33 2017

@author: zhangr12
"""

#build models

import os
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm


#n is the number of grams
def n_gram(document, n):
    document = re.sub('\s+', '_', document)
    new_doc = ""
    for i in range(0, len(document) - n):
        new_doc += document[i:i+n]
        new_doc += " "
    return new_doc

num_gram = 5

new_en = n_gram(udhr_entxt, num_gram)
new_ru = n_gram(udhr_rutxt, num_gram)
new_ind = n_gram(udhr_indtxt, num_gram)
new_fr = n_gram(udhr_frtxt, num_gram)
new_pt = n_gram(udhr_pttxt, num_gram)
new_es = n_gram(udhr_estxt, num_gram)
new_ar = n_gram(udhr_artxt, num_gram)

train = []
train.append(new_ar)
train.append(new_en)
train.append(new_es)
train.append(new_fr)
train.append(new_ind)
train.append(new_pt)
train.append(new_ru)


new_en = n_gram(entxt, num_gram)
new_ru = n_gram(rutxt, num_gram)
new_ind = n_gram(idtxt, num_gram)
new_fr = n_gram(frtxt, num_gram)
new_pt = n_gram(prtxt, num_gram)
new_es = n_gram(estxt, num_gram)
new_ar = n_gram(artxt, num_gram)

train.append(new_ar)
train.append(new_en)
train.append(new_es)
train.append(new_fr)
train.append(new_ind)
train.append(new_pt)
train.append(new_ru)


#1 = en
#2 = ru
#3 = ind
train_label = [1,2,3,4,5,6,7,1,2,3,4,5,6,7]
target_name = ["ar", "en", "es", "fr", "id", "pt", "ru", "ar", "en", "es", "fr", "id", "pt", "ru"]


#train the model with pipline

#text_clf = Pipeline([('vect', CountVectorizer()),
#                     ('tfidf', TfidfTransformer()),
#                     ('clf', MultinomialNB(alpha = 0.5)),
#])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#use this part if want to use NB model

#text_clf = text_clf.fit(train, train_label)

clf_NB = MultinomialNB().fit(X_train_tfidf, train_label)
clf_LR = LogisticRegression().fit(X_train_tfidf, train_label)
clf_SVM = SGDClassifier(loss='hinge', penalty='l1', alpha=1e-3, n_iter=5).fit(X_train_tfidf, train_label)
clf_kernel_svm = svm.SVC(kernel='rbf').fit(X_train_tfidf, train_label)


from urllib.parse import urlparse

def prunetweet(string):
    new_string = ''
    for i in string.split():
        s, n, p, pa, q, f = urlparse(i)
        if s and n:
            pass
        elif i[:1] == '@':
            pass
        elif i[:1] == '#':
            new_string = new_string.strip() + ' ' + i[1:]
        else:
            new_string = new_string.strip() + ' ' + i
    return emoji_pattern.sub(r'', new_string)

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

#get test data
files = [f for f in os.listdir('twitter-120k/')]

#loading data
samples = []

for i in range(0, len(files)):
    crt_file = os.path.join("twitter-120k/", files[i])
    f = pickle.load( open( crt_file, "rb" ) )
    for line in f:
        samples.append(line)

test_label = []
test_data = []
#clean data
for sample in samples:
    if sample[0] in target_name:
        test_data.append(n_gram(prunetweet(sample[1]), 5))
        test_label.append(target_name.index(sample[0]) + 1)


X_new_counts = count_vect.transform(test_data)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


NB_predicted = clf_NB.predict(X_new_tfidf)
LR_predicted = clf_LR.predict(X_new_tfidf)
SVM_predicted = clf_SVM.predict(X_new_tfidf)
kernel_svm_predicted = clf_kernel_svm.predict(X_new_tfidf)


print(metrics.classification_report(test_label, NB_predicted, target_names=target_name))
print(metrics.classification_report(test_label, LR_predicted, target_names=target_name))
print(metrics.classification_report(test_label, SVM_predicted, target_names=target_name))
print(metrics.classification_report(test_label, kernel_svm_predicted, target_names=target_name))


'''
mix_predicted = []
for i in range(len(NB_predicted)):
    #use LR as major classifier, use NB to classify Arabic
    if NB_predicted[i] == 1:
        mix_predicted.append(1)
    else:
        mix_predicted.append(LR_predicted[i])

print(metrics.classification_report(test_label, mix_predicted, target_names=target_name))
'''


#create 10 multilabel test samples
#mix English and Russian
test_num = 2
find_en = 0
find_ru = 0
mix_en = []
mix_ru = []
#find first 100 samples for English and Russian, store them in mix_en and mix_ru
for item in samples:
    if item[0] == 'en' and find_en < test_num:
        find_en += 1
        #print (item[1])
        mix_en.append(item[1])
        #mix_sample += ' '
    if item[0] == 'ru' and find_ru < test_num:
        find_ru += 1
        #print (item[1])
        mix_ru.append(item[1])

#combine the items in mix_en and mix_ru to create 100 mix_samples
mix_sample = []
for i in range(len(mix_en)):
    if mix_en[i][0] == '"':
        mix_en[i] = mix_en[i][1:-1]
    if mix_ru[i][0] == '"':
        mix_ru[i] = mix_ru[i][1:-1]
    mix_sample.append(n_gram(prunetweet(mix_en[i] + ' ' + mix_ru[i]), 5))
    
    
    
#transform mixed text message in mix_sample
test_mix = []
for j in range(len(mix_sample)):
    temp_mix = []
    for i in range(len(mix_sample[j].split())):
        temp = ""
        for k in range(i + 1):
            temp += mix_sample[j].split()[k]
            temp += " "
        temp_mix.append(temp[:-1])
    test_mix.append(temp_mix)


log_prob = []
for i in range(len(test_mix)):
    log_prob_temp = []
    for j in range(len(test_mix[i])):
        X_new_counts = count_vect.transform([test_mix[i][j]])
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        log_prob_temp.append(clf_NB.predict_log_proba(X_new_tfidf))
    log_prob.append(log_prob_temp)



#start from here!!!
plot_prob = []
for k in range(len(log_prob)):
    plot_prob_temp = []    
    for j in range(7):
        temp = []
        for i in range(len(log_prob[k])):
            temp.append(log_prob[k][i][0][j])
        plot_prob_temp.append(temp)
    plot_prob.append(plot_prob_temp)


#plt.axis([1,128,-1.5, 0])
plot_num = 0
for i in range(len(plot_prob[plot_num])):
    plt.plot(plot_prob[plot_num][i], label=target_name[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.legend()

plt.show()

predict_switch = []
for j in range(len(plot_prob)):
    count_times = 0
    for i in range(1, len(plot_prob[j][-1])):
        #print (plot_prob[j][-1][i-1] - plot_prob[j][-1][i])        
        if plot_prob[j][-1][i] - plot_prob[j][-1][i-1] > 0.015:
            predict_switch.append(i)
            break
        if plot_prob[j][-1][i] - plot_prob[j][-1][i-1] > 0.0002:
            count_times += 1
        #continuous increasing for 3 times
        if count_times == 3:
            predict_switch.append(i-3)
            break
    

true_switch = []
for i in range(len(plot_prob)):
    true_switch.append(len(plot_prob[i][0]) - len(prunetweet(mix_ru[i])) + 1)
#true_switch = len(plot_prob[0]) - len(prunetweet(mix_sample2)) + 1












