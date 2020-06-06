# -*- coding: utf-8 -*-
"""
functions for performing clustering, like:
    - data preprocessing function
    - function to get data from json
    - function to get BERT embeddings
    - function to perform clustering and get summaries
    - function to evaluate results
"""

#import libraries
import json
import os
import re
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn import metrics
from konlpy.tag import Kkma

#import file for getting bert embeddings
import getting_bert_embeddings as gbe

#a function to load the dataset in json format
#gets path to the folder with data
#returns texts and their corresponding summaries
def get_files_func(data_folder): 
    data_files = os.listdir(data_folder)
    
    data_text_list = []
    data_res_list = []
    
    for file in data_files:
        with open(os.path.join(data_folder, file), 'r', encoding = "utf-8") as myfile:
            data=myfile.read()
        obj = json.loads(data)
        
        obj_to_list = list(obj.items())
        data_text = obj_to_list[4][1]
        data_test_res = obj_to_list[5][1]
        data_text_list.append(data_text)
        data_res_list.append(data_test_res)
    
    return data_text_list, data_res_list

#a function to split korean text into sentences with konlpy and "clean" it
#(in case of processing text out of dataset) 
def prepare_text_func(full_text):
    
    kkma = Kkma()
    #split text to sentences
    splitted_text = kkma.sentences(full_text)
    
    #clean sentences
    cleaned_text = []
    for sent in splitted_text:
        sent = re.sub(r'https?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE)
        sent = re.sub(r'\<a href', ' ', sent)
        sent = re.sub(r'&amp;', '', sent) 
        sent = re.sub(r'[_"\-;%()|+&=*%.,!?:…#·$@\[\]/]', ' ', sent)
        sent = re.sub(r'[“‘”’<>◆【】▲■ㆍ©․—∙�‧⋅•･▶・｢｣˚´゛〜–⁺ⓒ]', ' ', sent)
        sent = sent.strip()
        cleaned_text.append(sent)
    
    return cleaned_text
        
#function to get BERT sentece embeddings
#gets a text in form of list of sentences
def get_sentence_embeddings(sentences_list, dim=768):
    embedded_sentences = []
    
    for i in range(len(sentences_list)): #going through the text
        embeddings = gbe.get_features(["" + sentences_list[i] + ""], dim=dim) #get BERT embeddings
        embeddings_list = list(embeddings.items()) #turn BERT embeddings into list
        
        sentence_vector = np.zeros(dim)
        for j in range(1, (len(embeddings_list)-1)): #because need to cut off [CLS] and [SEP] tokens
            sentence_vector = sentence_vector + embeddings_list[j][1] #add all tokens' vectors
        sentence_vector_mean = sentence_vector/(len(sentence_vector)-2) #devide by the num of tokens minus tokens [CLS] and [SEP]
        
        embedded_sentences.append(sentence_vector_mean)
        
        print("------- Finished sentence number " + str(i+1) + " out of " + str(len(sentences_list)) + " -----")
        
    return embedded_sentences #return a list of BERT embeddings for each sentence in the text
            

#function for F1 score calculation
#gets ground truth prediction and system prediction
#returns f1 score value
def eval_func(y_true: list, y_pred: list):
    tp = 0 #true positive
    fp = 0 #false positive
    fn = 0 #false negative
    for idx in (y_true):
        if idx in y_pred:
            tp += 1
        elif idx not in y_pred:
            fn += 1
    for idx in (y_pred):
        if idx not in y_true:
            fp += 1
    
    #calculate precision and recall
    if tp != 0:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
    else:
        precision = 0
        recall = 0
     
    #calculate f1 score
    if precision and recall != 0:
        f1 = (2*(precision*recall))/(precision + recall)
    else:
        f1 = 0
        
    return f1

#K-Means clustering
def kmeans_clust(sentence_embeddings, data_text, data_test_res):
    all_summaries = []
    all_doc_acc_s = []
    
    #going through each text
    for k in range (len(sentence_embeddings)):
        
        max_ch_idx = 0
        max_n_clusters = 12 #to find best number of clusters
        
        #if the length of the text is shorter than max number of possible clusters
        if len(sentence_embeddings[k]) < max_n_clusters:
            max_n_clusters = len(sentence_embeddings[k])
        
        #finding best number of clusters with Calinski Harabasz score (variance ratio criterion)
        for i in range (2,max_n_clusters):
            kmeans = KMeans(n_clusters=i)
            kmeans = kmeans.fit(sentence_embeddings[k])
            labels = kmeans.labels_
        
            ch_idx = metrics.calinski_harabaz_score(sentence_embeddings[k],labels)
            
            if ch_idx > max_ch_idx:
                max_ch_idx = ch_idx
                best_kmeans = kmeans
        
        centers = best_kmeans.cluster_centers_
        labels = best_kmeans.labels_
        
        #finding closest to cluster centers sentences
        closest_sent, _ = pairwise_distances_argmin_min(centers, sentence_embeddings[k])
        sorted_summary = np.sort(closest_sent)
        
        #turning sentence index predictions into actual texts
        final_sum_idx = []
        for sent in sorted_summary:
            final_sum_idx.append(data_text[k][sent])
        
        final_summary = '. '.join(final_sum_idx)
        all_summaries.append(final_summary)
        
        #calculating f1 score for the text
        f1_score = eval_func(data_test_res[k], list(closest_sent))
        all_doc_acc_s.append(f1_score)   
        
    #calculating the overall f1 score        
    final_accuracy = sum(all_doc_acc_s)/50
    
    return final_accuracy, all_summaries, all_doc_acc_s

