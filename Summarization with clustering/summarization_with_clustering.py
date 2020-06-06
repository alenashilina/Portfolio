# -*- coding: utf-8 -*-
"""
performming text summarisation using K-means clustering
"""

#import libraries
import clustering_functions as clf #file containing all necessary functions for summarization
import pickle

#path to folder containing data in json format
data_folder = ('PATH')

#getting texts and corresponding summaries
data_text_list, data_res_list = clf.get_files_func(data_folder)

#getting BERT embeddings
texts_bert_embeddings = []
for k, text in enumerate(data_text_list):
    embedded_text = clf.get_sentence_embeddings(text)
    texts_bert_embeddings.append(embedded_text)
    print ("--------------------Finished text "+ str(k) + " -------------")
    
print("Finished getting BERT embeddings for all texts.")

#saving BERT embeddings into pickle file
with open('texts_bert_embeddings.pickle', 'wb') as f:
    pickle.dump(texts_bert_embeddings, f)

#getting texts' summaries
accuracy, summaries, all_docs_acc = clf.kmeans_clust(texts_bert_embeddings, data_text_list, data_res_list)

print("Final accuracy: " + str(accuracy))