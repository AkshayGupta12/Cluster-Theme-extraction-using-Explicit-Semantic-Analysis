# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


import json
import math
import os
import pickle

import gensim
import matplotlib.pyplot as plt
# import cPickle
import nltk
# import itertools.chain
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as csim
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors, TfidfModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# %%


# THE FOLLOWING ACTIONS OCCUR AT WIKIPEDIA (KNOWLEDGE SOURCE) SIDE #


# %%


stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


# %%


def cleanData(data):
	"""
		Input   : A string.
		Returns : Cleaned set of tokens.
	"""
	tokens = [token.lower() for token in word_tokenize(data)]
	cleaned_tokens = [token for token in tokens if (token.isalnum() and ps.stem(token) not in stop_words) ]
	return cleaned_tokens


# %%


def generateDataSet(directory):
	"""
		Input   : The directory containing the wikipedia dump.
		Returns : List of lists, where each sublist contains the cleaned tokens for an article.
	"""
	dataset = []
	id2concept = {}
	index = 0
	for filename in (os.listdir(directory)):
	    if filename.endswith(".raw"):
	    	file = os.path.join(directory, filename)
	    	with open(file) as json_file:
	    		concepts = json.load(json_file)
	    		for concept in concepts.keys():
	    			id2concept[index] = concept
	    			index += 1
	    			data = cleanData(concepts[concept]['text'])
	    			dataset.append(data)
	return (dataset,id2concept)


# %%


def getTFIDFModel(dataset):
	"""
		Input   : The dataset as a list of lists (as returned by generateDataSet() function).
		Returns : TF-IDF Model, the dictionary and the corpus corresponding to the dataset.
	"""
	id2word = Dictionary(dataset)
	corpus  = [id2word.doc2bow(line) for line in dataset]
	model   = TfidfModel(corpus)
	return (model,id2word,corpus)


# %%


def getESA(model,corpus):
    """
        Input   : the TF-TDF Model and corpus.
        Returns : The inverted index.
    """
    inverted_index = {}
    for i in (range(len(corpus))):
        tmp = dict(model[corpus[i]])
        for key in tmp.keys():
            p = tmp[key]
            word = id2word[key]
            try:
                inverted_index[key].append((i,p))
            except:
                inverted_index[key] = []
                inverted_index[key].append((i,p))
    return inverted_index


# %%


# THE FOLLOWING ACTIONS OCCUR AT DOCUMENT SIDE USING WIKIPEDIA PROCESSED OUTPUTS #


# %%


def getDocumentsData(dir_path,num_docs):
    """
        Input   : Path to directory where formatted documents are stored and number of documents in each category.
        Returns : cleaned document content list and their corresponding labels.
    """    
    files = os.listdir(dir_path)
    
    label = []
    all_data = []
    
    for file in files:
        filepath = dir_path+"\\"+file
        f = open(filepath)
        data = f.readlines()
        f.close()
        for i in tqdm(range(num_docs)):
            all_data.append(cleanData(data[i].split(",")[1]))
            label.append(data[i].split(",")[0])
    
    return (all_data,label)


# %%


def getDocumentsModel(filepath, num_docs):
    data, labels = getDocumentsData(filepath, num_docs)
    with open(saved_model_url+"doc_data", "wb") as fp:
	    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(saved_model_url+'doc_labels', 'wb') as fp1:
        pickle.dump(labels, fp1, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open(saved_model_url+"doc_data", "rb") as fp:
	#     data = pickle.load(fp)
    # with open(saved_model_url+'doc_labels', 'rb') as fp1:
	#     labels = pickle.load(fp1)
    tf_idf_model_doc, id2word_doc, corpus_doc = getTFIDFModel(data)
    return (tf_idf_model_doc, id2word_doc, corpus_doc, labels,data)


# %%



def transformToWikipediaBOWList(bow_list,id2word,id2word_doc):
    """
        Input   : BOW list, mapping from ID to word in the Wikipedia, and mapping from ID to word in Docs. 
        Returns : BOW List with IDs from Wikipedia mapping.
    """
    transformed_bow_list = []
    for pair in bow_list:
        word = id2word_doc[pair[0]]
        word_bow_pair = id2word.doc2bow([word])
        if len(word_bow_pair)>0:
            transformed_bow_list.append((word_bow_pair[0][0],pair[1]))
    return transformed_bow_list


# %%


def getConcept(bow_list_entry,inverted_index):
    """
        Input   : BOW List entry
        Returns : New BOW List with 1st element as the concept, and 2nd element as the strength of association with it.
    """
    index = bow_list_entry[0]
    value = bow_list_entry[1]
    concept_list = []
    try:
        wiki_index   = inverted_index[index]
        for k in wiki_index:
            concept_list.append((k[0], k[1]*value))
        return concept_list
    except:
        return concept_list


# %%


def mergeConcept(bow_list,id2word,id2word_doc,inverted_index):
    """
        Input   : BOW List.
        Returns : Concept List with same concepts merged together.
    """
    bow_list = transformToWikipediaBOWList(bow_list,id2word,id2word_doc)
    concept_list = []
    for bow_list_entry in bow_list:
        concepts = getConcept(bow_list_entry,inverted_index)
        if len(concepts)>0:
            concept_list.append(concepts)
            
    merged_concepts = {}
    for concept in concept_list:
        for pair in concept:
            try:
                merged_concepts[pair[0]]+=pair[1]
            except:
                merged_concepts[pair[0]]=pair[1]
    return merged_concepts


# %%


def getInputDocumentVector(filepath,id2word,inverted_index,num_docs):
    """
        Input   : Path to formatted documents, wiki_mapping from ID to word, ESA Model, number of test documents for each category.
        Returns : ESA Vectors for each document.
    """
    tf_idf_model_doc, id2word_doc, corpus_doc, labels,data = getDocumentsModel(filepath, num_docs)
    doc_vecs = []
    for i in tqdm(range(len(data))):
        bow = id2word_doc.doc2bow(data[i])
        bow_rep = tf_idf_model_doc[bow]
        doc_vecs.append({k: v for k, v in sorted(mergeConcept(bow_rep,id2word,id2word_doc,inverted_index).items(), key=lambda item: item[1], reverse = True)})
    return (doc_vecs,labels)


# %%


# THE FOLLOWING CODES ARE FOR ANALYSIS PART #6


# %%


def getConceptNames(dict_vec,id2concept):
    """
        Input   : dictionary vector and mapping from ID to concept.
        Returns : List of associated concepts.
    """
    concepts = []
    for pair in dict_vec:
        concepts.append(id2concept[pair])
    return concepts


# %%


def getTopClusterConcepts(min_vec,id2concept,top_k):
    """
        Input   : Minimum vector, ID to concept mapping, number of top concepts to be retrieved
        Returns : top_k number of top concepts.
    """
    sorted_list = np.argsort(-1*min_vec)
    top_concepts = []
    for i in range(top_k):
        top_concepts.append(id2concept[sorted_list[i]])
    return top_concepts


# %%


# THE FOLLOWINGS CODES ARE FOR CLUSTERING #


# %%


def normalizeVectors(vecs):
    """
        Input   : List of vectors.
        Returns : List of normalized Vectors.
    """
    for i in range(len(vecs)):
        mag = np.linalg.norm(vecs[i])
        if(mag != 0):
            vecs[i] = vecs[i]/mag
    return vecs


# %%


def getVectorsFromDictionary(dict_vec,size):
    """
        Input   : Dictionary representation of a vector.
        Returns : Its vector represenetation.
    """
    vec = np.zeros(size)
    for i in dict_vec:
        vec[i] = dict_vec[i]
    return vec


# %%


def getKMeansLabels(vecs,num_clusters):
    """
        Inputs  : The vectors and number of clusters.
        Returns : A List indicating cluster to which the corresponding vector belongs.
    """
    kmeans = KMeans(n_clusters=num_clusters,init='k-means++',max_iter=1000, n_init=20, random_state=1)
    kmeans.fit(vecs)
    return kmeans.labels_


# %%


def getGMMLabels(vecs,num_clusters):
    """
        Inputs  : The vectors and number of clusters.
        Returns : A List indicating cluster to which the corresponding vector belongs.
    """
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(vecs)
    return gmm.predict(vecs)


# %%


def getClusterAssignments(doc_vecs,size,num_clusters,algorithm):
    """
        Input   : The vectors, number of concepts and number of clusters.
        Returns : A List indicating cluster to which the corresponding vector belongs.
    """
    all_vecs = []
    for i in range(len(doc_vecs)):
        all_vecs.append(getVectorsFromDictionary(doc_vecs[i],size))
    all_vecs = normalizeVectors(all_vecs)
    if(algorithm=="k_means"):
        assigned_labels = getKMeansLabels(all_vecs,num_clusters)
    elif(algorithm=="gmm"):
        assigned_labels = getGMMLabels(all_vecs,num_clusters)
    else:
        assigned_labels = []
    return assigned_labels


# %%


def groupClusterVectorsTogether(vecs,assigned_labels):
    """
        Input   : Vectors and a List indicating cluster to which the corresponding vector belongs.
        Returns : A list of lists of vectors, where each sublist contains the vectors belonging to the same cluster. 
    """
    num_clusters = len(set(assigned_labels))
    sorted_vecs = []
    for cluster_num in range(num_clusters):
        curr_list = []
        for i in range(len(assigned_labels)):
            if(assigned_labels[i]==cluster_num):
                curr_list.append(vecs[i])
        sorted_vecs.append(curr_list)
    return sorted_vecs


# %%


def plotKMeansForDiffK(doc_vecs,min_k,max_k):
    """
        Input    : The document dictionaries, minimum value of k, and maximum value of k.
        Displays : A curve showing the variation of 'sum of squared errors' with different values of k. 
    """
    all_vecs = []
    for i in range(len(doc_vecs)):
        all_vecs.append(getVectorsFromDictionary(doc_vecs[i],num_concepts))
    all_vecs = normalizeVectors(all_vecs)
    
    
    # test_cluster = [1, 25, 50, 75, 100, 125, 150]
    # test_cluster =  np.arange(5, 25, 5).tolist()
    sum_of_sq_error = []
    test_cluster = np.arange(5, 25, 5).tolist()
    test_cluster = [1]+test_cluster
    for i in test_cluster:#range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(all_vecs)
        sum_of_sq_error.append(kmeans.inertia_)
        
    return test_cluster, sum_of_sq_error


# %%
def silhouetteScore(doc_vecs,test_cluster):
    """
        Input    : The document dictionaries, minimum value of k, and maximum value of k.
        Displays : A curve showing the variation of 'sum of squared errors' with different values of k. 
    """
    all_vecs = []
    for i in range(len(doc_vecs)):
        all_vecs.append(getVectorsFromDictionary(doc_vecs[i], num_concepts))
    all_vecs = normalizeVectors(all_vecs)
    # test_cluster = [1, 25, 50, 75, 100, 125, 150]
    # test_cluster =  np.arange(5, 25, 5).tolist()
    sil = []
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2

    for k in test_cluster:
        kmeans = KMeans(n_clusters=k).fit(all_vecs)
        labels = kmeans.labels_
        sil.append(silhouette_score(all_vecs, labels, metric='euclidean'))

    return test_cluster, sil

# THE FOLLOWING CODES ARE FOR DIFFERENT HEURISTICS TO DECIDE THE REPRESENTATIVE VECTOR FOR A CLUSTER #


# %%


def getMinimumVectorForACluster(vecs,size):
    """
        Input   : Vectors belonging to the same cluster, number of concepts.
        Returns : The minimum vector. ( Refer to report for definition of minimum vector )
    """
    vecs = np.sort(np.asarray(vecs).T)
    min_vec = vecs.T[0]
    return min_vec


# %%


def getMeanVectorForACluster(vecs,size):
    """
        Input   : Vectors belonging to the same cluster, number of concepts.
        Returns : The mean vector.
    """
    return np.sum(np.asarray(vecs),axis=0)/len(vecs)


# %%


def getMedianVectorForACluster(vecs,size):
    """
        Input   : Vectors belonging to the same cluster, number of concepts.
        Returns : The median vector.
    """
    vecs = np.sort(np.asarray(vecs).T)
    median_vec = np.zeros(size)
    for i in range(len(median_vec)):
        median_vec[i] = np.median(vecs[i])
    return median_vec


# %%


def getMinimumVectorWithoutOutliersForACluster(vecs,size,margin,frac_region):
    """
        Input   : Vectors belonging to the same cluster, number of concepts, the margin for a dimension value not to be an outlier.
        Returns : The minimum vector taking into account the possibility of outliers.
    """
    vecs = np.sort(np.asarray(vecs).T)
    num_of_vals = math.floor(len(vecs[0])*frac_region)
    vecs = vecs.T[0:num_of_vals].T
    min_vec_no_outlier = vecs.T[0]
    
    for i in range(len(vecs)):
        for j in reversed(range(len(vecs[i])-1)):
            if(vecs[i][j]!=0):
                frac_diff = (vecs[i][j+1] - vecs[i][j])/vecs[i][j]
                if(frac_diff >= margin):
                    min_vec_no_outlier[i] = vecs[i][j+1]
                    break
    return min_vec_no_outlier


# %%


# LATENT DIRICHLET ANALYSIS #


# %%


def cleanDataForLDA(text):
    """
        Input   : A string.
        Returns : Cleaned set of tokens for LDA
    """
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(ps.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
    return result


# %%


def getDocumentsDataForLDA(dir_path, num_docs):
    """
        Input   : Path to directory where formatted documents are stored and number of documents in each category.
        Returns : cleaned document content list for LDA and their corresponding labels.
    """    
    files = os.listdir(dir_path)
    
    label = []
    all_data = []
    
    for file in files:
        filepath = dir_path+"\\"+file
        f = open(filepath)
        data = f.readlines()
        f.close()
        for i in range(num_docs):
            all_data.append(cleanDataForLDA(data[i].split(",")[1]))
            label.append(data[i].split(",")[0])
    
    return (all_data,label)


# %%


def getDocumentsModelForLDA(filepath, num_docs):
    """
        Input   : The path to the input files directory.
        Returns : The TF_IDF Model, mapping from ID to word type, corpus and labels for data.
    """
    data, labels = getDocumentsDataForLDA(filepath, num_docs)
    tf_idf_model_doc, id2word_doc, corpus_doc = getTFIDFModel(data)
    return (tf_idf_model_doc, id2word_doc, corpus_doc, labels)


# %%


def getLDAModel(corpus_doc,k,id2word_doc,num_passes,num_workers):
    """
        Input   : the BOW corpus, number of topics, mapping from ID to word, num_passes, and num_workers.
        Returns : The LDA Model.
    """
    # lda_model = gensim.models.LdaMulticore(corpus_doc, num_topics=k, id2word=id2word_doc, passes=num_passes, workers=num_workers)
    lda_model = gensim.models.LdaModel(corpus_doc, num_topics=k, id2word=id2word_doc, passes=num_passes)

    return lda_model


# %%
def printLDATopics(lda_model,top_k):
    """
        Input   : The LDA Model.
        Prints  : The LDA Topics.
    """
    for idx, topic in lda_model.print_topics(num_topics=50):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    return 0


# %%
def mergeConceptTc(bow_list,inverted_index):
    concept_list = []
    for bow_list_entry in bow_list:
        concepts = getConcept(bow_list_entry,inverted_index)
        if len(concepts)>0:
            concept_list.append(concepts)
            
    merged_concepts = {}
    for concept in concept_list:
        for pair in concept:
            try:
                merged_concepts[pair[0]]+=pair[1]
            except:
                merged_concepts[pair[0]]=pair[1]
    return merged_concepts


# %%
def getTopicVectorTC(data,id2word,inverted_index):
    doc_vecs = []
    bow_rep = id2word.doc2bow(data)
    doc_vecs.append({k: v for k, v in sorted(mergeConceptTc(bow_rep,inverted_index).items(), key=lambda item: item[1], reverse = True)})
    return (doc_vecs)


# %%
def tcAuto(vecs,id2concept,top_k,id2word,inverted_index,num_concepts):
    top_concepts_all_clusters = []
    cos_sim_total=[]
    total=0
    for vec in vecs:
        top_concepts_all_clusters.append(getTopClusterConcepts(vec,id2concept,top_k))
#     getTopicVectorTC(top)
    for top_concepts in top_concepts_all_clusters:
        cos_sim_cluster=[]
        concepts_vector = []
        for concept in top_concepts:
            concept = cleanData(concept)
            if(len(concept)!=0):
                # print(concept)
                concept_dict = getTopicVectorTC(concept,id2word,inverted_index)[0]
                concepts_vector.append(getVectorsFromDictionary(concept_dict, num_concepts))

        concepts_vector = normalizeVectors(concepts_vector)

        for i in range(len(concepts_vector)):
            concept_i_to_array = np.array(concepts_vector[i])
            concept_i_to_array = np.reshape(concept_i_to_array, (1,-1))
            for j in range(i+1,len(concepts_vector)):
                concept_j_to_array = np.array(concepts_vector[j])
                concept_j_to_array = np.reshape(concept_j_to_array, (1,-1))
                cos_sim = csim.cosine_similarity(concept_i_to_array,concept_j_to_array)
                # if(cos_sim[0][0] > 0.7):
                cos_sim_cluster.append(cos_sim[0][0])
                total = total + cos_sim[0][0]
        cos_sim_total.append(cos_sim_cluster)
    
    tc_auto_sum = sum(map(sum, cos_sim_total))
    tc_auto_average = tc_auto_sum/len(cos_sim_total)
    # print(tc_auto_sum,tc_auto_average)
    # print(total)
    return tc_auto_average

    


# %%


# DRIVER CODE #


# %%


# THE FOLLOWING CODE DEALS WITH WIKIPEDIA SIDE #
# %%
# directory = "./ESA_input/NIPS_Subset_depth_2"
directory = "./ESA_input/NIPS"
# directory = "./ESA_input/Newsgroup"
# saved_model_url = "./SavedModels/NIPS_subsetdepth_2/"
saved_model_url = "./SavedModels/"
# saved_model_url = "./SavedModels/Newsgroup/"

dataset, id2concept = generateDataSet(directory)
num_concepts = len(dataset)


# %%


tf_idf_model, id2word, corpus = getTFIDFModel(dataset)


# %%

esa = getESA(tf_idf_model,corpus)


# %%


# THE FOLLOWING CODE SAVES AND LOADS THE MODELS #


# %%

tf_idf_model.save(saved_model_url+'TF_IDF_Model_v'+str(directory[-1]))
np.save(saved_model_url+'TF_IDF_Model_v'+str(directory[-1]), id2concept)
id2word.save(saved_model_url+'ID_2_Word_v'+str(directory[-1]))
# np.save(saved_model_url+'ESA_v'+str(directory[-1]), esa)
with open(saved_model_url+'ESA_v'+str(directory[-1]), "wb") as fp3:
    pickle.dump(esa, fp3, protocol=pickle.HIGHEST_PROTOCOL)
with open(saved_model_url+'dataset_v'+str(directory[-1]), "wb") as fp:
    pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(saved_model_url+'id2concept_v'+ str(directory[-1]), 'wb') as fp1:
    pickle.dump(id2concept, fp1, protocol=pickle.HIGHEST_PROTOCOL)




# %%


# THE FOLLOWING CODE DEALS WITH THE DOCUMENT SIDE, AND USES ESA MODEL #


# %%


num_docs = 7241
# input_dir = r"./Doc_inputs/Newsgroup"
input_dir = r"./Doc_inputs/NIPS"
# print(id2concept)
doc_vecs, labels = getInputDocumentVector(input_dir,id2word,esa,num_docs)

with open(saved_model_url+"doc_vecs_v"+str(directory[-1]), "wb") as fp2:
    pickle.dump(doc_vecs, fp2, protocol=pickle.HIGHEST_PROTOCOL)
with open(saved_model_url+'doc_vec_labels_v' + str(directory[-1]), 'wb') as fp3:
    pickle.dump(labels, fp3, protocol=pickle.HIGHEST_PROTOCOL)



# %%
num_clusters = 50
algo = "k_means"
# # algo = "gmm"
assigned_labels = getClusterAssignments(doc_vecs,num_concepts,num_clusters,algo)
with open(saved_model_url+"assigned_labels_50", "wb") as fp:
    pickle.dump(assigned_labels, fp, protocol=pickle.HIGHEST_PROTOCOL)


# %%

clustered_docs = groupClusterVectorsTogether(doc_vecs,assigned_labels)
for i in range(num_clusters):
    for j in range(len(clustered_docs[i])):
        clustered_docs[i][j] = getVectorsFromDictionary(clustered_docs[i][j],num_concepts)


# %%
top_k = 20
min_vecs = []
for i in range(num_clusters):
    min_vecs.append(getMinimumVectorForACluster(clustered_docs[i],num_concepts))
for min_vec in min_vecs:
    print(getTopClusterConcepts(min_vec,id2concept,top_k))

min_vecs_tc = tcAuto(min_vecs, id2concept, top_k, id2word, esa, num_concepts)






# %%


top_k = 20
mean_vecs = []
for i in range(num_clusters):
    mean_vecs.append(getMeanVectorForACluster(clustered_docs[i],num_concepts))
for mean_vec in mean_vecs:
    print(getTopClusterConcepts(mean_vec,id2concept,top_k))

mean_vecs_tc = tcAuto(mean_vecs,id2concept,top_k,id2word,esa,num_concepts)




# %%


top_k = 20
median_vecs = []
for i in range(num_clusters):
    median_vecs.append(getMedianVectorForACluster(clustered_docs[i],num_concepts))
for median_vec in median_vecs:
    print(getTopClusterConcepts(median_vec,id2concept,top_k))
median_vecs_tc = tcAuto(median_vecs,id2concept,top_k,id2word,esa,num_concepts)






# %%
print("Min vecs Topic Coherence:", min_vecs_tc)
print("Mean vecs Topic Coherence:", mean_vecs_tc)
print("Median vecs Topic Coherence:", median_vecs_tc)
