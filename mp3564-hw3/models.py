from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk
import io 
nltk.download('punkt')
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import re 
import scipy
from nltk.tokenize import RegexpTokenizer
import numpy as np 
import scipy.sparse 
import pickle as pickle 
from scipy.sparse.linalg import svds 
import math 
# I will be using 1 late day 
# mp3564

# Preprocessing 
with io.open('data/brown.txt', 'r') as dataset:
    data  = dataset.readlines()  
    tokenizer = RegexpTokenizer(r'\w+')
    sent_all = [tokenizer.tokenize(datapoint.lower()) for datapoint in data]

    unique_set = set(word for sent in sent_all for word in sent)
    unique = sorted(list(unique_set))
    vocab_size = len(unique_set)
    

# Word2Vec Model 

model = Word2Vec(sent_all, size=300, window =2, negative=5)
word_vectors = model.wv

word_vectors.save("modelswv.kv")
word_vectors = KeyedVectors.load("modelswv.kv")


# SVD Helper Functions 
# Returns dictionary of unique word and index  
def get_word2idx(unique, vocab_size):
    word2ind = {}
    for i in range(vocab_size):
        word2ind[unique[i]] = i
    return word2ind 

# Create co-ocurrence matrix
def co_occurrence(vocab_size, corpus, w):
    word2ind = get_word2idx(unique, vocab_size)
    vocabulary = {}
    row = []
    col = []
    data = [] 
    for sent in sent_all:  # getting individual sentence
        for i in range(len(sent)): # position of word in sentence
            current = sent[i] # current word in sentence, what we want to find co-occurrences 
            #print(current)
            current_ind = word2ind[current]
            #print(current_ind)
            x = vocabulary.setdefault(current_ind, len(vocabulary)) # building row  
            left = max(0,i-w)
            right = min(len(sent), i+w+1)

            for j in range(left, right):
                window_word = sent[j]   
                #print(window_word)
                #print("hi")
                window_ind = word2ind[window_word]
                if current_ind == window_ind:
                    continue
                y = vocabulary.setdefault(window_ind, len(vocabulary)) # building column 
                data.append(1.)
                row.append(x)
                col.append(y)
    # shows (row, col) = non_zero value 
    C = scipy.sparse.coo_matrix((data, (row, col)))
    #print(vocabulary)
    return C

# log p(wi,wj)/p(wi)p(wj) => (#w, #c) dot D/(#w) dot #(c)
# pij = Mij/total sum, pwi = row_sum/total, pcj = col_sum/total 
def PMI(co_matrix): 
    co_matrix = co_matrix.tocsr()
    total   = float(co_matrix.sum())

    row_sum = np.array(co_matrix.sum(axis=1, dtype=np.float64)).flatten()  #w, list of row sums     
    col_sum = np.array(co_matrix.sum(axis=0, dtype=np.float64)).flatten()  #c, list of columns sums 
    #col_sum = np.array(col_nnz.sum().flatten())

    x, y = co_matrix.nonzero()
    P_xy = np.array(co_matrix[x,y], dtype=np.float64).flatten()
    
    numerator = P_xy * total 
    denominator = (row_sum[x] * row_sum[y])
    pmi = np.log(numerator/denominator)
    ppmi = np.maximum(0, pmi)

    final = scipy.sparse.csc_matrix((ppmi, (x,y)), shape=co_matrix.shape, dtype=np.float64)
    
    return final 

def PPMI(co_matrix,k):
    co_matrix = co_matrix.tocsr()
    total   = float(co_matrix.sum())

    row_sum = np.array(co_matrix.sum(axis=1, dtype=np.float64)).flatten()  #w, list of row sums     
    col_sum = np.array(co_matrix.sum(axis=0, dtype=np.float64)).flatten()  #c, list of columns sums 
    #col_sum = np.array(col_nnz.sum().flatten())

    x, y = co_matrix.nonzero()
    P_xy = np.array(co_matrix[x,y], dtype=np.float64).flatten()
    
    numerator = P_xy * total 
    denominator = (row_sum[x] * row_sum[y])
    pmi = np.log(numerator/denominator)
    ppmi = np.maximum(0, pmi - math.log(k))

    final = scipy.sparse.csc_matrix((ppmi, (x,y)), shape=co_matrix.shape, dtype=np.float64)
    
    return final 

# W and C are matrices whose rows are word and context embeddings respectively 
# W = US^1/2, C = VS^1/2
def SVD(M):
    u, s, v = scipy.sparse.linalg.svds(M, k=1000)
    
    # perform dimensionality reduction 
    svd_wv = np.dot(u, np.sqrt(np.diag(s)))
    svd_wv = svd_wv/np.linalg.norm(svd_wv, axis=1).reshape([-1,1])

    return svd_wv
   


def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key 



# SVD Model 
co_occurrence_w2 = co_occurrence(vocab_size, sent_all, 2)
co_occurrence_w5 = co_occurrence(vocab_size, sent_all, 5)
co_occurrence_w10 = co_occurrence(vocab_size, sent_all, 10)


M = PMI(co_occurrence_w2)
#M = PMI(co_occurrence_w5)
#M = PMI(co_occurrence_w10)
#M = PPMI(co_occurrence_w2, 5)
#M = PPMI(co_occurrence_w5,5)
#M = PPMI(co_occurrence_w10, 5)

word2ind = get_word2idx(unique, vocab_size)
svd_final = SVD(M)

# Write to file 

f = open("svd_wv.txt", "w+")
for i in range(len(svd_final)):  
    vec = svd_final[i]
    f.writelines(get_key(i, word2ind))
    vec = [str(item) for item in vec]
    vec = " ".join(vec)
    f.write(" ")
    f.write(vec)
    f.write("\n")

         
    

