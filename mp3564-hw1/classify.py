import sys 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold 
from sklearn import svm 
from sklearn.feature_extraction.text import CountVectorizer 
import sklearn.metrics 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import SGDClassifier 
from sklearn.feature_selection import chi2 
from sklearn.feature_selection import SelectKBest
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC 
from nltk.stem  import WordNetLemmatizer

# I am using 1 late day for this project 

# Computes top 20 features for Ngram model 
def top_20_feat(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    labels = clf.classes_

    top_p = sorted(zip(clf.coef_[0], feature_names))[:n]
    top_c = sorted(zip(clf.coef_[0], feature_names))[-n:]

    for coef, feature in top_p:
        print(labels[0], coef, feature)
    
    for coef, feature in reversed(top_c):
        print(labels[1], coef, feature)

# Computes top 20 features for Custom Feature model 
def top_20_feat2(clf,feature_names, n=20):
    labels = clf.classes_

    top_p = sorted(zip(clf.coef_[0], feature_names))[:n]
    top_c = sorted(zip(clf.coef_[0], feature_names))[-n:]

    for coef, feature in top_p:
        print(labels[0], coef, feature)
    
    for coef, feature in reversed(top_c):
        print(labels[1], coef, feature)

# Unique authors are mapped to integers 
def map_authors(auth):
    x = np.sort(np.unique(auth))
    y= np.searchsorted(x, auth)
    return y

# SVM Classifier for Custom Features 
# The types of features I explored, but did not use are commented out 
def svm_custom_feat(y, sentences, pos, neg, auth):
    
    #lemmatizer = WordNetLemmatizer()

    accuracy_feat = []
    f1_score_feat = []

    kf = KFold(n_splits = 5)
    for train_index, test_index in kf.split(sentences):
        sentences_train, sentences_test = sentences[train_index], sentences[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Reshape input features to be concatenated later 
        author = auth.reshape((auth.shape[0], 1)) 
        pos_emo = pos.reshape((pos.shape[0], 1))
        neg_emo = neg.reshape((neg.shape[0], 1))
        #w_count = words_count.reshape((words_count.shape[0], 1))
        #w_sen = words_sen.reshape((words_sen.shape[0], 1))
        #w_six = words_six.reshape((words_six.shape[0], 1))
        
        # Feature Type 1
        auth_train = map_authors(author[train_index])
        auth_test = map_authors(author[test_index])

        # Feature Type 2 - LIWC Features
        #w_count_train =w_count[train_index]
        #w_sen_train = w_sen[train_index]
        #w_six_train = w_six[train_index]
        pos_train = pos_emo[train_index]
        neg_train = neg_emo[train_index]

        #w_count_test = w_count[test_index]
        #w_sen_test = w_sen[test_index]
        #w_six_test = w_six[test_index]
        pos_test = pos_emo[test_index]
        neg_test = neg_emo[test_index]
        
        # Feature Type 3 - Ngram 
        """
        tfid = TfidfVectorizer(ngram_range=(1,1), stop_words='english') 
        sentence_train_count=tfid.fit_transform(sentences_train) 
        sentence_test_count = tfid.transform(sentences_test)
        sentences_train_count_dense = sentence_train_count.todense()
        sentences_test_count_dense = sentence_test_count.todense()
        """ 

        # Concatenate features 
        text = np.concatenate((pos_train, neg_train, auth_train), axis=1)
        text_test =np.concatenate(( pos_test, neg_test, auth_test), axis=1)
        
        # Create SVM Classifier 
        clf_svm = LinearSVC(penalty='l2', loss = 'squared_hinge', tol=1e-3, C=0.01, max_iter=1000000)

        # Training 
        clf_svm.fit(text, y_train)

        # Prediction  
        y_predict = clf_svm.predict(text_test)
        
        #print(sklearn.metrics.classification_report(y_test, y_predict))
        accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
        f1_score = sklearn.metrics.f1_score(y_test, y_predict, average = 'macro')

        accuracy_feat.append(accuracy)
        f1_score_feat.append(f1_score)

        """
        GridSearchCV 

        grid = GridSearchCV(estimator=clf_svm, param_grid={'loss': ('hinge', 'squared_hinge'), 
         'C': (0.01,0.1,1.0,10.0)
        })

        grid.fit(sentences_train_count, y_train)
        print("grid search cv")
        print(grid.best_params_)
        """
    print("Top 20 Feature Names")
    print("Author")
    print(top_20_feat2(clf_svm, author, n=20))
    print("Positive Emotion Words")
    print(top_20_feat2(clf_svm, pos_emo, n=20))
    print("Negative Emotion Words")
    print(top_20_feat2(clf_svm, neg_emo, n=20))
    avg_accuracy_feat = np.mean(accuracy_feat)
    avg_f1_score_feat = np.mean(f1_score_feat)

    return avg_accuracy_feat, avg_f1_score_feat

# Naive Bayes Classifier for Custom Features
def nb_custom_feat(y, sentences, pos_emo, neg_emo, author):
    
    accuracy_feat = []
    f1_score_feat = []
    kf = KFold(n_splits = 5)

    for train_index, test_index in kf.split(sentences):
        sentences_train, sentences_test = sentences[train_index], sentences[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Feature Type 1
        auth_train = author[train_index]
        auth_test = author[test_index]

        # Feature Type 2  -  LIWC Features
        pos_train = pos_emo[train_index]
        neg_train = neg_emo[train_index]

        pos_test = pos_emo[test_index]
        neg_test = neg_emo[test_index]

        pos_train_reshape = pos_train.reshape((pos_train.shape[0],1))
        neg_train_reshape = neg_train.reshape((neg_train.shape[0], 1))
        auth_train_reshape =  auth_train.reshape((auth_train.shape[0], 1))

        pos_test_reshape = pos_test.reshape((pos_test.shape[0],1))
        neg_test_reshape = neg_test.reshape((neg_test.shape[0], 1))
        auth_test_reshape =  auth_test.reshape((auth_test.shape[0], 1))

        # Feature Type 3 - Ngram
        tfid = TfidfVectorizer(ngram_range=(1,3)) 
        sentences_train_count=tfid.fit_transform(sentences_train)
        sentences_test_count = tfid.transform(sentences_test)
        sentences_train_count_dense = sentences_train_count.todense()
        sentences_test_count_dense = sentences_test_count.todense()

        # Concatenate features to output of tfid 
        text = np.concatenate((sentences_train_count_dense, pos_train_reshape), axis=1)
        text_test=np.concatenate((sentences_test_count_dense, pos_test_reshape), axis=1)
        
        # Create Naive Bayes Classifier 
        clf_nb = MultinomialNB()

        # Training 
        clf_nb.fit(text, y_train)

        # Prediction  
        y_predict = clf_nb.predict(text_test)

        print(sklearn.metrics.classification_report(y_test, y_predict))
        accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
        f1_score = sklearn.metrics.f1_score(y_test, y_predict, average = 'macro')

        accuracy_feat.append(accuracy)
        f1_score_feat.append(f1_score)
    
    avg_accuracy_feat = np.mean(accuracy_feat)
    avg_f1_score_feat = np.mean(f1_score_feat)

    return avg_accuracy_feat, avg_f1_score_feat

# Naive Bayes Classifier for N-gram Model 
def nb_ngram(y, sentences):

    # Store accuracies and f1 scores in two lists for each fold 
    accuracy_ngram = []
    f1_score_ngram =  []

    # KFold Cross Validation  
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(sentences):
        sentences_train, sentences_test = sentences[train_index], sentences[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Vectorization 
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
        sentences_train_count = vectorizer.fit_transform(sentences_train)
        sentences_test_count  = vectorizer.transform(sentences_test)

        # Create Naive Bayes Classifier 
        clf_nb= MultinomialNB(alpha=1.0, fit_prior=True)        
         
        # Training
        clf_nb.fit(sentences_train_count, y_train)

        # Prediction 
        y_predict = clf_nb.predict(sentences_test_count)
        #print(sklearn.metrics.classification_report(y_test, y_predict))
        
        # Compute Accuracy and F1 Score 
        accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
        f1_score = sklearn.metrics.f1_score(y_test, y_predict, average = 'macro')
        
        accuracy_ngram.append(accuracy)
        f1_score_ngram.append(f1_score)

       
        """
        I have commented out the GridSearchCV portion for efficiency purposes

        grid = GridSearchCV(estimator=clf_nb, param_grid={'alpha': (0,0.1,1.0,10.0 ,100.0), 'fit_prior': (True, False)})
        grid.fit(sentences_train_count, y_train)
        print("grid search cv")
        print(grid.best_params_)

        """
    print("Top 20 Features")
    print(top_20_feat(vectorizer, clf_nb, n=20))
    avg_accuracy_ngram = np.mean(accuracy_ngram)
    avg_f1_score_ngram = np.mean(f1_score_ngram)
    return avg_accuracy_ngram, avg_f1_score_ngram

if __name__ == "__main__":
    input_file =  sys.argv[1]
    topic = sys.argv[2]

    data = pd.read_csv("stance-data.csv") 
    d = data.as_matrix()

    if topic == "abortion":
        y =  d[0:1159, 3]
        sentences = d[0:1159, 0]
        pos_emo = d[0:1159, 9]
        neg_emo = d[0:1159, 10]
        word_count = d[0:1159, 5]
        words_per_sen = d[0:1159, 7]
        words_over_six = d[0:1159, 8]
        author = d[0:1159, 2]

        (avg_accuracy_ngram, avg_f1_score_ngram) = nb_ngram(y, sentences)
        print('\nAverage accuracy of Ngram model is %f' % (avg_accuracy_ngram *100))
        print('\nAverage f1 score of Ngram model is %f' %(avg_f1_score_ngram *100))

        """
        This commented section demonstrates that I compared the accuracy and f1 score of both models 
        I have commented it out for efficiency purposes, since I know which model is the best

        (avg_accuracy_feat, avg_f1_score_feat) = svm_custom_feat(y, sentences, pos_emo, neg_emo,author)
        if avg_accuracy_ngram > avg_accuracy_feat:
            print('\nAverage accuracy ngram is %f' % (avg_accuracy_ngram *100))
            print('\nAverage f1 score ngram is %f' %(avg_f1_score_ngram *100))
            print("Top 20 Features:")
        else:
            print('\nAverage accuracy is %f' % (avg_accuracy_feat *100))
            print('\nAverage f1 score is %f'%(avg_f1_score_feat *100))
        """ 

    else:
        y = d[1159:2164, 3]
        sentences  = d[1159:2164, 0]
        pos_emo = d[1159:2164, 9]
        neg_emo = d[1159:2164, 10]
        word_count = d[1159:2164, 5]
        words_per_sen = d[1159:2164, 7]
        words_over_six = d[1159:2164, 8]
        author = d[1159:2164, 2]

        (avg_accuracy_feat, avg_f1_score_feat) = svm_custom_feat(y, sentences, pos_emo, neg_emo, author)
        print('\nAverage accuracy of Custom Feature Model is  %f' % (avg_accuracy_feat *100))
        print('\nAverage f1 score of Custom Feature Model is %f'%(avg_f1_score_feat *100))

        """
        This commented section demonstrates that I compared the accuracy and f1 score of both models 
        I have commented it out for efficiency purposes, since I know which model is the best

        (avg_accuracy_feat, avg_f1_score_feat) = svm_custom_feat(y, sentences, pos_emo, neg_emo,author)
        if avg_accuracy_ngram > avg_accuracy_feat:
            print('\nAverage accuracy ngram is %f' % (avg_accuracy_ngram *100))
            print('\nAverage f1 score ngram is %f' %(avg_f1_score_ngram *100))
            print("Top 20 Features:")
        else:
            print('\nAverage accuracy is %f' % (avg_accuracy_feat *100))
            print('\nAverage f1 score is %f'%(avg_f1_score_feat *100))
        """ 
