1)	How to train and test classifier

In both types of models I use kfold cross validation where n_splits = 5. The first step in training my classifier is feature extraction. In my n-gram model, for example, I used the bag of words representation CountVectorizer(), and used a combination of unigrams, bigrams, and trigram. In the training process, the feature vector transfers the text into a feature of vector, and pairs of features vectors and labels are fed into the ML algorithm. The ML algorithm for abortion and gay rights, respectively, are Naive Bayes and SVM. The trained model can now make predictions on the test set. Using the array of true class labels, I can then evaluate the accuracy of my model’s classifier. 


2)	Special Features and Limitations:

In my Ngram model I used MultinomialNB() as this in comparison to LinearSVC() produced optimal results. In my custom feature model I used the LinearSVC() classifier. I used GridSearchCV() for hyper parametrization for both types of classifiers. I kept the default parameter values for MultinomialNB(). For LinearSVC() I changed the loss function to hinge instead of squared_hinge and changed C to 0.01 instead of 1.0. There may be significant limitations in my custom feature model, used for Gay Rights, as the same accuracy is outputted regardless of different feature combinations. 

There are significant disadvantages and limitations for both SVM and Naive Bayes classifiers. Disadvantages of the Naïve Bayes classifier is its assumption that features are independent of each other, when in reality, features can be related. Also, the frequency based estimate can be zero when there are minimal occurrences of a class label. This can happen when vectors are not fully representative of a population. Disadvantages of the SVM Classifier are that several key parameters need to be set to generate high accuracy. My GridSearchCV() was not exhaustive, and I am certain there are additional parameters and parameter values I could have tested. 

Additional Notes:
The features I explored but did not use in my custom feature model using linear SVC are commented out but clearly indicated. The written portion of this assignment demonstrates the testing of different combinations of features. 

References I used:

https://scikit-learn.org/0.18/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
https://scikit-learn.org/stable/modules/cross_validation.html#k-fold
https://scikit-learn.org/stable/modules/svm.html#svm-classification
https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search
https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
https://scikit-learn.org/stable/modules/naive_bayes.html

https://www.researchgate.net/post/What_are_the_disadvantages_of_Naive_Bayes




