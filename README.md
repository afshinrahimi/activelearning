Created on Jul 4, 2014
based on http://scikit-learn.org/stable/auto_examples/document_classification_20newsgroups.html

This program implements active learning for classification tasks with scikit-learn's LinearSVC classifier.
Instead of using a Stochastic Gradient Descent training we used the batch mode because 
the data is not that big and accuracy is more important for us than efficiency.

The algorithm trains the model based on train dataset and evaluates using the test dataset.
After each evaluation algorithm selects 2*NUM_QUESTIONS samples from unlabeled dataset in order
to be labeled by a user/expert. The labeled sample is moved to the corresponding directory in
train dataset and the model will start again training with the new improved training set.

The selection of unlabeled samples is based on decision_function of SVM which is
the distance of the samples X to the separating hyperplane. This distance is between
[-1, 1] but because we need confidence levels we use absolute values. In case the classes
are more than two the decision function will return a confidence level for each class for each sample
so in case we have more than 2 classes we average over the absolute values of confidence over all classes.

We use top NUM_QUESTIONS samples with highest average absolute confidence and also top NUM_QUESTIONS
samples with lowest average absolute confidence for expert labeling. This procedure can be easily changed
by modifying the code in benchmark function.

This program requires a directory structure similar to what is shown below:
    mainDirectory
       train
           pos
               1.txt
               2.txt
           neg
               3.txt
               4.txt
       test
           pos
               5.txt
               6.txt
           neg
               7.txt
               8.txt
       unlabeled
           unlabeled
               9.txt
               10.txt
               11.txt
The filenames in unlabeled should not be a duplicate of filenames in train directory because every time we label a file
we will move that file into the corresponding class directory in train directory.

The pos and neg categories are arbitrary and both the number of the classes and their name can be different with what is shown here.
The classifier can also be changed to any other classifier in scikit-learn.


@author: afshin rahimi
