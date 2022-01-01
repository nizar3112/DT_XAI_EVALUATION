# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:22:42 2020

@author: fatih
"""
from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import pandas as pd
from sklearn.datasets import fetch_20newsgroups



##Read Data
#################################
df= pd.read_csv("ohsume_dataset_V2.csv", names=['tags','text'], header=0)

df['class_label'] = df['tags'].factorize()[0]

class_names = list(dict.fromkeys(df.class_label))

class_tags = list(dict.fromkeys(df.tags))

##Spliting into Training and Testing
##################################
training_portion = .9
train_size = int(len(df) * training_portion)

train=df[0: train_size]
test = df[train_size:]

train_docs = train.text.to_list()
train_labels = train.class_label.to_list()

test_docs = test.text.to_list()
test_labels = test.class_label.to_list()


##Encoding Documents and Labels
##################################

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True)
train_vectors = vectorizer.fit_transform(train_docs)
test_vectors = vectorizer.transform(test_docs)

## Building the model
##################################
import tensorflow as tf
from tensorflow.python.keras.preprocessing import text
import tensorflow.compat.v1.keras.backend as K 
#import keras.backend.tensorflow_backend as K
K.set_session


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(50, input_shape = (31727,), activation='relu'))
model.add(tf.keras.layers.Dense(25, activation='relu'))
model.add(tf.keras.layers.Dense(len(class_names), activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

##Applying the model
##################################
model.fit(train_vectors, train_labels, epochs =10, batch_size=128, validation_split=0.1)

##Evaluating the model
##################################
test_labels= test.class_label.to_list()
print('Eval loss/accuracy:{}'.format(model.evaluate(test_vectors, test_labels, batch_size = 128)))



predictions = model.predict(test_vectors[0])


##Appling XAI using LIME
##################################


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, model)

print(c.predict_proba([train_docs[0]]).round(3))

##Add the Explainer
##################################

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


print(len(train_docs))

##Extract features and thier importance scores from a range of documents
##################################

all_data=list()
True_class=list()

for idx in range(len(train_docs)):
	exp = explainer.explain_instance(train_docs[idx], c.predict_proba, num_features=500, labels=[train_labels[idx]])
	print('Document id: %d' % idx)

	print('Predicted class =', class_names[model.predict(train_vectors[idx]).argmax()])

	print('True class: %s' % class_names[train_labels[idx]])


	features_impscores= exp.as_list(label= train_labels[idx])
	
	TC=class_tags[class_names[train_labels[idx]]]
	True_class.append(TC)
	
	features_impscores_list=[list(features_impscores) for features_impscores in features_impscores]
	#print(features_impscores_list)

	data_dict= {}
	for l2 in features_impscores_list:
		data_dict[l2[0]] = l2[1:]
		
	all_data.append(data_dict)
	
##Featurs of all documents and thier related labels
##################################

#print(all_data)
#print(True_class)

##Puting features in dataframes
##################################

dframe=pd.DataFrame([])
for line in all_data:
	data_frame= pd.DataFrame.from_dict(line)
	dframe=dframe.append(data_frame)

dframe.index = range(0,len(dframe))
print(dframe)
##Puting labels also in the dataframe
##################################

#dframe["Class"]= True_class


##Save the dataframe ın a CSV file
##################################
#dframe.to_csv('Output.csv', index = False)



##Building DT 
##################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dframe, True_class, test_size=0.20)

##Convert NULL values into 0`s 
##################################
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


##Check NULL and infintif values 
##################################
np.any(np.isinf(X_train))
np.any(np.isnan(X_train))

##Building and fitting DT model
##################################
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

##Predictions on y_pred
##################################
y_pred = classifier.predict(X_test)

##K-fold cross validation
##################################
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
scores = cross_val_score(classifier, X_train, y_train, cv=kf)

avg_score = np.mean(scores)
print(avg_score)

##Evaluation on DT using Conf. Matrix
##################################
from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("The depth of the tree: "+ str(classifier.get_depth()))

print("No. of leaves: "+ str(classifier.get_n_leaves()))

##Class Based Evaluation on DT 
##################################

tree_structure= classifier.tree_
nodes= tree_structure.__getstate__()['nodes']
no_nodes_samples=tree_structure.n_node_samples

def get_node_depths(tree):
    """
    Get the node depths of the decision tree

    >>> d = DecisionTreeClassifier()
    >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    >>> get_node_depths(d.tree_)
    array([0, 1, 1, 2, 2])
    """
    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths) 
    return np.array(depths)

class_depth= get_node_depths(classifier.tree_)


mynodes=[]

for n in nodes:
	mynodes.append([n[0],n[5]])

nodesDF = pd.DataFrame(mynodes, columns = ['nodes','nos'])
nodesDF['depth']=class_depth.tolist()

leavesDF=nodesDF[nodesDF.nodes==-1]

res=0

for indx,row in leavesDF.iterrows():
	score=row.depth*(row.nos/len(X_train))
	res+=score
	
print("The Average Class Depth: "+ str(res))


##Drawing the tree
##################################

from sklearn import tree
text_representation = tree.export_text(classifier,max_depth= 32)
print(text_representation)

with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)

features_name=list(dframe.columns.values)

import graphviz
# DOT data
dot_data = tree.export_graphviz(classifier, out_file=None, 
                                feature_names=features_name,  
                                class_names=True_class,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph

graph.render("decision_tree_graphivz")
