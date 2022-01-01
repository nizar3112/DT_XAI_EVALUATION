# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:12:28 2020

@author: Nizar Ahmed
"""

import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.preprocessing import text
import keras.backend.tensorflow_backend as K
K.set_session
import shap


# Read, shuffle, and preview the data
data = pd.read_csv('ohsume_dataset_V2.csv', names=['tags','text'], header=0)

data['class_label'] = data['tags'].factorize()[0]

class_names = list(dict.fromkeys(data.class_label))

class_tags = list(dict.fromkeys(data.tags))

# Identify and split the labels
tags_split = [tags.split(',') for tags in data['tags'].values]
tag_encoder = MultiLabelBinarizer()
tags_encoded = tag_encoder.fit_transform(tags_split)
num_tags = len(tags_encoded[0])
train_size = int(len(data) * .9)

y_train = tags_encoded[: train_size]
y_test = tags_encoded[train_size:]

y_train_cat=tag_encoder.inverse_transform(y_train)
y_test_cat=tag_encoder.inverse_transform(y_test)

# Identify, processing and splitting the text
class TextPreprocessor(object):
    def __init__(self, vocab_size):
        self._vocab_size = vocab_size
        self._tokenizer = None
    def create_tokenizer(self, text_list):
        tokenizer = text.Tokenizer(num_words = self._vocab_size)
        tokenizer.fit_on_texts(text_list)
        self._tokenizer = tokenizer
    def transform_text(self, text_list):
        text_matrix = self._tokenizer.texts_to_matrix(text_list)
        return text_matrix
     

from nltk.corpus import stopwords
stop = stopwords.words('english')

VOCAB_SIZE = 40611
data1 = data['text'].str.replace('\n', '')
data1 = (data1.str.lower())
data1= (data1.str.replace(r'[^\w\s]+', ''))
data1 = data1.apply(lambda x: [item for item in x.split() if item not in stop])

train_post = data1.values[: train_size]
test_post = data1.values[train_size: ]

processor = TextPreprocessor(VOCAB_SIZE)
processor.create_tokenizer(train_post)
X_train = processor.transform_text(train_post)
X_test = processor.transform_text(test_post)


# Building the model

def create_model(vocab_size, num_tags):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(50, input_shape = (VOCAB_SIZE,), activation='relu'))
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(num_tags, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

model = create_model(VOCAB_SIZE, num_tags)

# Applying the model
model.fit(X_train, y_train, epochs = 10, batch_size=128, validation_split=0.1)

# Evaluating the model
print('Eval loss/accuracy:{}'.format(model.evaluate(X_test, y_test, batch_size = 128)))

# Explaining the behaviour of the model using SHAP

# we use the first 100 training examples as our background dataset to integrate over
explainer = shap.DeepExplainer(model, X_train[:100])

# explain the first 10 predictions
# explaining each prediction requires 2 * background dataset size runs


dataframe=list()
count=0
for i in X_train:
	
	train_class=y_train_cat[count][0]
	train_clas_number=class_tags.index(train_class)
	
	shap_values = explainer.shap_values(i.reshape(1,40611))
	
	predicted_rec=shap_values[train_clas_number]
	l=predicted_rec.tolist()
	l=l[0]
	l.append(train_class)
	dataframe.append(l)
	count+=1
	if count>1000:
		break

words = processor._tokenizer.word_index
word_lookup = list()
for i in words.keys():
  word_lookup.append(i)
word_lookup = [''] + word_lookup
word_lookup.append("class_name")

#df = DataFrame (dataframe,columns=['Column_Name'])
df = pd.DataFrame (dataframe,columns = word_lookup)



##Building DT 
##################################
DTtext= df.iloc[: , :-1]

##Selecting the top N features
##################################
topn=20
sort_decr2_topn = lambda row, nlargest=topn: sorted(pd.Series(zip(DTtext.columns, row)), key=lambda cv: -cv[1]) [:nlargest]

tmp = DTtext.apply(sort_decr2_topn, axis=1)
np.array(tmp)

all_data=list()

for l2 in tmp:
	data_dict= {}
	for l3 in l2:
		data_dict[l3[0]] = l3[1:]
	all_data.append(data_dict)

newframe=pd.DataFrame([])
for line in all_data:
	data_frame= pd.DataFrame.from_dict(line)
	newframe= newframe.append(data_frame)
newframe.index = range(0,len(newframe))
newframe = newframe.fillna(0)

##Building and fitting DT model
##################################
from sklearn.model_selection import train_test_split
X_DTtrain, X_DTtest, y_DTtrain, y_DTtest = train_test_split(newframe, df["class_name"], test_size=0.20)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_DTtrain, y_DTtrain)

##Predictions on y_pred
##################################
y_pred = classifier.predict(X_DTtest)

##K-fold cross validation
##################################
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
scores = cross_val_score(classifier, X_DTtrain, y_DTtrain, cv=kf)

avg_score = np.mean(scores)
print(avg_score)

##Evaluation on DT using Conf. Matrix
##################################
from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_DTtest, y_pred))

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
	score=row.depth*(row.nos/len(X_DTtrain))
	res+=score
	
print("The Average Class Depth: "+ str(res))


##Drawing the tree
##################################

from sklearn import tree
text_representation = tree.export_text(classifier,max_depth= 36)
print(text_representation)

with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)

features_name=list(newframe.columns.values)

import graphviz
# DOT data
dot_data = tree.export_graphviz(classifier, out_file=None, 
                                feature_names=features_name,  
                                class_names=df["class_name"],
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph

graph.render("decision_tree_graphivz")

