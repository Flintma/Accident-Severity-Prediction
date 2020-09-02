#Download intial packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import urllib2

#download dataset

response=urllib2.urlopen (https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv)


df_collisions = pd.read_csv(response)
df_collisions.head()

#create subset with just the desired features
df_collide = df_collisions[['SEVERITYCODE','LIGHTCOND','ROADCOND','WEATHER']]

df_collide.head()

df_collide['WEATHER'].value_counts()


#Make a dictionary that replaces the values of the features with numerals
cleanup_nums = {"LIGHTCOND": {"Daylight": 0, "Dark - Street Lights On": 3, "Unknown": 8, "Dusk": 1, "Dawn": 2, "Dark - No Street Lights": 4,
                                 "Dark - Street Lights Off": 5, "Other": 7, "Dark - Unknown Lighting": 6},
                "ROADCOND": {"Dry": 0, "Wet": 1, "Unknown": 8, "Ice": 2, "Snow/Slush": 3, "Other": 7, "Standing Water": 6, "Sand/Mud/Dirt": 5,
                            "Oil": 4},
               "WEATHER": {"Clear": 0, "Raining": 3, "Overcast": 2, "Unknown": 10, "Snowing": 4, "Other": 9, "Fog/Smog/Smoke": 6, 
                          "Sleet/Hail/Freezing Rain": 5, "Blowing Sand/Dirt": 7, "Severe Crosswind": 8, "Partly Cloudy": 1}}

df_collide.replace(cleanup_nums, inplace=True)
df_collide

df_collide.head()

#Eliminate rows with missing values
df_collide=df_collide.dropna()
df_collide.shape

df_collide.dtypes

#Convert floats to integers
df_collide = df_collide.astype(int)
df_collide.dtypes

df_collide.head()

df_collide['SEVERITYCODE'].value_counts()

#Separate the dataframe into two different ones by 'SEVERITYCODE'
df_majority = df_collide[df_collide['SEVERITYCODE']==1]
df_minority = df_collide[df_collide['SEVERITYCODE']==2]

df_majority.shape

df_minority.shape

from sklearn.utils import resample

#Downsample the majority and then combine with minority dataframe to balance
df_majority_downsampled = resample(df_majority, 
                                 replace=False,   
                                 n_samples=57052,     
                                 random_state=123)

df_collide_bal = pd.concat([df_majority_downsampled, df_minority])
df_collide_bal['SEVERITYCODE'].value_counts()
#Define X and Y
X = np.asarray(df_collide_bal[['LIGHTCOND','ROADCOND','WEATHER']])
Y = np.asarray(df_collide_bal['SEVERITYCODE'])

#Normalize the dataset
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
#Split the data into a testing set and a training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LogR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LogR

yhat = LogR.predict(X_test)
yhat
yhat_prob = LogR.predict_proba(X_test)
yhat_prob

print("CollisionLR Accuracy: ", metrics.accuracy_score(y_test, yhat))

from sklearn.tree import DecisionTreeClassifier
Collision_Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 6)
Collision_Tree
Collision_Tree.fit(X_train,y_train)
Treeyhat=Collision_Tree.predict(X_test)
print (Treeyhat [0:5])
print (y_test [0:5])
from sklearn import metrics
print("CollisionTrees's Accuracy: ", metrics.accuracy_score(y_test, Treeyhat))

from sklearn.neighbors import KNeighborsClassifier
k=20

colneigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
colneigh

KNNyhat = colneigh.predict(X_test)
KNNyhat[0:5]
print("CollisionKNN Accuracy: ", metrics.accuracy_score(y_test, KNNyhat))

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
LRJ = jaccard_similarity_score(y_test, yhat)
LRF1 = f1_score(y_test, yhat, average='weighted')
LRLL = log_loss(y_test, yhat_prob)

print("LOG LogLoss: : %.4f" % LRLL)
print("LOG F1-score: %.4f" % LRF1)
print("LOG Jaccard score: %.4f" % LRJ)
CTJ = jaccard_similarity_score(y_test, Treeyhat)
CTF1 = f1_score(y_test, Treeyhat, average='weighted')

print("Decision Tree F1-score: %.4f" % CTF1 )
print("Decision Tree Jaccard Score: %.4f" % CTJ)
KNNJ = jaccard_similarity_score(y_test, KNNyhat)
KNNF1 = f1_score(y_test, KNNyhat, average='weighted')

print("KNN F1-score: %.4f" % KNNF1 )
print("KNN Jaccard Score: %.4f" % KNNJ)
