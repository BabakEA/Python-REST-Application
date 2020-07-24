# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:27:20 2020

@author: babakea
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random
import pandas.core.algorithms as algos
import scipy.stats.stats as stats
import re
import traceback
import string
import seaborn as sns

from pandas import Series
from pandas import Series, DataFrame
from sklearn.feature_selection import RFE
#from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split , cross_val_score# Python 3
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle










class  Model_Generator:
    def __init__(self,exclud=[]):
        self.Dataset_file='./data/iris.csv'
        self.target='target'
        self.exclud=exclud
        self.Exclud=['Unnamed: 0','Index','index']
        self.Exclud+=[self.target]+self.exclud
        self.dataset_reader()
        self.RF()



    def dataset_reader(self):
        self.df=pd.read_csv(self.Dataset_file)




    def RF(self):
        
        col=[x for x in self.df.columns.tolist() if x not in self.Exclud]
        self.df=shuffle(self.df)
        shuffle(self.df)
        #x=df[df.columns[:-1]]
        x=self.df[col]
        y=self.df[self.target]
        #random.seed(123)
        y=y.astype('int')
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        model = RandomForestClassifier( n_estimators=300, max_depth=50 ,n_jobs = -1,
                                       bootstrap=True,
                                       max_features='auto',
                                       min_samples_leaf= 4,
                                       min_samples_split= 8,
                                      warm_start=True)
    
        model.fit(X_train, y_train) # train with selected Features
        
        y_pred = model.predict(X_test)# test the model performance using the selected features  
        #robY=model.predict_proba(X_test)#predict the probability for x
        testscores = cross_val_score(model, X_train, y_train, cv=5)
        print("Accuracy of test dataset : %0.2f (+/- %0.2f)" % (testscores.mean(), testscores.std() * 2))
        # evaluate accuracy
        print ("ML model accuracy score:", accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        con_mat=confusion_matrix(y_test, y_pred)
        report=classification_report(y_test, y_pred)
        print (report)
        print('Model Performance')
        pickle.dump(model, open('./model/model.pkl','wb'))
        print('The New Model has been Trained')


"""  
if __name__ =="__main__":
    df=dataset_reader(Dataset_file)
    target='target'
    exclud=[]
    RF(df,target,exclud)
    
    """
