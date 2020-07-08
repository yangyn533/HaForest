# -*- coding:utf-8 -*-
import csv 
import numpy as np 
import pandas as pd
import linecache
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score, accuracy_score,auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from xgboost import XGBClassifier
from sklearn.decomposition import NMF, LatentDirichletAllocation,FastICA
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
#from GCForest import *
from gcforest.gcforest import GCForest
#from gcforest.gc import gcForest
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import t_sne, MDS, Isomap
from keras.utils import to_categorical
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.externals import joblib
from sklearn.linear_model import Lasso, RandomizedLasso,LogisticRegression
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from numpy import interp
import lightgbm as lgb
from gcforest.utils.config_utils import load_json
from sklearn.decomposition import NMF
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.decomposition import KernelPCA,PCA
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE

np.random.seed(4)

def run_multilda(parser):
    window_size = parser.window
    data = parser.data
    label = parser.label
    TrainData = np.load(data)
    TrainDatalist = TrainData.tolist()
    y = np.load(label)
    lda = LinearDiscriminantAnalysis(n_components= 2)
    SS = []
    for each_item in window_size:
        n = each_item
        i = 0
        while i < (TrainData.shape[1]-int(each_item)+1):
            if i == 0:
                Seq = []
                for j in range(len(TrainDatalist)):
                    line = TrainDatalist[j]
                    line = line[i:i+n]
                    Seq.append(line)
                X = np.array(Seq)
                trans = lda.fit_transform(X,y)
            else:
                Seque = []
                for j in range(len(TrainDatalist)):
                    line = TrainDatalist[j]
                    line = line[i:i+n]
                    Seque.append(line)
                X = np.array(Seque)
                trans1 = lda.fit_transform(X,y) 
                trans2 = np.hstack((trans,trans1))
                trans = trans2
            i = i + 1
        SS.append(trans)
    Concenate = np.hstack((SS[0],SS[1],SS[2]))
    return Concenate
 
def get_toy_config():

    config = {}
    ca_config = {}
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500,  "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 3, "type": "XGBClassifier", "n_estimators":500, "eval_metric": "auc", "objective": "binary:logistic", "nthread": -1})
    config["cascade"] = ca_config
    return config

def tsne(data_set,data_y):
    X_tsne = TSNE(n_components=2).fit_transform(data_set)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_y,marker='o',color='r')
    plt.show()

def pca(data_set,data_y):
    X_tsne = PCA(n_components=2).fit_transform(data_set)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_y,marker='o')
    plt.show()
    
def load_data(parser):
    data = parser.data
    label = parser.label    
    Train = np.load(data)
    TrainLabel = np.load(label)
    return TrainLabel

def plot(y_true,y_scores,auc_value):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RF ROC Curve of HIS and HS')
    plt.legend(loc="lower right")
    plt.show()    

def parse_arguments(parser):
    parser.add_argument('--window', type=list,  default=[100,200,300])
    parser.add_argument('--data', type=str, default='../HIHS.npy',help='the data for training model')
    parser.add_argument('--label', type=str, default='../Label.npy',help='the label for training model')
    args = parser.parse_args()
    return args    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    Train = run_multilda(args)  
    TrainLabel = load_data(args)
    AUC =[]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)    

    correct = 0
    AUC1 = 0
    for i in range(1,101):
        X_train, X_test, Y_train, Y_test = train_test_split(Train, TrainLabel, test_size=0.1)
        
        config = get_toy_config()
        gc = GCForest(config)     
        gc.fit_transform(X_train, Y_train, X_test=X_test, y_test=Y_test) 
 
        y_true = Y_test
        y_pred = gc.predict(X_test)
        y_pred_pro = gc.predict_proba(X_test)
        y_scores = y_pred_pro[:,1] 
        auc_value = roc_auc_score(y_true, y_scores)
        acc = accuracy_score(y_true, y_pred)
           
        AUC.append(auc_value)    
        
        
        fpr,tpr,threshold = roc_curve(y_true, y_scores) 
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        plt.plot(fpr, tpr, lw=1, alpha=0.3)
        
        i = i + 1
    ##########################################################################################
    
    plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ###################################
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
    
    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    ###################################
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic of 10 Fold Cross Validation')
    plt.title('Receiver Operating Characteristic of 100 Runs')
    plt.legend(loc="lower right",fontsize='x-large')
    #plt.legend(loc='lower right', fontsize=6.5)
    ##################################
    print("acid AUC: %.4f " % np.mean(AUC))
    #plt.show()  

    