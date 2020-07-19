from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np

class Metrics:
    """Calculate and return the accuracy and macro-f1 values."""

    def find_TP(y_true, y_pred, label):
        """
        Count the number of true positives (y_true = 1, y_pred = 1).
        Arguments:
           y_true (dataframe): the target containing the true labels 
           y_pred (dataframe): the predicted labels
           label: the true label
        Return:
           (integer) true positive value
        """
        sum_tp = 0
        for i in range(len(y_true)):
            if y_true[i] == label and y_pred[i] == label:
                sum_tp += 1
        return sum_tp

    def find_FN(y_true, y_pred, label):
        """
        Count the number of false negatives (y_true = 1, y_pred = 0).
        Arguments: 
           y_true (dataframe): the target containing the true labels  
           y_pred (dataframe): the predicted labels
	   label: the true label
        Return:
           (integer) false negative value
        """
        sum_fn = 0
        for i in range(len(y_true)):
            if y_true[i] == label and y_pred[i] != label:
                sum_fn += 1
        return sum_fn

    def find_FP(y_true, y_pred, label):
        """
        Count the number of false positives (y_true = 0, y_pred = 1).
        Arguments:
           y_true (dataframe): the target containing the true labels
           y_pred (dataframe): the predicted labels
           label: the true label
        Return:
           (integer) false positive value
        """
        sum_fp = 0
        for i in range(len(y_true)):
            if y_true[i] != label and y_pred[i] == label:
                sum_fp += 1
        return sum_fp

    def find_TN(y_true, y_pred, label):
        """
        Count the number of true negatives (y_true = 0, y_pred = 0).
        Arguments:
           y_true (dataframe): the target containing the true labels
           y_pred (dataframe): the predicted labels
           label: the true label
        Return:
           (integer) true negative value
        """
        sum_tn = 0
        for i in range(len(y_true)):
            if y_true[i] != label and y_pred[i] != label:
                sum_tn += 1
        return sum_tn

    def get_metrics(model, X_train, y_train, X_test, y_test, labels_list):
        """
        Arguments:
           model: trained classifier
           X_train (dataframe): train dataset records
           y_train (series): target in the train dataset
           X_test (dataframe): test dataset records
           y_test (series): target in the test dataset 
           labels_list (list): a list consisting of the classes
        Return:
          Accuracy, Macro-F1, AUC Score
        """
        y_test = y_test.values
        y_pred = model.predict(X_test)

        tn = []
        fp = []
        fn = []
        tp = []
        accuracy = []
        tpr = []
        tnr = []
        fpr = []
        precision  = []
        recall = []
        f1 = []
        f1_macro = []
        for i in range(len(labels_list)):
            tn.append(Metrics.find_TN(y_test, y_pred, i))
            fp.append(Metrics.find_FP(y_test, y_pred, i))
            fn.append(Metrics.find_FN(y_test, y_pred, i))
            tp.append(Metrics.find_TP(y_test, y_pred, i))
            #print(tp, i)
            accuracy.append((tp[i] + tn[i])/(tp[i] + tn[i] + fp[i] + fn[i]))
            tpr.append(tp[i]/(fn[i] + tp[i]))
            tnr.append(tn[i]/(tn[i] + fp[i]))
            fpr.append(fp[i]/(fp[i] + tn[i]))
            try:
                precision.append(tp[i]/(tp[i] + fp[i]))
            except ZeroDivisionError:
                precision.append(0)
            try:
                recall.append(tp[i]/(tp[i] + fn[i]))
            except ZeroDivisionError:
                recall.append(0)
            try:
                f1.append((2*precision[i]*recall[i])/(precision[i] + recall[i]))
            except ZeroDivisionError:
                f1.append(0)
        #calulate macro-f1 - average of the F1 scores of all the classes
        f1_macro = sum(f1)/len(f1)
        
        #auc scores calculation

        #no skill prediction
        ns_probs_0 = [0 for _ in range(len(y_test))]
        ns_probs_1 = [1 for _ in range(len(y_test))]
        try:
            #calculate roc curves
            #ns0_fpr, ns0_tpr, _ = roc_curve(y_test, ns_probs_0)
            #ns1_fpr, ns1_tpr, _ = roc_curve(y_test, ns_probs_1)
            #model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
    
            ns0_auc = roc_auc_score(y_test, ns_probs_0)
            ns1_auc = roc_auc_score(y_test, ns_probs_1)
            model_probs = model.predict_proba(X_test)
            model_probs = model_probs[:, 1]

            model_auc = roc_auc_score(y_test, model_probs)
        except:
            model_auc = -1
        return {"ns0_auc": ns0_auc, "ns1_auc": ns1_auc, "model_auc": model_auc, 'f1_macro': f1_macro, 'accuracy': accuracy[1]} #accuracy values will be the same at all the indices

