import math

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support


def evaluate_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mse = mean_squared_error(y_true, y_pred, squared=True)
    mae = mean_absolute_error(y_true, y_pred)
    return {'rmse': rmse, 'mse': mse, 'mae': mae}

def evaluate_classification(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred, multi_class='ovo')    
    y_pred = y_pred.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {
        'accuracy': acc,
        'precision': precision, 'recall': recall, 'f1': f_score,
        'auc': auc
    }


def evaluate_anomaly_detection(x_test, x_recon, y_true):
    def _flatten_anomaly_scores(values, stride, flatten=False):
        flat_seq = []
        if flatten:
            for i, x in enumerate(values):
                if i == len(values) - 1:
                    flat_seq = flat_seq + list(np.ravel(x).astype(float))
                else:
                    flat_seq = flat_seq + list(np.ravel(x[:stride]).astype(float))
        else:
            flat_seq = list(np.ravel(values).astype(float))
    
        return flat_seq
        
    anomaly_scores = np.mean(np.mean(np.square(x_test - x_recon), axis=-1), axis = 0) if len(x_recon.shape) > 3 else np.mean(np.square(x_test - x_recon), axis=-1)
    flat_scores = _flatten_anomaly_scores(anomaly_scores, stride=1, flatten=len(anomaly_scores.shape) == 2)
    auc = roc_auc_score(y_true, flat_scores)
    
    return { 'auc': auc }