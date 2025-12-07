import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear', save_ckpt=False):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if save_ckpt:
        import os
        os.makedirs('training/reps', exist_ok=True)
        np.save('training/reps/reps.npy', {'reps':test_repr, 'label': test_labels})
        
    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    elif eval_protocol == 'dtw':
        fit_clf = eval_protocols.fit_dtw
    elif eval_protocol == 'tnc':
        fit_clf = eval_protocols.fit_tnc
    elif eval_protocol == 'tst':
        fit_clf = eval_protocols.fit_tst
    elif eval_protocol == 'tstcc':
        fit_clf = eval_protocols.fit_tstcc
    elif eval_protocol == 'tloss':
        fit_clf = eval_protocols.fit_tloss
    elif eval_protocol == 'timesnet':
        fit_clf = eval_protocols.fit_timesnet
    else:
        assert False, f'unknown evaluation protocol: {eval_protocol}'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc }
