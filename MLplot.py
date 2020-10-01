import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools


def plot_roc(y_score, y_test, labels=None):
    if labels is None: labels = list(set(y_test))
    # encode labels
    y_test_enc = np.zeros((len(y_test), len(labels)))
    for i, lab in enumerate(labels):
        indices = np.where(np.array(y_test) == lab)[0]
        y_test_enc[indices, i] = 1
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i, lab in enumerate(labels):
        fpr[lab], tpr[lab], _ = roc_curve(y_test_enc[:, i], y_score[:, i])
        roc_auc[lab] = auc(fpr[lab], tpr[lab])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(np.array(y_test_enc).ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    # for i, lab in enumerate(['micro']):
    for i, lab in enumerate(labels[::-1]):
    # for i, lab in enumerate(labels):
        plt.plot(fpr[lab], tpr[lab],# color='darkorange',
                 lw=2, label=lab + ', AUC = %0.2f' % roc_auc[lab],
                )
        if len(labels)==2: break
    plt.grid(color='lightgray')
    ax = plt.gca()
    for spine in ax.spines.values(): spine.set_edgecolor('lightgray')
    plt.plot([0, 1], [0, 1], color='lightgray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.gca().set_aspect('equal')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    marg = 1e-2; plt.ylim(-marg, 1+marg), plt.xlim(-marg, 1+marg)
    plt.gcf().patch.set_facecolor('white')


def plot_cm(y_pred, y_test, labels=None):
    if labels is None: labels = list(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.imshow(cm, cmap='Blues')
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.gcf().patch.set_facecolor('white')
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    accuracy = np.trace(cm) / float(np.sum(cm))
    plt.title('Confusion Matrix, Acc.={:0.0%}'.format(accuracy))



def log(x, margin=1e-6):
    x = np.array(x)
    return np.log(x - x.min() + margin)
    
    
def plot_clusters(df, y, feature1, feature2, *, labels=None, log_conversion=False):
    if labels is None: labels = sorted(list(set(y)))
    for lab in labels:
        scatter_x = df.iloc[np.where(np.array(y) == lab)][feature1].astype('float')
        scatter_y = df.iloc[np.where(np.array(y) == lab)][feature2].astype('float')
        if log_conversion:
            scatter_x = log(scatter_x)
            scatter_y = log(scatter_y)
        plt.scatter(scatter_x, scatter_y,
                    label=lab, alpha=.5,
                   )
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(labels, bbox_to_anchor=(1,1))
    plt.gcf().patch.set_facecolor('white')