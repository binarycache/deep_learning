import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt



def get_num_correct(preds, targets):
    return preds.argmax(dim=1).eq(targets).sum().item()


def get_accuracy(correct_predictions, size:int):
    return correct_predictions/size


# We are not tracking the gradients while doing predictions
@torch.no_grad()
def get_all_predictions(model, loader):
    all_preds = torch.Tensor()

    for batch in loader:
        images, labels = batch

        preds = model(images)

        all_preds = torch.cat(all_preds, preds, dim=1)

    return all_preds

def get_confusion_matrix(targets, predictions, n_classes:int):
    stacked = torch.stack(
                    (targets, predictions.argmax(dim=1)),dim=1
                )
    conf_mat = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    for p in stacked:
        j, k = p.tolist()
        conf_mat[j,k] = conf_mat[j,k] + 1
    return conf_mat

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')