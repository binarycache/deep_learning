def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()