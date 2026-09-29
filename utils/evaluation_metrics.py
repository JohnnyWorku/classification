from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


def accuracy_metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred) 
    
    return accuracy


def precision_metric(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    
    return precision


def recall_metric(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    return recall


# Most of the time for evaluation of specific class like "EC:1"
def f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, pos_label="EC:1")
    
    return f1


def macro_f1_metric(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return macro_f1


def mattews_corrcoef_metric(y_true, y_pred):
    matthews_metric = matthews_corrcoef(y_true, y_pred)
    
    return matthews_metric

    