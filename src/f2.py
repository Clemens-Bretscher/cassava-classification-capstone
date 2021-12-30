#<a href="https://www.kaggle.com/ryanzhang/keras-f2-metric">https://www.kaggle.com/ryanzhang/keras-f2-metric</a>

def f2_micro(y_true, y_pred):

    agreement = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_true_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    total_pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = agreement / (total_true_positive + K.epsilon())
    precision = agreement / (total_pred_positive + K.epsilon())
    return (1+2**2)*((precision*recall)/(2**2*precision+recall+K.epsilon()))

if __name__=='__main__':
    f2_micro()