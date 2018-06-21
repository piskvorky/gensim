from keras import backend as K
from keras.layers import Lambda


def rank_hinge_loss(y_true, y_pred):
    margin = 0.5
    y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
    loss = K.maximum(0., margin + y_neg - y_pos)
    return K.mean(loss)
