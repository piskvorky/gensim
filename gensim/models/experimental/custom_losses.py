try:
    from keras import backend as K
    from keras.layers import Lambda
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

"""Script where all the custom loss functions will be defined"""


def rank_hinge_loss(y_true, y_pred):
    """Loss function for Ranking Similarity Learning tasks
    More details here : https://en.wikipedia.org/wiki/Hinge_loss

    Parameters
    ----------
    y_true : list of list of int
        The true relation between a query and a doc
        It can be either 1 : relevant or 0 : not relevant
    y_pred : list of list of float
        The predicted relation between a query and a doc
    """
    if not KERAS_AVAILABLE:
        raise ImportError("Please install Keras to use this function")
    margin = 0.5
    y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
    loss = K.maximum(0., margin + y_neg - y_pos)
    return K.mean(loss)
