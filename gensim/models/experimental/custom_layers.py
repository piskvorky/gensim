try:
    from keras.engine.topology import Layer
    import keras.backend as K
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

"""All custiom keras layers will be housed here"""


class TopKLayer(Layer):
    """Layer to get top k values from the interaction matrix in drmm_tks model"""

    def __init__(self, output_dim, topk, **kwargs):
        """
        Parameters:
        ----------
        output_dim : The dimension of the tensor after going through this layer

        topk : int
            The k topmost values to be returned"""
        self.output_dim = output_dim
        self.topk = topk
        super(TopKLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TopKLayer, self).build(input_shape)

    def call(self, x):
        return K.tf.nn.top_k(x, k=self.topk, sorted=True)[0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0], self.output_dim[1])

    def get_config(self):
        config = {
            'topk': self.topk,
            'output_dim': self.output_dim
        }
        base_config = super(TopKLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
