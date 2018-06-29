import logging
try:
    from keras.callbacks import Callback
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ValidationCallback(Callback):
    """Callback for providing validation metrics on the model trained so far"""
    def __init__(self, test_data):
        """
        Parameters
        ----------
        test_data : dict
            A dictionary which holds the validation data
            It consists of the following keys:
                "X1" : numpy array
                    The queries as a numpy array of shape (n_samples, text_maxlen)
                "X2" : numpy array
                    The candidate docs as a numpy array of shape (n_samples, text_maxlen)
                "y" : list of int
                      It is the labels for each of the query-doc pairs as a 1 or 0 with shape (n_samples,)
                      where 1: doc is relevant to query
                            0: doc is not relevant to query
                "doc_lengths" : list of int
                                It contains the length of each document group. I.e., the number of queries
                                which represent one topic. It is needed for calculating the metrics.
        """

        if not KERAS_AVAILABLE:
            raise ImportError("Please install Keras to use this class")

        # Check if all test_data is a dicitonary with all the right keys
        try:
            # If an empty dict is passed
            if len(test_data.keys()) == 0:
                raise ValueError(
                      "test_data dictionary is empty. It doesn't have the keys: 'X1', 'X2', 'y', 'doc_lengths'"
                    )
            for key in test_data.keys():
                if key not in ['X1', 'X2', 'y', 'doc_lengths']:
                    raise ValueError("test_data dictionary doesn't have the  keys: 'X1', 'X2', 'y', 'doc_lengths'")
        except AttributeError:
            raise ValueError("test_data must be a dictionary with the keys: 'X1', 'X2', 'y', 'doc_lengths'")
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        # Import has to be here to prevent cyclic import
        from evaluation_metrics import mapk, mean_ndcg
        X1 = self.test_data["X1"]
        X2 = self.test_data["X2"]
        y = self.test_data["y"]
        doc_lengths = self.test_data["doc_lengths"]

        predictions = self.model.predict(x={"query": X1, "doc": X2})

        Y_pred = []
        Y_true = []
        offset = 0

        for doc_size in doc_lengths:
            Y_pred.append(predictions[offset: offset + doc_size])
            Y_true.append(y[offset: offset + doc_size])
            offset += doc_size

        logger.info("MAP: %.2f", mapk(Y_true, Y_pred))
        for k in [1, 3, 5, 10, 20]:
            logger.info("nDCG@%d : %.2f", k, mean_ndcg(Y_true, Y_pred, k=k))
