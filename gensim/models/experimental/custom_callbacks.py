import logging
try:
    from keras.callbacks import Callback
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE =False

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ValidationCallback(Callback):

    def __init__(self, test_data):
        if not KERAS_AVAILABLE:
            raise ImportError("Please install Keras to use this class")

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

        print("MAP: ", mapk(Y_true, Y_pred))
        for k in [1, 3, 5, 10, 20]:
            print("nDCG@", str(k), ": ", mean_ndcg(Y_true, Y_pred, k=k))
