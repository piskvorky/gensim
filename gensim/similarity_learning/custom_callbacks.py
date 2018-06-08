from keras.callbacks import Callback

class SLCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        data = self.test_data
        print("query", type(data[0][0]["query"]))
        print("doc ", data[0][0]["doc"])
        print(self.model.predict({'query': data[0][0]["query"], 'doc': data[0][0]["doc"]}))