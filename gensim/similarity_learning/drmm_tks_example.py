from sl_vocab import WikiQA_DRMM_TKS_Extractor
from drmm_tks import DRMM_TKS

"""Proof of Concept/Example script to demonstrate the trianing of DRMM_TKS model"""

if __name__ == '__main__':

    # Fill in your folder specific details in the variables below
    # TODO make this into argument using arfparse
    file_path = 'data/WikiQACorpus/WikiQA-train.tsv'
    word_embedding_path='evaluation_scripts/glove.6B.50d.txt'


    wikiqa = WikiQA_DRMM_TKS_Extractor(file_path=file_path,
                                       word_embedding_path=word_embedding_path,
                                       embedding_dim=50,
                                       maxlen=40)

    model = DRMM_TKS(embedding=wikiqa.embedding_matrix,
                     embed_dim=wikiqa.embedding_dim,
                     vocab_size=wikiqa.vocab_size + 1)
    model = model.build()
    model.summary()

    optimizer = 'adadelta'
    loss = 'mse'  # we need to define a custom loss TODO
    display_interval = 100

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    generator_function = wikiqa.get_batch_generator(batch_size=32)
    history = model.fit_generator(
        generator_function,
        steps_per_epoch=display_interval,
        epochs=10,
        shuffle=False,
        verbose=1
    )
