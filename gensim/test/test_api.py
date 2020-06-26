import logging
import unittest
import os
import gensim.downloader as api
import shutil
import numpy as np


@unittest.skipIf(
    os.environ.get("SKIP_NETWORK_TESTS", False) == "1",
    "Skip network-related tests (probably SSL problems on this CI/OS)"
)
class TestApi(unittest.TestCase):
    def test_base_dir_creation(self):
        if os.path.isdir(api.BASE_DIR):
            shutil.rmtree(api.BASE_DIR)
        api._create_base_dir()
        self.assertTrue(os.path.isdir(api.BASE_DIR))
        os.rmdir(api.BASE_DIR)

    def test_load_dataset(self):
        dataset_path = os.path.join(api.BASE_DIR, "__testing_matrix-synopsis", "__testing_matrix-synopsis.gz")
        if os.path.isdir(api.BASE_DIR):
            shutil.rmtree(api.BASE_DIR)
        self.assertEqual(api.load("__testing_matrix-synopsis", return_path=True), dataset_path)
        shutil.rmtree(api.BASE_DIR)
        self.assertEqual(len(list(api.load("__testing_matrix-synopsis"))), 1)
        shutil.rmtree(api.BASE_DIR)

    def test_load_model(self):
        if os.path.isdir(api.BASE_DIR):
            shutil.rmtree(api.BASE_DIR)
        vector_dead = np.array([
            0.17403787, -0.10167074, -0.00950371, -0.10367849, -0.14034484,
            -0.08751217, 0.10030612, 0.07677923, -0.32563496, 0.01929072,
            0.20521086, -0.1617067, 0.00475458, 0.21956187, -0.08783089,
            -0.05937332, 0.26528183, -0.06771874, -0.12369668, 0.12020949,
            0.28731, 0.36735833, 0.28051138, -0.10407482, 0.2496888,
            -0.19372769, -0.28719661, 0.11989869, -0.00393865, -0.2431484,
            0.02725661, -0.20421691, 0.0328669, -0.26947051, -0.08068217,
            -0.10245913, 0.1170633, 0.16583319, 0.1183883, -0.11217165,
            0.1261425, -0.0319365, -0.15787181, 0.03753783, 0.14748634,
            0.00414471, -0.02296237, 0.18336892, -0.23840059, 0.17924534
        ])
        dataset_path = os.path.join(
            api.BASE_DIR, "__testing_word2vec-matrix-synopsis", "__testing_word2vec-matrix-synopsis.gz"
        )
        model = api.load("__testing_word2vec-matrix-synopsis")
        vector_dead_calc = model.wv["dead"]
        self.assertTrue(np.allclose(vector_dead, vector_dead_calc))
        shutil.rmtree(api.BASE_DIR)
        self.assertEqual(api.load("__testing_word2vec-matrix-synopsis", return_path=True), dataset_path)
        shutil.rmtree(api.BASE_DIR)

    def test_multipart_load(self):
        dataset_path = os.path.join(
            api.BASE_DIR, '__testing_multipart-matrix-synopsis', '__testing_multipart-matrix-synopsis.gz'
        )
        if os.path.isdir(api.BASE_DIR):
            shutil.rmtree(api.BASE_DIR)
        self.assertEqual(dataset_path, api.load("__testing_multipart-matrix-synopsis", return_path=True))
        shutil.rmtree(api.BASE_DIR)
        dataset = api.load("__testing_multipart-matrix-synopsis")
        self.assertEqual(len(list(dataset)), 1)

    def test_info(self):
        data = api.info("text8")
        self.assertEqual(data["parts"], 1)
        self.assertEqual(data["file_name"], 'text8.gz')
        data = api.info()
        self.assertEqual(sorted(data.keys()), sorted(['models', 'corpora']))
        self.assertTrue(len(data['models']))
        self.assertTrue(len(data['corpora']))
        name_only_data = api.info(name_only=True)
        self.assertEqual(len(name_only_data.keys()), 2)
        self.assertTrue({'models', 'corpora'} == set(name_only_data))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    unittest.main()
