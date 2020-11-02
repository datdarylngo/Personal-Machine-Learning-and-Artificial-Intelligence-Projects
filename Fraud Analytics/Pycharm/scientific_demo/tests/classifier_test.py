import unittest
import numpy as np

from helpers.downloader import download
from helpers.split import split
from models.classifier import TheAlgorithm

# test variables and fixtures
train_fixture = np.load(
    '/Users/datdarylngo/PycharmProjects/scientific_demo/tests/fixtures/train_confusion_matrix_fixture.npy')
test_fixture = np.load(
    '/Users/datdarylngo/PycharmProjects/scientific_demo/tests/fixtures/test_confusion_matrix_fixture.npy')

train_accuracy = 72.65166666666667
test_accuracy = 73.18
splitRatio = 60000
seed = 31337


class TestInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # print('setupClass')
        pass

    @classmethod
    def tearDownClass(cls):
        # print('teardownClass')
        pass

    def setUp(self):
        print('setUp')
        X, y = download()
        self.X_train, self.y_train, self.X_test, self.y_test = split(X, y, splitRatio)
        self.train_accuracy = train_accuracy
        self.train_confusion_matrix = train_fixture
        self.test_accuracy = test_accuracy
        self.test_confusion_matrix = test_fixture

    def tearDown(self):
        # print('tearDown')
        pass

    def test_fit(self):
        np.random.seed(seed)
        self.ta = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.assertEqual(self.ta.fit(), self.train_accuracy)
        self.assertEqual(self.ta.train_confusion_matrix.tolist(), self.train_confusion_matrix.tolist())

    def test_predict(self):
        np.random.seed(seed)
        self.ta = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)
        self.ta.fit()
        self.assertEqual(self.ta.predict(), self.test_accuracy)
        self.assertEqual(self.ta.train_confusion_matrix.tolist(), self.train_confusion_matrix.tolist())


if __name__ == '__main__':
    # run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
