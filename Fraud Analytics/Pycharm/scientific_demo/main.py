import numpy as np

from helpers.downloader import download
from models.classifier import TheAlgorithm
from helpers.split import split


if __name__ == '__main__':
    X, y = download()
    print('MNIST:', X.shape, y.shape)

    splitRatio = 60000
    X_train, y_train, X_test, y_test = split(X, y, splitRatio)

    np.random.seed(31337)
    ta = TheAlgorithm(X_train, y_train, X_test, y_test)
    train_accuracy = ta.fit()
    print()
    print('Train Accuracy:', train_accuracy, '\n')
    print("Train confusion matrix:\n%s\n" % ta.train_confusion_matrix)
    # np.save('/Users/datdarylngo/PycharmProjects/scientific_demo/tests/fixtures/train_confusion_matrix_fixture', ta.train_confusion_matrix)

    test_accuracy = ta.predict()
    print()
    print('Test Accuracy:', test_accuracy, '\n')
    print("Test confusion matrix:\n%s\n" % ta.test_confusion_matrix)
    # np.save('/Users/datdarylngo/PycharmProjects/scientific_demo/tests/fixtures/test_confusion_matrix_fixture', ta.test_confusion_matrix)
