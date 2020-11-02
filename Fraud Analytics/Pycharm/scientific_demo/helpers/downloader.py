from sklearn.datasets import fetch_openml


def download():
    mnist = fetch_openml('mnist_784')
    X = mnist.data
    y = mnist.target
    return X, y
