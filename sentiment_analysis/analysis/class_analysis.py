import numpy as np


def main():
    test_data = np.load("./data/training.npz")
    labels = test_data["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    print(unique)
    print(counts)
    print(counts / np.sum(counts))


def temp():
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.linspace(0, 100, 100)
    ys = np.tanh(xs / 30) * 30

    plt.plot(xs, ys)
    plt.show()


if __name__ == "__main__":
    temp()
