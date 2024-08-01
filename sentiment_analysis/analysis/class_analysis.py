import numpy as np


def main():
    test_data = np.load("./data/training.npz")
    labels = test_data["labels"]
    unique, counts = np.unique(labels, return_counts=True)
    print(unique)
    print(counts)
    print(counts / np.sum(counts))


if __name__ == "__main__":
    main()
