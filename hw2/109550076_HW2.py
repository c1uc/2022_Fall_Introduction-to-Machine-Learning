import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def fld():
    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)

    m1 = np.sum(np.stack((y_train, y_train), 1) * x_train, 0).reshape(-1, 1) / np.sum(y_train)
    m2 = np.sum(np.stack(((1 - y_train), (1 - y_train)), 1) * x_train, 0).reshape(-1, 1) / np.sum(1 - y_train)
    print(f"mean vector of class 1:\n{m1}\n", f"mean vector of class 2:\n{m2}\n")

    sw = np.sum(
        [
            np.dot((x_train[i].reshape(-1, 1) - m1), (x_train[i].reshape(-1, 1) - m1).T) if y_train[i] == 1 # (xn - m1)^2 for class 1
            else np.dot((x_train[i].reshape(-1, 1) - m2), (x_train[i].reshape(-1, 1) - m2).T)  # (xn - m2)^2 for class 2
            for i in range(y_train.size)
         ], 0
    )
    print(f"Within-class scatter matrix SW:\n{sw}\n")

    sb = np.dot((m2 - m1), (m2 - m1).T)  # Sb = (m2 - m1) * (m2 - m1).T
    print(f"Between-class scatter matrix SB:\n{sb}\n")

    sw_inv = np.linalg.inv(sw)
    w = np.dot(sw_inv, (m2 - m1))
    w /= np.linalg.norm(w)
    print(f" Fisher’s linear discriminant:\n{w}\n")
    SLOPE = w[1]/w[0]
    plt.plot([-6 * w[0], 6 * w[0]], [-6 * w[1], 6 * w[1]])
    w = w.reshape(2, )
    projection = []
    for i in range(y_train.size):
        pr = w * np.dot(w, x_train[i])
        projection.append(pr)
        if y_train[i] == 1:
            plt.plot([pr[0], x_train[i][0]], [pr[1], x_train[i][1]], 'bo', ms=3, alpha=0.9)
            plt.plot([pr[0], x_train[i][0]], [pr[1], x_train[i][1]], 'c-', alpha=0.2)
        else:
            plt.plot([pr[0], x_train[i][0]], [pr[1], x_train[i][1]], 'ro', ms=3, alpha=0.9)
            plt.plot([pr[0], x_train[i][0]], [pr[1], x_train[i][1]], 'c-', alpha=0.2)
    plt.title(f"Projection Line: w={SLOPE}, b=0")
    plt.axis('scaled')
    plt.show()

    projection = np.asarray(projection)
    y_pred = []
    for x in x_test:
        p = [0, 0]  # the # of neighbor labeled with 0, 1
        pred = []  # predicted label
        pr = w * np.dot(w, x)  # project the test data and calculate the distance
        sub = projection - pr
        dist = np.linalg.norm(sub, axis=1)
        top = np.argsort(dist)  # arg sort and take the closest K points
        for j in range(5):
            p[y_train[top[j]]] += 1
            pred.append(np.argmax(p))  # take the result with more neighbor
        y_pred.append(pred)

    y_pred = np.asarray(y_pred).T
    [print(f"Accuracy for K={i+1}: {accuracy_score(y_test, y_pred[i])}") for i in range(5)]  # print the result in ninja mode

    return


if __name__ == "__main__":
    fld()
