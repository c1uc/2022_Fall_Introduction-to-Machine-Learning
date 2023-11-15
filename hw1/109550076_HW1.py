import argparse
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.01
LOSS = []


def p1():
    def mean_square_error(x, y):
        return np.sum((x - y) ** 2) / x.size

    def gradient_descent(x: np.ndarray, y: np.ndarray, m: int, alpha: float):
        theta = np.random.random(2)
        x = np.concatenate((x, np.ones((m, 1))), axis=1)  # (x, 1)
        xt = x.transpose()
        prev_cost = None
        while True:
            hypo = np.dot(x, theta)
            loss = y - hypo

            cost = mean_square_error(hypo, y)
            LOSS.append(cost)

            if prev_cost == cost:  # converge
                break
            # print(f"COST: {cost}")
            prev_cost = cost
            grad = np.dot(xt, loss) / m  # grad = (t - y) * x / m
            theta += alpha * grad
        return theta

    x_train, x_test, y_train, y_test = np.load('regression_data.npy', allow_pickle=True)

    m = x_train.size
    w = gradient_descent(x_train, y_train, m, ALPHA)

    m = x_test.size
    x_pred = np.concatenate((x_test, np.ones((m, 1))), axis=1)
    y_pred = np.dot(x_pred, w)
    print(f"weight: {w[0]}, intersection: {w[1]}")
    print(f"mean_square_error: {mean_square_error(y_pred, y_test)}")

    plt.subplot(2, 2, 1)
    plt.title("training loss")
    plt.plot(LOSS, label="training loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("data")
    plt.plot(x_train, y_train, '.', color="xkcd:ocean blue", label="training data")
    plt.plot(x_test, y_test, '.', color="xkcd:electric pink", label="testing data")
    plt.plot(x_test, y_pred, '.', color="xkcd:lemon lime", label="prediction")
    plt.legend()


def p2():
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    def hypothesis(x, theta):
        return sigmoid(np.dot(x, theta))

    def cross_entropy_error(x, y):
        return -np.sum(y * np.log(x + 1e-100) + (1-y) * np.log(1-x + 1e-100))  # plus 1e-100 to prevent x too close to 0

    def gradient_descent(x: np.ndarray, y: np.ndarray, m: int, alpha: float):
        theta = np.random.random(2)
        x = np.concatenate((x, np.ones((m, 1))), axis=1)  # (x, 1)
        xt = x.transpose()
        prev_cost = None
        while True:
            hypo = hypothesis(x, theta)
            loss = hypo - y

            cost = cross_entropy_error(hypo, y)
            LOSS.append(cost)
            if prev_cost == cost:  # converge
                break
            # print(f"COST: {cost}")
            prev_cost = cost

            grad = np.dot(xt, loss)  # grad = sum((hypo - t) * x)
            theta -= alpha * grad
        return theta

    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
    m = x_train.size
    w = gradient_descent(x_train, y_train, m, ALPHA)

    m = x_test.size
    x_pred = np.concatenate((x_test, np.ones((m, 1))), axis=1)
    y_pred = hypothesis(x_pred, w)
    print(f"weight: {w[0]}, intersection: {w[1]}")
    print(f"cross_entropy_error: {cross_entropy_error(y_pred, y_test)}")

    plt.subplot(2, 2, 3)
    plt.title("training loss")
    plt.plot(LOSS, label="training loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("data")
    plt.plot(x_train, y_train, '.', color="xkcd:ocean blue", label="train")
    plt.plot(x_test, y_test, '.', color="xkcd:electric pink", label="test")
    plt.plot(x_test, (y_pred > 0.5).astype(int), '.', color="xkcd:lemon lime", label="prediction")
    plt.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="109550076 ML hw1")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p1", action="store_true", help="to run only p1")
    group.add_argument("-p2", action="store_true", help="to run only p2")
    parser.add_argument("-o", help="output figure")
    args = parser.parse_args()

    plt.figure(figsize=(8, 8), dpi=100)

    if args.p1 or not args.p2:
        print("p1:")
        p1()

    LOSS.clear()

    if args.p2 or not args.p1:
        print("p2:")
        p2()

    if args.o:
        plt.savefig(args.o)

    if not args.o:
        plt.show()
