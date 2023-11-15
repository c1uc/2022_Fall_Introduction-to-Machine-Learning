import csv
import random

TRAIN_PATH = "./train"

with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        print(row)

