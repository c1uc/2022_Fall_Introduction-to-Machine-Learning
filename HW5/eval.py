import csv
import torch
from torch.utils.data import DataLoader
from model import Model
from dataset import TaskDataset

TEST_PATH = "./test"
device = "cuda"
char_len = [0, 1, 2, 4]


def main():
    test_data = []
    res = []
    with open(f'{TEST_PATH}/../sample_submission.csv', newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            test_data.append(row)

    for i in [1, 2, 3]:
        net = Model(i, device).to(device)
        net.load_state_dict(torch.load(f"./models/task{i}"))
        test_ds = TaskDataset(test_data, root=TEST_PATH, task=i, char_len=char_len[i], return_filename=True)
        test_dl = DataLoader(test_ds, batch_size=500, num_workers=4, drop_last=False, shuffle=False)
        print(f"task{i} evaluating...")
        net.eval()
        for image, filenames in test_dl:
            image = image.to(device)

            pred = net(image)
            mm = torch.argmax(pred[0], dim=1)
            mm = mm.view((1, -1))
            if i == 2:
                mm = torch.stack((
                        torch.argmax(pred[0], dim=1),
                        torch.argmax(pred[1], dim=1)
                    ))
            if i == 3:
                mm = torch.stack((
                        torch.argmax(pred[0], dim=1),
                        torch.argmax(pred[1], dim=1),
                        torch.argmax(pred[2], dim=1),
                        torch.argmax(pred[3], dim=1)
                    ))

            for ii in range(len(filenames)):
                p = ''.join([TaskDataset.LABEL2CHAR[_.item()] for _ in mm.T[ii]])
                res.append([filenames[ii], p])

    print('writing results...')
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(res)
    print('output done.')


if __name__ == '__main__':
    main()
