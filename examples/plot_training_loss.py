import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("files", type=str, nargs="+")
    parser.add_argument("-names", type=str, nargs="+")

    args = parser.parse_args()

    file_list = args.files
    names = args.names

    print(file_list, names)
    assert len(file_list)==len(names)

    for file_path,file_name in zip(file_list,names):
        loss = []
        with open(file_path, 'r') as file:
            data = file.readlines()
            csvdata = csv.DictReader(data)
            for row in csvdata:
                loss.append(float(row["train_loss"]))
        plt.plot(np.arange(len(loss)), np.asarray(loss), label=file_name)

plt.yscale('symlog')
plt.ylabel("Training loss")
plt.xlabel("Epochs")
plt.legend(loc='best')
plt.show()
