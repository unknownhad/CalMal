
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import config


def main():

    df = pd.read_csv(config.dataset_path)

    sample1 = df.sample(1).values.tolist()[0]
    sample2 = df.sample(1).values.tolist()[0]

    d1 = np.fromiter(sample1[2:], dtype=np.uint8) + 255
    d2 = np.fromiter(sample2[2:], dtype=np.uint8) + 255

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    ax = axs[0]
    ax.imshow(np.reshape(d1, (100, 100)), cmap='gray_r')
    ax.set_title(sample1[0], fontweight="bold", size=20)
    
    ax = axs[1]
    ax.imshow(np.reshape(d2, (100, 100)), cmap='gray_r')
    ax.set_title(sample2[0], fontweight="bold", size=20)

    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()