import os
import torch
import torch.utils.data
import pandas as pd
import numpy as np
from config import config


class Data_Loading(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        print(f"\nLoading dataset from: {dataset_path}")
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.labels_unique = self.df.iloc[:, 0].unique().tolist()
        print("\n{}\n".format(self.df.iloc[:, 0].value_counts()))

    def __getitem__(self, index):
        try:
            data = self.df.iloc[index, :]
            val = torch.tensor(list(map(int, data[2:])))  # Assuming all values from 2nd column are numeric
            label = data.iloc[0]
            mal_hash = data.iloc[1]
            return val, mal_hash, label
        except Exception as e:
            print(f"Error processing row {index} in file {self.dataset_path}")
            print("Row data:", data)
            raise e

    def __len__(self):
        return self.df.shape[0]



class DataEncoder(torch.nn.Module):
    def __init__(self, layer_size):
        # must have at least [input, hidden, output] layers
        assert len(layer_size) >= 3
        assert layer_size[0] == layer_size[-1]  # input equals output
        assert len(layer_size) % 2 == 1  # must have odd number of layers

        super(DataEncoder, self).__init__()
        self.layer_size = layer_size
        self.relu = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        for i in range(len(self.layer_size) - 1):
            self.layers.append(torch.nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            if i < len(self.layer_size) - 2:
                self.batchnorm.append(torch.nn.BatchNorm1d(self.layer_size[i + 1]))

    def forward(self, x):
        encoded = None
        for i, (layer, batch_norm) in enumerate(zip(self.layers[:-1], self.batchnorm)):
            x = layer(x)
            x = batch_norm(x)  # can only be applied for NxM, where N = batch size & M = data size
            x = self.relu(x)
            if i == len(self.layer_size) // 2 - 1:  # get middle (thus encoded data)
                encoded = x

        decoded = self.layers[-1](x)
        return encoded, decoded


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nDevice used : {}".format('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Pytorch version: {}".format(torch.__version__))

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # hyper parameters
    num_epochs = 600  # how many iterations for complete single dataset training
    learning_rate = 0.001
    batch_size = 50  # batch per-training
    enable_checkpoint = True
    denoise_ratio = 0.2
    checkpoint_name = config.check_point
    layer_size = [10000, 3000, 500, 100, 20, 100, 500, 3000, 10000]  # layers size

    # load dataset
    malware_data = Data_Loading(dataset_path=config.dataset_path)
    train_loader = torch.utils.data.DataLoader(malware_data, batch_size=batch_size, pin_memory=True, shuffle=True)

    # setup appropriate objects
    mals = DataEncoder(layer_size).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mals.parameters(), lr=learning_rate)
    epoch = 0

    # load previous checkpoint if it exists
    if enable_checkpoint:
        if os.path.exists(checkpoint_name):
            print("Previous checkpoint model found!\n")
            checkpoint = torch.load(checkpoint_name)
            mals.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            mals.eval()

    def denoise(x, ratio):
        noise = np.random.binomial(1, ratio, size=x[0].shape[0])
        noise = torch.tensor(noise).float().to(device)
        return (x + noise) % 2

    # train our model
    while epoch < num_epochs:
        avg_loss = 0
        for i, (X, _, _) in enumerate(train_loader):

            mals.train()  # switch back to train mode

            x = X.float().to(device)
            x_noise = denoise(x, denoise_ratio)  # denoise input data
            _, outputs = mals(x_noise)
            loss = criterion(outputs, x)
            avg_loss += loss.item()

            optimizer.zero_grad()  # clear our previous calc
            loss.backward()  # calc all parameters gradient
            optimizer.step()  # apply weight tuning based on calculated gradient

            if (i + 1) % 5 == 0:
                mals.eval()  # turns off dropout and batch normalization
                epoch_fmt = str(epoch + 1).rjust(len(str(num_epochs)))
                batch_fmt = str(i + 1).rjust(len(str(len(train_loader))))
                fmt_str = "Epochs [" + epoch_fmt + "/{}], Batch [" + batch_fmt + "/{}], Loss = {:.8f}"
                print(fmt_str.format(num_epochs, len(train_loader), loss.item()))

        avg_loss /= len(train_loader)
        if (epoch + 1) % 5 == 0:
            print("\nAverage loss for epochs [{}] = {:.8f}".format(epoch + 1, avg_loss))

        # generate compressed malware output for testing
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # turns off dropout and batch normalization
                mals.eval()
                # save encoded form for all entire dataset
                filename = config.encoded_csv
                encoded_data = []  # x, hash and label combined
                # calculate total losses from all batches
                for X, mal_hash, label in train_loader:
                    x = X.float().to(device)
                    encoded, _ = mals(x)
                    for each_encoded, each_hash, each_label in zip(encoded.tolist(), mal_hash, label):
                        encoded_data.append([each_label] + [each_hash] + each_encoded)
                # export current compressed file into csv for previewing
                print("\nExporting encoded malware form into csv..\n")
                encoded_df = pd.DataFrame(encoded_data)
                encoded_df.to_csv(filename, index=False, header=False)

        # save model for every 10 iterations -- make sure we don't lost everything
        if enable_checkpoint:
            if (epoch + 1) % 10 == 0:
                print("\nSaving checkpoint model..\n")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': mals.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_name)

        epoch += 1

    torch.save(mals.state_dict(), config.model_file)


if __name__ == '__main__':
    main()
