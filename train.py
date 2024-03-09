import os
import pandas as pd
import torch
import torch.utils.data
import numpy as np
import json
import utils
from data_process import preprocess_json
from data_encoder import DataEncoder
import predict_utils
from config import config

#divie all porovide data, use 70% of data for training and rest for prediction
def split_data():
    TRAIN_SIZE = 0.7
    csv_filename = config.encoded_csv
    df = pd.read_csv(csv_filename, header=None)

    msk = np.random.rand(len(df)) < TRAIN_SIZE
    train = df[msk]
    test = df[~msk]

    train.to_csv(config.training_csv, header=False, index=False)
    test.to_csv(config.testing_csv, header=False, index=False)

#class for loading data set 
class Data_Loading(torch.utils.data.Dataset):
    def __init__(self, encoded_features_path):
        print("\nLoading dataset...")
        self.df = pd.read_csv(encoded_features_path, header=None)
        self.unique_labels = sorted(self.df.iloc[:, 0].unique().tolist())
        print("\n{}\n".format(self.df.iloc[:, 0].value_counts()))

    def __getitem__(self, index):
        data = self.df.iloc[index, :]
        val = torch.tensor(list(map(float, data[2:].values)))
        label = torch.tensor(self.unique_labels.index(data[0]))
        mal_hash = data[1]
        return val, mal_hash, label

    def __len__(self):
        return self.df.shape[0]

#GPU acceleration 
class MultilayerPerceptron(torch.nn.Module):

    def __init__(self, layer_size):
        # initialize nn.Module object
        super(MultilayerPerceptron, self).__init__()

        # save parameters
        self.layer_size = layer_size

        # prepare func locally
        self.relu = torch.nn.ReLU()
        self.batchnorm = torch.nn.ModuleList()

        self.layers = torch.nn.ModuleList()
        for i in range(len(self.layer_size) - 1):
            self.layers.append(
                torch.nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            if i < len(self.layer_size) - 2:
                self.batchnorm.append(
                    torch.nn.BatchNorm1d(self.layer_size[i + 1]))

    def forward(self, x):

        # hidden layer
        for i, (layer, batchnorm) in enumerate(zip(self.layers[:-1], self.batchnorm)):
            x = layer(x)
            x = batchnorm(x)  # can only be applied for NxM, where N is batch size & M = data size
            x = self.relu(x)

        # output layer
        x = self.layers[-1](x)

        return x


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nDevice used : {}".format(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    print("Pytorch version: {}".format(torch.__version__))

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # hyper parameters
    num_epochs = 600  # how many iterations for complete single dataset training
    learning_rate = 0.003
    batch_size = 40  # batch per-training
    layer_size = [20, 60, 200, 40, 15, 6]

    enable_checkpoint = True
    # model filename
    checkpoint_name = config.check_point_training

    # load dataset
    malware_train = Data_Loading(encoded_features_path=config.training_csv)
    malware_test = Data_Loading(encoded_features_path=config.testing_csv)

    print("\nSize of training dataset: {}".format(len(malware_train)))
    print("Size of testing dataset: {}\n".format(len(malware_test)))

    train_loader = torch.utils.data.DataLoader(
        malware_train, batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        malware_test, batch_size=batch_size, pin_memory=True, shuffle=False)

    # setup appropriate objects
    mlp = MultilayerPerceptron(layer_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    epoch = 0

    # load previous checkpoint if it exists
    if enable_checkpoint and os.path.exists(checkpoint_name):
        print("Previous checkpoint model found!\n")
        checkpoint = torch.load(checkpoint_name)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        mlp.eval()

    while epoch < num_epochs:
        avg_loss = 0
        for i, (X, _, labels) in enumerate(train_loader):

            mlp.train()  # switch back to train mode

            X, labels = X.to(device), labels.to(device)
            outputs = mlp(X)
            loss = criterion(outputs, labels)
            avg_loss += loss.item()

            optimizer.zero_grad()  # clear our previous calc
            loss.backward()  # calc all parameters gradient
            optimizer.step()  # apply weight tuning based on calculated gradient

            if (i + 1) % 30 == 0:
                mlp.eval()  # turns off dropout and batch normalization
                epoch_fmt = str(epoch).rjust(len(str(num_epochs)))
                batch_fmt = str(i + 1).rjust(len(str(len(train_loader))))
                fmt_str = "Epochs [" + epoch_fmt + "/{}], Batch [" + batch_fmt + "/{}], Loss = {:.6f}"
                print(fmt_str.format(num_epochs, len(train_loader), loss.item()))

        avg_loss /= len(train_loader)
        if (epoch + 1) % 5 == 0:
            print("\nAverage loss for epochs [{}] = {:.8f}\n".format(epoch + 1, avg_loss))

        # test accuracy of model for every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # turns off dropout and batch normalization
                mlp.eval()
                correct_cnt, total_cnt = 0, 0
                for X, mal_hash, labels in test_loader:
                    X, labels = X.float().to(device), labels.to(device)
                    outputs = mlp(X)
                    max_accuracy, pred_label = torch.max(outputs.data, 1)

                    total_cnt += X.cpu().data.size()[0]
                    correct_cnt += (pred_label == labels.data).sum()
                accuracy = correct_cnt.cpu().item() * 1.0 / total_cnt
                print("Test - Epoch {} -- Accuracy : {}\n".format(epoch + 1, accuracy))

        # save model for every 10 iterations -- make sure we don't lost everything
        if enable_checkpoint:
            if (epoch + 1) % 10 == 0:
                print("Saving checkpoint model..\n")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': mlp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_name)

        epoch += 1

    torch.save(mlp.state_dict(), config.trained_model_file)

    mlp.eval()

    predict_list, label_list = [], []
    with torch.no_grad():
        correct_cnt, total_cnt = 0, 0
        try:
            for X, labels in test_loader:
                X, labels = X.float().to(device), labels.to(device)
                outputs = mlp(X)
                _, pred_label = torch.max(outputs.data, 1)
                predict_list.extend(pred_label.cpu().numpy().tolist())
                label_list.extend(labels.cpu().numpy().tolist())
                total_cnt += X.cpu().data.size()[0]
                correct_cnt += (pred_label == labels.data).sum()
            accuracy = correct_cnt.cpu().item() * 1.0 / total_cnt
            print("Final Accuracy = {}\n".format(accuracy))
        except:
            pass
    with open(config.label_text, 'w') as fo:
        fo.write('[' + ','.join(malware_train.unique_labels) + ']')


if __name__ == '__main__':
    split_data()
    train()
