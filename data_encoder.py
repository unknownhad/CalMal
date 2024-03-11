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
        data = self.df.iloc[index, :]
        val = torch.tensor([float(x) for x in data[2:]], dtype=torch.float32)
        label = data.iloc[0]  # Assuming label is the first column
        mal_hash = data.iloc[1]  # Assuming hash is the second column
        return val, mal_hash, label

    def __len__(self):
        return self.df.shape[0]

class DataEncoder(torch.nn.Module):
    def __init__(self, layer_size):
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
        for i, (layer, batch_norm) in enumerate(zip(self.layers[:-1], self.batchnorm)):
            x = layer(x)
            x = batch_norm(x)
            x = self.relu(x)
        decoded = self.layers[-1](x)
        return decoded

def custom_collate_fn(batch):
    batch_tensors = [item[0] for item in batch]  # Numeric tensor data
    batch_hashes = [item[1] for item in batch]   # Malware hashes (strings)
    batch_labels = [item[2] for item in batch]   # Labels (could be numeric or strings)
    
    # Collate the tensors using the default collate function
    batch_tensors_collated = default_collate(batch_tensors)
    
    return batch_tensors_collated, batch_hashes, batch_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nDevice used : {}".format('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Pytorch version: {}".format(torch.__version__))

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    num_epochs = 600
    learning_rate = 0.001
    batch_size = 50
    enable_checkpoint = True
    denoise_ratio = 0.2
    checkpoint_name = config.check_point
    layer_size = [10000, 3000, 500, 100, 20, 100, 500, 3000, 10000]

    malware_data = Data_Loading(dataset_path=config.dataset_path)
    train_loader = torch.utils.data.DataLoader(malware_data, batch_size=batch_size, pin_memory=True, shuffle=True, collate_fn=custom_collate_fn)

    mals = DataEncoder(layer_size).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mals.parameters(), lr=learning_rate)
    epoch = 0

    if enable_checkpoint and os.path.exists(checkpoint_name):
        print("Previous checkpoint model found!\n")
        checkpoint = torch.load(checkpoint_name)
        mals.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        mals.eval()

    while epoch < num_epochs:
        avg_loss = 0
        for i, (X, _, _) in enumerate(train_loader):
            mals.train()
            x = X.to(device)
            x_noise = denoise(x, denoise_ratio)
            outputs = mals(x_noise)
            loss = criterion(outputs, x)
            avg_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss /= len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"\nAverage loss for epochs [{epoch + 1}] = {avg_loss:.8f}")

        epoch += 1

    torch.save(mals.state_dict(), config.model_file)

if __name__ == '__main__':
    main()
