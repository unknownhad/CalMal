import os
import torch
import torch.utils.data
import json
import utils
from data_encoder import DataEncoder
import predict_utils
from config import config


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


def test(file):
    print("\nDevice used : {}".format(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    print("Pytorch version: {}".format(torch.__version__))

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # hyper parameters
    layer_size_dae = [10000, 3000, 500, 100, 20, 100, 500, 3000, 10000]
    layer_size = [20, 60, 200, 40, 15, 6]

    # model filename
    checkpoint_name = config.check_point_training

    dae = utils.initialize_model(
        DataEncoder, layer_size_dae,
        config.model_file)

    mlp = utils.initialize_model(
        MultilayerPerceptron, layer_size,
        config.trained_model_file)

    # load checkpoint if it exists
    if os.path.exists(checkpoint_name):
        print("Previous checkpoint model found!\n")
        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_name)
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.eval()
    else:
        print("Checkpoint model can not be bound.")
        return

    vt_json_data = open(file, 'r', encoding='utf-8').read()

    if 'additional_info' not in vt_json_data:
        print('PredictError: Not a valid behaviors data file!')
        return

    vt_json_data = json.loads(vt_json_data)
    label_text = open(config.label_text).readline()[1:-1].split(',')

    top_unigrams = utils.load_unigram(config.unigram_path)

    malware_unigrams = predict_utils.preprocess_json(vt_json_data)

    malware_bitstr = predict_utils.unigrams_to_bitstring(malware_unigrams, top_unigrams)

    malware_sign = predict_utils.gen_signs_from_bitstring(
        dae, malware_bitstr)

    # using generated signature for prediction
    prediction_probability, malware_predicted = predict_utils.predict_from_malware_sign(
        mlp, malware_sign, label_text
    )
    print("{}: {}".format(malware_predicted, prediction_probability[malware_predicted]))
    return malware_predicted, prediction_probability[malware_predicted]


if __name__ == '__main__':
    test('./data/Cerber_0a2a6c298656d5b8f886580792108de1c669f2afabb52afbebce954cd279b8b4.json')
