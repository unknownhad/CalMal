import os
import configparser


def create_path(fn):
    def _wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if not os.path.exists(result):
            os.makedirs(result, exist_ok=True)
        return result

    return _wrapper


class Config:
    def __init__(self):
        self.root = os.getcwd() + '/'
        self.parser = configparser.ConfigParser()
        self.parser.read(self.root + 'config.ini')

    @property
    @create_path
    def output_path(self):
        return self.root + parser['PATH']['output']

    @property
    def input_path(self):
        return self.root + parser['PATH']['input']

    @property
    def get_json_data_path(self):
        return self.root + parser['TRAIN']['json_data']

    @property
    @create_path
    def temp_data(self):
        return self.root + parser['TRAIN']['temp_data']

    @property
    def training_csv(self):
        return os.path.join(self.temp_data, 'train_dataset.csv')

    @property
    def testing_csv(self):
        return os.path.join(self.temp_data, 'test_dataset.csv')

    @property
    @create_path
    def model_directory(self):
        return self.root + parser['TRAIN']['model_data']

    @property
    def check_point(self):
        return os.path.join(self.model_directory, 'checkpoint-mal.pt')

    @property
    def check_point_training(self):
        return os.path.join(self.model_directory, 'checkpoint-MLP.pt')

    @property
    def model_file(self):
        return os.path.join(self.model_directory, 'mal-Trained-Model.pt')

    @property
    def trained_model_file(self):
        return os.path.join(self.model_directory, 'MLP-Trained-Model.pt')

    @property
    def encoded_csv(self):
        return os.path.join(self.model_directory, "encoded-form-mal.csv")

    @property
    def label_text(self):
        return os.path.join(self.model_directory, 'malware-label-index.txt')

    @property
    def unigram_path(self):
        return os.path.join(self.temp_data, 'top_unigrams.txt')

    @property
    def dataset_path(self):
        return os.path.join(self.temp_data, 'dataset.csv.xz')

    @property
    def training_params(self):
        return self.parser['TRAIN']


config = Config()
parser = config.parser

