# CalMal: Malware-Behavior Clustering

## Overview

CalMal is a project focused on detecting and classifying malware behavior using machine learning techniques. It assumes the availability of a dataset in JSON format within the "data/json" directory, which can be modified in the `config.ini` file.

### Requirements

- Python 3
- Docker (optional, for Docker-based setup)
- Git

## Installation

### Without Docker

1. **Clone the Repository**

```bash
git clone git@github.com:unknownhad/CalMal.git
cd CalMal
```

 # Install Poetry : 
Follow the instructions at [Python Poetry Documentation](https://python-poetry.org/docs/) to install Poetry on your machine.

# Setup the Project Environment:

```
poetry shell
poetry install
```

# Running the Application:
```
poetry run python app.py
```
Access the web service by navigating to http://localhost:1234 in your browser. You can test predictions by uploading a JSON file.

For training : 
Put all the JSON from VirusTotal to  /data/json then run 

`poetry run python data_process.py`

This will process the data and make it consumeable 

After that run :
`poetry run python data_encoder.py`
This will encode the baove data ot generate csv file.

Example output :

```
(calmal-py3.11) bash-3.2$ poetry run python data_encoder.py

Device used : cpu
Pytorch version: 2.2.1

Loading dataset from: /CalMal/result/temporary/dataset.csv.xz

0
Name: count, dtype: int64

Epochs [  1/600], Batch [ 5/25], Loss = 0.04834136
Epochs [  1/600], Batch [10/25], Loss = 0.03662824
Epochs [  1/600], Batch [15/25], Loss = 0.03420896
Epochs [  1/600], Batch [20/25], Loss = 0.02952765
......................Trimmed......................
......................Trimmed......................


```

After that run 

`poetry run python train.py`

For training the model and finding the aquracy.

```
(calmal-py3.11) bash-3.2$ poetry run python train.py

Device used : cpu
Pytorch version: 2.2.1

Size of training dataset: 857
Size of testing dataset: 349

Previous checkpoint model found!

Final Accuracy = 0.0057306590257879654
```


### With Docker



Visualization Result:
![Result image](visualize.png)

###  Contribution guideline
Contributions to CalMal are welcome! Please follow the established coding and commit message guidelines. For more details, refer to the contribution guide in the repository.


### Contact

For questions or contributions, please open an issue or a pull request in the GitHub repository.