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

After that run 

`poetry run python train.py`

For training the model.

### With Docker



Visualization Result:
![Result image](visualize.png)

###  Contribution guideline
Contributions to CalMal are welcome! Please follow the established coding and commit message guidelines. For more details, refer to the contribution guide in the repository.


### Contact

For questions or contributions, please open an issue or a pull request in the GitHub repository.