# Fairness meets Cross-Domain Learning

## Datasets
1. Download and extract the CelebA "Align&Cropped Images" dataset and the Attributes Annotations from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Download the COVID-19 Chest X-Ray dataset from https://github.com/ieee8023/covid-chestxray-dataset
3. Download the UTKFace "Aligned&Cropped Faces" dataset and the Landmarks Annotations from https://susanqq.github.io/UTKFace/

## Requirements
We used: 
- Python version 3.7.6
- CUDA 11.3

### g-SMOTE Dependencies
Since g-SMOTE official code was not provided we implemented it using HyperInverter (https://github.com/VinAIResearch/HyperInverter). To be able to train and run 
g-SMOTE take a look at the README.md file in the `gsmote/hyperinverter` folder.

### General Dependencies
To install all the required dependencies go to the root folder of this project and run:
```bash
pip install -r requirements.txt
```

## Experiments

### Single-Task Attribute Prediction
Please refer to `parse_args.py` for the complete list of arguments.

The simplest way to run a Single-Task attribute prediction experiment on CelebA is:
```bash
python main.py --experiment=baseline --attribute=3 --protected_attribute=20
```

Note that to run g-SMOTE you need to do: (see `gsmote.py` file for the complete list of arguments)
```bash
cd gsmote/hyperinverter/
python gsmote.py [--arguments...]
```

To run the Single-Task attribute prediction experiment on COVID-19 Chest X-Ray dataset you need to pass the following flag:
```bash
python main.py --use_medical_data [--other_arguments...]
```

### Multi-Task Attribute Prediction
Please refer to `parse_args.py` for the complete list of arguments and check `main_multitask.py` to know which methods are supported.

The simplest way to run the Multi-Task attribute prediction experiment on CelebA is:
```bash
python main.py --multitask [--other_arguments...]
```

### Landmark Detection
To run the landmark detection baseline: (see the file for the complete list of arguments)
```bash
python landmark_detection/baseline.py
```

To run Fish for landmark detection: (see the file for the complete list of arguments)
```bash
python landmark_detection/fish.py
```

To run SWAD for landmark detection: (see the file for the complete list of arguments)
```bash
python landmark_detection/SWAD.py
```