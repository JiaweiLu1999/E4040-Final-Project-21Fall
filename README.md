# Introduction to this repo
This repository contains the test code, train code, and pre-trained models for E4040 final project "A New Backbone for Hyperspectral Image Construction and Improvement based on Mask Mixture Training and Energy Normalization". This repository firstly reimplemented the SSI-ResU-Net from arXiv paper "A New Backbone for Hyperspectral Image Reconstruction" and get the pre-trained model v3_mask1. Next, we applied mask mixture training, energy normalization, and a combination of these two techniques to train models, resulting in the pre-trained model v3_mix, v3_en_mask1, and v3_en_mix respectively. The testing result proves that our modifications can improve the generalizing ability of SSI-ResU-Net.

# File Organization
```
.
├── Data
│   ├── cave
│   ├── mask1.mat
│   ├── mask2.mat
│   ├── mask3.mat
│   ├── mask4.mat
│   └── testing
│       └── simu
│           ├── scene1.npy
│           ├── scene10.npy
│           ├── scene2.npy
│           ├── scene3.npy
│           ├── scene4.npy
│           ├── scene5.npy
│           ├── scene6.npy
│           ├── scene7.npy
│           ├── scene8.npy
│           └── scene9.npy
├── E4040.2021Fall.YZJH.report.ys3493.zy2501.jl5999.pdf
├── Model_Test.ipynb
├── Model_Train.ipynb
├── README.md
├── models
│   ├── v3_en_mask1
│   ├── v3_en_mix
│   ├── v3_mask1
│   └── v3_mix
├── requirements.txt
└── utils
    ├── __pycache__
    ├── model.py
    ├── train.py
    └── utils.py

6 directories, 27 files
```
## Detailed description about code organization

### Jupyter Notebooks 
Here are two main jupyter notebooks to run:

`Model_Train.ipynb` works for training of different models. It is not runnable directly since training data is missing. You should download training data from the google drive link we provided following and unzip them to the right directory before running this notebook.

`Model_Test.ipynb` works for the test over test data and visualize the results. It is directly runnable since we have included models and testing data inside this repo.

### Data
This directory contains the training dataset, testing dataset, and masks that used in the training. We have included four masks mask1-4.mat in ./Data and testing data in ./Data/testing. The training dataset is too large to upload to Github, so we upload it on google drive and share it by this link: https://drive.google.com/drive/folders/1fiGZQAxybwC6xMs_FP0L7yExIVXRDjf_?usp=sharing. You should download the directory named '**cave**' and unzip it to ./Data/cave. After that, you can run the training code provided. 

### models
This directory includes all four models whose performance are listed in the final report. `v3_mask1` is the model to reimplement the original work, `v3_mix` is the model applied mask mixture training, `v3_en_mask1` is the model applied energy normalization and `v3_en_mix` is the model applied with both two tricks. You can run Model_Test.ipynb to reproduce the results.

### utils
This directory includes functions that are imported in two jupyer notebooks to realize training and testing.

`model.py` contains model definition

`train.py` contains the training process for every model

`utils.py` contains all utilization functions


# Code running instruction
If you want to reproduce the results provided in the final report, you can just run Model_Test.ipynb. It should give out the same results by prints on the screen.

If you want to reproduce the training process, you should first download training data provided by the link above, unzip them in ./Data/cave, then run Model_Train.ipynb. There are two flags named `energy_flag` and `mix_flag` that are initialized as True in the notebook. You can change them to train different models with or without energy normalization and mix masks training.
