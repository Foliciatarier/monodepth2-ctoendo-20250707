# monodepth2-ctoendo-20250707
monodepth2 on synthetic bronchoscopy image dataset

Code modified from https://github.com/nianticlabs/monodepth2/tree/master
Monodepth2 original paper https://arxiv.org/abs/1806.01260

Deleted some unnecessary content for this project
Added the dataloader code for synthetic bronchoscopy image dataset, located in "datasets/ctoendo_dataset.py"
Partially rewrote the training and testing code, renaming to "train.py" and "test.py"
Modified "options.py"

The synthetic bronchoscopy image dataset is located in "./data/CToEndo", which was generated from ATM22 lung CT dataset
Scripts is located in "./scripts"


requirements

numpy
matplotlib
pytorch
torchvision
progressbar
