# CIFAR10-ResNet
A ResNet implementation for CIFAR10

# Making the paper
```
git checkout master
cd paper/
pdflatex ./nips_2017.tex
bibtex ./nips_2017.aux
pdflatex ./nips_2017.tex
xdg-open ./nips_2017.pdf
```
# Getting the Data
Simply run `python3 cifar10_download_and_extract.py`

This will downlaod the CIFAR10 dataset into the working directory in a subfolder named "cifar10_data"

# Running the Model
- Run `python3 cifar10_main.py`
- In another terminal run `tensorboard --logdir ./cifar10_model`
- Finally, point your web browser at `localhost:6006`

# Branch for Implementations

### master: ResNet
### lm-net: LM-ResNet

