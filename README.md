# federated-decision-forest-inference
A decision forest inference library using Quickscorer algorithm in vertical federated learning setting.

How to setup the environment:
It is recommeneded to use `conda` to manage your virtual environment.
First, install `git` and `Miniconda`. Make sure your can run `git` and `conda` commands on the terminal. Then go to a folder that you want to download and edit the code and run the following commands
```
git clone https://github.com/asu-cactus/federated-decision-forest-inference.git
cd federated-decision-forest-inference
conda conda env create -f environment.yml
```
The last command create a new conda environment named "fed" and install the required python packages to run the program. You can run `conda activate fed` to activate this environment.
Note that `black` is installed to as the python formatter to make it look more delightful to our human eyes. You may need to activate `black` in your IDE.