# DL4TSF

## Description

TO COMPLETE

## Getting started

```
git clone git@gitlab.com:quantmetry/expertises/time-series/dl4tsf.git
cd dl4tsf
```
## Installation

### Miniconda
If new machine and you do not have miniconda, start by installing it using this :
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### Install conda environment

This command will install all packages in environment.yml

`conda env create -f environment.yml`

Once installed, activate it using this command:

`conda activate dl4tsf`


Please make sure whenever you add a new package to add it manually to `environment.yaml` file according to alphabetical order

To update the existing environment, launch this command:

`conda env update --file environment.yml --prune`

### Install pre-commit

Once the conda environment is activated, install pre-commit using this command

`pre-commit install`

## Retrieve Data
Make sure you never add data files to the git repository.
The current data folders that must be available for a proper testing of the modules are available here:
[Quantmetry sharepoint](https://quantmetryparis.sharepoint.com/:f:/s/QM-Capitalisation-INT/Ei45bH_tU6FDh5msNzS0bvsBnNj69EwRq64W63tBcwFhRw?e=z9buEp)

The current data needed are:
- climate_delhi
- energy
- enedis
- all_weather (The mapping between the station name and the station number is available here: [posteSynop.csv](https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/postesSynop).csv ))

# Jupyter notebooks
Make sure that code done in jupyter notebook are only for testing or visualization.
ipynb files are not accepted in git. Please transform to md using jupytext commands below.

## Create new jupyter notebook server

`nohup jupyter notebook &`
- Nohup, short for no hang. meaning the server will not shutdown when you exit the cmd.
- By appending the & operator to any command, you dictate the shell to execute that Linux command in the background so that you can continue using the shell untethered

To check current active notebook servers:

`jupyter notebook list`

To stop a notebook server : (always make sure you have only one server active)

`jupyter notebook stop <port number>`

## Jupytext
### Transform notebooks from md to ipynb and vice versa
```
jupytext --to ipynb notebook_1.md
jupytext --to md - notebook.ipynb
```

### To sync notebook with md
`
jupytext --sync notebook.ipynb
`
# Hydra
The configs files are based on the library hydra. Please refer to https://hydra.cc/docs/intro/ for more info.

## Test and Deploy

### Available working models
Currently the following models are implemented and tested:
- TFT

### Available working datasets
Currently the following datasets are implemented and tested:
- traffic
- climate_delhi
- energy
- enedis

### Modification to make in .yaml files
Dataset:
- Modifiy the lists in 'name_feats' with the variables name corresponding to the different features


### Testing the train/forecast script:
Run the following command to train and forecast using TFT:
(for now it takes so much time to forecast traffic data, so avoid it)
```
python dl4tsf/train.py model=tft dataset=traffic
# or
python dl4tsf/train.py model=tft dataset=climate_delhi
# or
python dl4tsf/train.py model=tft dataset=energy
# or
python dl4tsf/train.py model=tft dataset=enedis
```
### PENDING

```
python dl4tsf/train.py model=informer dataset=energy
# or
python dl4tsf/train.py model=informer dataset=climate_delhi
```
If you encountered a warning (and you have only a CPU), run this command (to fallback to CPU if no gpu is available):
`export PYTORCH_ENABLE_MPS_FALLBACK=1`

### Launching Tensorboard
TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.

To check the loss evolution during training run this command:

`tensorboard --logdir tensorboard_logs/ --host=localhost --port=8889`

then open this link: [localhost:8889/](localhost:8889/)


## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
