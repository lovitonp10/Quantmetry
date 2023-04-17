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



### Install pre-commit

Once the conda environment is activated, install pre-commit using this command

`pre-commit install`

## Retrieve Data
Make sure you never add data files to the git repository.
The current data folders that must be available for a proper testing of the modules are available here:
https://quantmetryparis.sharepoint.com/:f:/s/QM-Capitalisation-INT/Ei45bH_tU6FDh5msNzS0bvsBnNj69EwRq64W63tBcwFhRw?e=z9buEp

The current data needed are:
- climate_delhi


## Test and Deploy

### Available working models
Currently the following models are implemented and tested:
- TFT

### Available working datasets
Currently the following datasets are implemented and tested:
- traffic
- climate_delhi

### Testing the train/forecast script:
Run the following command to train and forecast using TFT:
(for now it takes so much time to forecast traffic data, so avoid it)
```
python dl4tsf/train.py model=tft dataset=traffic
# or
python dl4tsf/train.py model=tft dataset=climate_delhi
```




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
