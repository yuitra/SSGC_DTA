# dependencies
numpy == 1.17.4 <br>
kreas == 2.3.1 <br>
Pconsc4 == 0.4 <br>
pytorch == 1.4.0 <br>
PyG (torch-geometric) == 2.0.3 <br>
hhsuite (https://github.com/soedinglab/hh-suite)<br>
rdkit == 2019.03.4.0 <br>
ccmpred (https://github.com/soedinglab/CCMpred) <br>

# Data preparation
1. Prepare the data need for train. Get all msa files of the proteins in datasets and using Pconsc4 to predict all the contact map. A script in the repo can be run to do all the steps: <br>
**python scripts.py** <br><br>


# Train (cross validation)
5 folds cross validation. <br>
**python training_5folds.py 0 0 0** <br>
where the parameters are dataset selection, gpu selection, fold (0,1,2,3,4).

# Test
This is to do the prediction with the models we trained. And this step is to reproduce the experiments. <br>
**python test.py 0 0** <br>
and the parameters are dataset selection, gpu selection.


