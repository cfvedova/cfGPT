# cfGPT
cfGPT repo for running and training model

## Overview
cfGPT is a GPT-based Artificial Intelligence used to decrypt cell-free RNA. It builds on previously developed research in the form of single-cell GPT, whose github can be found [here](https://github.com/bowang-lab/scGPT) and research paper can be found [here](https://www.nature.com/articles/s41592-024-02201-0). The different training procedures and files for cfGPT are described below.

There are three main training procedures contained in this github.

1. Train classifier model that uses the outputs of single-cell GPT (scGPT) as input.
   1. Transfer learns previously trained single cell scGPT model
   2. train_classifier.py
2. Retraining the encoder performs the masked language modeling task
   1. Retrains scGPT encoder model on cell-free simulated data
   2. Can then use this retrained model in training procedure 1 to create a new cfGPT model
   3. retrain_encoder.py
3. Train encoder model from scratch
   1. Uses the same code as the procedure 2. Simply remove the option of loading a model and it will train a new one from scratch
   2. Again, can use this encoder model base for procedure 1 to create a new cfGPT model
   3. retrain_encoder.py

### Folders and Files
#### Dataset
Upload your input datasets to be trained using whichever procedure you like.

#### conda_specs
Folder containing environment.yml files used to structure your conda environment.

#### save
Folder used to save previously trained scGPT models and newly trained cfGPT models.

#### scGPT
Folder containing files from the scGPT [github](https://github.com/bowang-lab/scGPT) that are used to structure the GPT-based transformer model.

#### Options to Test.txt
Text file containing some ideas for further models to be tested/trained.

#### create_simulated_data_preprocessed.py
Python file transforms single cell RNA datasets, such as Tabula Sapiens, into a simulated bulk RNA sequencing dataset.

#### retrain_encoder.py
Used to perform training procedures 2 and 3 listed above.

#### train_classifier.py
Used to perform training procedure 1 listed above.

#### test_model.ipynb and test_model.py
Both files fulfill the same purpose but one is a jupyter notebook and the other is a python file. These files are used to test our trained cfGPT models.