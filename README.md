# LATTE: Label-efficient Incident Phenotyping from Longitudinal Electronic Health Records
The code of paper "*LATTE: Label-efficient Incident Phenotyping from
Longitudinal Electronic Health Records*"


## Dependencies

* python 3.8
* R 4.2.1
* tensorflow 2.10



## Code

### Setup

Before running the codes, set up requirements packages of python and R.

```shell
>> pip install -r requirements.txt
# install python requirements packages for running model

>> Rscript pack_prepare.R
# install R requirements packages for simulation data generation & plotting result.
```



### Demo

The demo is running on simulation dataset(simulation data generation method),  

To run a demo on generated data, simply execute the following script:

```shell
>> python a_train_final_sim.py 
```

To generate your own simulation data, and then run scripts on generated data:

```shell
# Before running the R program, adjust the home path in set_mdir.R to yours.

>> Rscript gen_Data_SIM.R {1} {2} {3} {4}     
# ARGS: 1.total number 2.labeled number 3.train(labeled) num 
# Example: Rscript gen_Data_SIM.R 2000 1000 200

>> python a_train_final_sim.py 
```

 

### Customized running

 To run the code on your own dataset:

1. **Format your data as described in [Dataset](#dataset):**

2. **Customize your running**

   * Terminal Script

     ```shell
     python a_train_final.py 
     ```

   * arguments

     | Arguments                    | Description                                                  |
     | ---------------------------- | ------------------------------------------------------------ |
     | home_path                    | Home path of the project file. Default value: './'           |
     | train/test/unlabel_directory | *(3 arguments)* path of directory of train/test/unlabel data |
     | train/test/unlabel_filename  | *(3 arguments)* filenames of train/test/unlabel data         |
     | embedding_filename           | The whole path of embedding file                             |
     | key_code                     | the codes corresponding to the entity names in input data(for example: 'PheCode:250.2' for H2D data). If there're multiple codes, separate them in comma, all of which should have corresponding embedding in the [embedding file]. |
     | colums_min,  colums_max      | the numerical index of all the covariates                    |
     | embedding_dim                | # of dimension of embedding                                  |
     | max_visits                   | maximum number of visits in prediction                       |
     | number_labels                | (function only when cross validation is activated: flag_cross-validation = 1, train dataset = test dataset)the model randomly sample number_labels patients from trainset for training and the others for test |
     | epochs                       | total epoch to train the model                               |
     | epoch_silver                 | epoches for pre-training using the silver labels             |
     | layers_incident              | how mayn LSTM layers and units for temporal modelling        |
     | weight_prevalence            | the model calculate incident prediction and EVER/NEVER phenotyping loss simultaneously; this argument denotes the weight of EVER/NEVER phenotyping in the loss |
     | weight_unlabel               | the weights of unlabeled loss ($\mathcal{L}_{silver}$ in paper) |
     | weight_constrastive          | the weights of contrastive loss($\mathcal{L}_{ct}$ in paper) |
     | **flag_traintest_sep**       | flag indicating whether train dataset and test dataset are separated: if 1 --> train set and test set are different, 0 --> train set and test set are the same set, and the program will divide it into two parts |
     
     

3. **Result**

   * Results are saved in `/Results/`, you can see example results at `/Results_Example/`

   * Document Tree of Results directory:

     ```shell
     C:.
     |   Incident_epoch39results.csv                         # incident phenotyping result at 39 epoch
     |   Incident_epoch40results.csv                         # incident phenotyping result at 40 epoch
     |   Prevalence_results.csv                              # prevalence(ever/never) result at 39 epoch
     |   results_incident_evaluation.txt          # summary of incidence phenotyping evaluation metric
     |   results_prevalence_evaluation.txt        # summary of prevalence prediction evaluation metric
     |   results1__embedding_patient_hiddenFCN_label_train.pkl
     |   resultsAttenation_value_patient_visit_code_prediction_label_weight_test.pkl
     |   results_embedding_patient_hiddenFCN_label_test.pkl
     |   _results_code_weights.csv
     |
     \---results_model                                     # saved models
     
     
     
     ```

     * Incident_epoch{x}results.csv :  **incident phenotyping results on every time points for every patient**, saved  after x training epoch.
     * Prevalence__results.csv: **Prevalence(EVER/NEVER) binary phenotyping results for every patient**
     * .txt: an overall summary of the incident phenotyping and prevalence result.

   

## Dataset

### Input Data -- Longitudinal
---
##### Columns:  

* ID: patient_num
* Y: the label 0 or 1
* T: integers that flags the dates
* other columns: covariates(column names are the index of covariates: names of medical concepts in EHRs in our paper)

##### Rows:

The data of one specific patient at one specific time point. A vector, each dimension is the occurring counts of corresponding medical entity.



### Embeddings
---
* Each columns is the concept embeddings of $i_{th}$  medical entity ($e_i$ in **Concept Reweight module** in the paper). The column names are the index of covariates: names of medical concepts in EHRs in our paper)

* **All the index names in input data $\in$ All the index names in embedding files**



### Examples
---
Please see examples of input data, simulation data and embedding file at:

```shell
| ├─example                                  # example for embeddings and input data
│      embedding_example.csv 
│      Input_example.csv
│      input_example_sim.csv
```





## Document Tree

```shell
│  a_semi_model_final.py                         # LATTE model
│  a_train_final.py                              # training on real data
│  a_train_final_sim.py                          # training on simulation data
│  a_Transformer.py
│  a_utilize_semi.py                         
│  gen_Data_SIM.R                                # SimDat generating script
│  get_Folds.R                                   # Split simulation data into train/test/unlabel sets
│  pack_prepare.R                                # package preparation for R scripts
│  simulation_data_functions_v4.R                # simulation data generating funtions
│  Simdat_generation.pdf                         # Principle of simulation data generation          
│
├─simdata_generate_rely                         # some reliment scripts to generate simulation data
├─Results                                          
├─Results_example
│
├─Simulation
│  └─SimDat                                     # Simdat generated by me
│      ├─SimDat.1                               # Simdat.1 is generated using
│      │  │  SimDat.1.csv                       # All of SimDat.1 
│      │  │  SimDat.1.Rds
│      │  │  SimDat.1_labeled.csv               # labeled part of SimDat.1 
│      │  │  SimDat.1_label_patient_info.csv
│      │  │  SimDat.1_unlabeled.csv             # unlabeled part of SimDat.1 
│      │  │  SimDat.1_unlabeled.csv6116_ALL.pkl
│      │  │  SimDat.1_unlabeled.csv9061_ALL.pkl
│      │  │  test_patients.csv                  # test patients' numbers
│      │  │  train_patients_450.csv             # train patients' numbers
│      │  │  unlabeled_patients.csv             # unlabeled patients' numbers
│      │  │
│      │  ├─test
│      │  │      test_data.csv
│      │  │
│      │  └─train                               # Simdat.2 is generated using
│      │          train_data.csv
│      │
│      └─SimDat.2

```



