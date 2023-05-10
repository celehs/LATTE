##################Dependencies##################
tensorflow==2.3.0
sklearn
pandas
argparse

##################input data: refer to Input_example.csv
ID: patient_num
Y: the label 0 or 1
T: integers that flags the dates
other columns for the covariants 

##################input embedding: refer to embedding_example.csv; the input codes should have the corresponding embeddings in embedding_example.csv

#############model runining: see example_final.sh###########
python a_train_final.py --train_directory "/n/data1/hsph/biostat/celehs/lab/junwen/T2D/R_deep/data_HF/" \
                        --train_filename "HF_Prod4_100_kumar_codes_CUIs_labeled_195features.csv" \
                        --test_directory "/n/data1/hsph/biostat/celehs/lab/junwen/T2D/R_deep/data_HF/"  \
                        --test_filename  "HF_biobank_100_codes_CUIs_195features.csv"                     \
                        --unlabel_filename  "HF_Prod4_100_kumar_codes_CUIs_labeled_195features.csv" \
                        --save_directory  "/n/data1/hsph/biostat/celehs/lab/junwen/T2D/R_deep/data_HF/results_final_test/" \
                        --results_filename  "HF_prod4_2_Biobank100.csv"   \
                        --embedding_filename  "embedding_Dou_VA_HF_Direct_200d_Keser_codes_CUIs_normalize.csv"   \
                        --key_code  "PheCode:428.1,PheCode:428.2,PheCode:428.3,PheCode:428.4,C0018802"  \
                        --colums_min  3 \
                        --colums_max  198 \
                        --embedding_dim 200 \
                        --number_labels 50  \
                        --flag_cross_dataset 1 \
                        --epochs  50  \
                        --epoch_silver  8 \
                        --layers_incident  '80,80,' \
                        --weight_prevalence 0.2 \
                        --weight_unlabel 0.2 \
                        --weight_constrastive  0.1 \
                        --weight_smooth  0.1 \
                        --weight_additional  0.1 \
                        --flag_save_attention 1 \
                        --flag_load_model 0  \
                        --flag_train_augment 0 \

#################
key_code: the codes, seperated in comma, should have corresponding embedding in the embedding file; if only one phecode, say T2D, then it's "PheCode:250.2"
colums_min to colums_max: for the covariants
embedding_dim: dimension of embedding
number_labels: if cross-validation using the train_filename: the model randomly sample number_labels patients for training and the others for test; in this case, the train_filename and test_filename are the same;
flag_cross_dataset: train_filename and test_filename are not the same; using the whole  train_filename and the whole test_filename for test
epochs: total epoch to train the model
epoch_silver: epoches for pre-training using the silver labels based on the key_code;  the silver label is computed the same as Phenorm
layers_incident: how mayn LSTM layers and units for temporal modelling; "80,": one layer of 80 units; "80, 90", two layers with 80, 90 units, resp.
weight_prevalence: the model do incident prediction and EVER/NEVER phenotyping simultaneously; it denotes the weight for EVER/NEVER phenotyping
weight_unlabel: the weights of unlabeled data to jointly train the model using silver-label
weight_constrastive, weight_smooth, weights_additional: can use the default values;
flag_save_attention: if save the learned visit representation, the codes weights for visualization
flag_load_model: the trained model is saved by default to save_directory with results_filename; the flag of if loadding the trained model if there exits
flag_train_augment: if augment the labeled data with varied time windows; for HF, T2D, the default window 3-month; if flag_train_augment=1, then will also use 2-month and 4-month window for training; but the evaluation is on 3-month;
####### to be noted:
1. for MS prediction, it's better to set weight_smooth,weights_constrastive to be smaller, say 0.05; 
2. for HF, T2D prediction: only LSTM layer, with layers_incident='80,' is ok; but for MS, two layers might need
########



