

# ------------------------------------------
cd C:\Users\NORTH\source\incident_phenotyping


python generate_Sigma.py


Rscript gen_Data_SIM.R 2000 500 100 200



python a_train_final_sim.py --home_path "C:/Users/NORTH/source/incident_phenotyping/" \
                        --train_directory "Simulation/SimDat/SimDat.1/train/" \
                        --train_filename "train_data.csv" \
                        --test_directory "Simulation/SimDat/SimDat.1/test/"  \
                        --test_filename  "test_data.csv"                     \
                        --unlabel_filename  "SimDat.1_unlabeled.csv" \
                        --save_directory  "Results/" \
                        --results_filename  "results_RETTAIN"  \
                        --embedding_filename  "data/embedding_selected.csv"  \
                        --key_code  "S.1,S.3,S.5"  \
                        --colums_min  5 \
                        --colums_max  15 \
                        --embedding_dim 10 \
                        --month_window 1 \
                        --number_labels 50  \
                        --flag_cross_dataset 1 \
                        --epochs  35  \
                        --epoch_silver  5 \
                        --layers_incident  '40' \
                        --weight_prevalence 0.2 \
                        --weight_unlabel 0.2 \
                        --weight_constrastive  0.1 \
                        --weight_smooth  0.1 \
                        --weight_additional  0.1 \
                        --flag_save_attention 1 \
                        --flag_load_model 0  \
                        --flag_train_augment 1 \



