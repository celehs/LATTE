# -*- coding: utf-8 -*-
"""
TRAIN LAUNCHER
"""
from a_utilize_semi import get_data_from_csv
from a_semi_model_final import Attention_train
import os
import argparse
import pandas as pd
import random


def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--train_directory', type=str,
                        default="C:/Users/NORTH/source/incident_phenotyping/",
                        help='Directory of train data ')
    parser.add_argument('--train_filename', type=str, default="input_example.csv",
                        help='Filename of the train data')
    parser.add_argument('--test_directory', type=str,
                        default="C:/Users/NORTH/source/incident_phenotyping/",
                        help='Directory of test data ')
    parser.add_argument('--test_filename', type=str, default="input_example.csv",
                        help='Filename of the test data')
    parser.add_argument('--embedding_filename', type=str, default="embedding_items_HF_PCA80.csv",
                        help='Filename of the embedding_filename data')
    parser.add_argument('--key_code', type=str, default="PheCode:428",  #PheCode:250.2
                        help='the name of key_code')
    parser.add_argument('--unlabel_directory', type=str,
                        default="C:/Users/NORTH/source/incident_phenotyping/",
                        help='Directory of unlabeled data ')
    parser.add_argument('--unlabel_filename', type=str, default="T2D_unlabeled_codified_all_removed3.csv",
                        help='Filename of the unlabeled data')
    parser.add_argument('--save_directory', type=str,
                        default="/n/data1/hsph/biostat/celehs/lab/",
                        help='Directory to save the results')
    parser.add_argument('--results_filename', type=str, default="results_RETTAIN.csv",
                        help='Filename to save the result data')
    parser.add_argument('--colums_min', type=int, default=4,
                        help='data beginning column index  ')
    parser.add_argument('--colums_max', type=int, default=120,
                        help='data end column index')
    parser.add_argument('--epochs', type=int, default=60,
                        help='training epoches')
    parser.add_argument('--month_window', type=int, default=3,
                        help='month_window')
    parser.add_argument('--max_visits', type=int, default=115,
                        help='max visits length')
    parser.add_argument('--flag_train_augment', type=int, default=1,
                        help='flag_train_augment if to augment training data with different windows')
    parser.add_argument('--flag_cross_dataset', type=int, default=0,
                        help='flag_cross_dataset: do evaluations on the test dataset; otherwise, perform cross-valiation using training dataset')
    parser.add_argument('--number_labels', type=int, default=50,
                        help='number_labels',)
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='dimensions of code embeddings', )
    parser.add_argument('--epoch_silver', type=int, default=10,
                        help='epoches of pre-training using silver labels', )
    parser.add_argument('--layers_incident', type=str, default="80",
                        help='layers of GRUs for temporal modelling, "80" for only layer, "80,90" for two layer with 80, 90 units ', )
    parser.add_argument('--weight_prevalence', type=float, default=0.25,
                        help='weights of EVER/NEVER prevalence learning', )
    parser.add_argument('--weight_unlabel', type=float, default=0.3,
                        help='weights of silver unlabeled data during training', )
    parser.add_argument('--weight_constrastive', type=float, default=0.1,
                        help='weights of constrastive representation learning ', )
    parser.add_argument('--weight_smooth', type=float, default=0.1,
                        help='weight of learning temporally-smooth representation', )
    parser.add_argument('--weight_additional', type=float, default=0.1,
                        help='weights of additional regulization for semi-supervised learning', )
    parser.add_argument('--flag_save_attention', type=int, default=1,
                        help='if save the learned weights and representation for visulization; 1:save; 0: do not save' )
    parser.add_argument('--flag_load_model', type=int, default=0,
                        help='model is saved by defualt using the same name as results_filename; 1: reload the model; 0: do not reload' )
    parser.add_argument('--flag_prediction', type=int, default=0,
                        help='if predict new data, where the labels are random and evaluation with the Y is invalid')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('---readding configurations---- ')
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    train_directory = ARGS.train_directory
    train_filename = ARGS.train_filename
    test_directory = ARGS.test_directory
    test_filename = ARGS.test_filename
    number_labels = ARGS.number_labels
    flag_cross_dataset=ARGS.flag_cross_dataset
    embedding_filename = ARGS.embedding_filename
    key_code = ARGS.key_code
    month_window = ARGS.month_window
    flag_train_augment = ARGS.flag_train_augment
    unlabel_directory =  ARGS.test_directory
    unlabel_filename=ARGS.unlabel_filename

    print("---train_directory: ", train_directory)
    print("---train_filename: ", train_filename)
    print("---test_directory: ", test_directory)
    print("---test_filename: ", test_filename)
    print("---unlabel_filename: ", unlabel_filename)
    save_directory = ARGS.save_directory  # make sure this save dirr exist
    results_filename = ARGS.results_filename  # results_RETTAIN.csv

    colums_min = ARGS.colums_min
    colums_max = ARGS.colums_max
    max_visits = ARGS.max_visits
    embedding_dim=ARGS.embedding_dim
    epochs = ARGS.epochs  # how many epoches
    dic_colums = {"Time": "T", "Patient": "ID", "Label": "Y"}
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    patients_all=list(pd.read_csv(train_directory+train_filename)["ID"])
    patients_all=list({}.fromkeys(patients_all).keys())
    print ("---total patients: ",len(patients_all))
    print("---number_labels patients: ", number_labels)
    random.shuffle(patients_all)
    train_patients=patients_all[0:number_labels]
    test_patients=patients_all[number_labels:]
    if flag_cross_dataset==0:
        target_patient=[train_patients,test_patients,["ALL"]]
        print ("----------------------------------no no  no cross dataset ...............")
    else:
        target_patient = [["ALL"], ["ALL"], ["ALL"]]
        print("---------------------------------- cross dataset...............")
    data_flag=random.randint(1,10000)
    get_data_from_csv(train_directory, train_filename,
                              train_directory,target_patient=target_patient[0], colums_min=colums_min, colums_max=colums_max,
                              visit_maximum=max_visits,
                              dic_items=dic_colums,month_window=month_window, train_mode="train",
                        key_code=key_code, embedding_file=embedding_filename,
                                     flag_train_augment=flag_train_augment,data_flag=data_flag)
    get_data_from_csv(test_directory, test_filename,
                              test_directory,target_patient=target_patient[1], colums_min=colums_min,
                              colums_max=colums_max, visit_maximum=max_visits,
                              dic_items=dic_colums,month_window=month_window, train_mode="test",
                        key_code=key_code, embedding_file=embedding_filename,flag_train_augment=flag_train_augment,data_flag=data_flag)
    get_data_from_csv(unlabel_directory, unlabel_filename,
                              unlabel_directory,target_patient=target_patient[2], colums_min=colums_min,
                              colums_max=colums_max, visit_maximum=max_visits,
                              dic_items=dic_colums,month_window=month_window,train_mode="ALL",
                        key_code=key_code, embedding_file=embedding_filename,flag_train_augment=flag_train_augment,data_flag=data_flag)

    print ("----------------------ARGS.epoch_silver: ",ARGS.epoch_silver)
    print("----------------------ARGS.layers_incident: ", ARGS.layers_incident)
    print("----------------------ARGS.weight_prevalence: ", ARGS.weight_prevalence)
    print("----------------------ARGS.weight_constrastive: ", ARGS.weight_constrastive)
    print("----------------------ARGS.weight_smooth: ", ARGS.weight_smooth)
    print("----------------------ARGS.weight_additional: ", ARGS.weight_additional)
    print("----------------------ARGS.flag_save_attention: ", ARGS.flag_save_attention)
    print("----------------------ARGS.flag_load_model: ", ARGS.flag_load_model)



    Attention_train(dirr_train=train_directory,
                  filename_train=train_filename+str(data_flag) + "_train.pkl",
                  dirr_test=test_directory,
                  filename_test=test_filename+str(data_flag) + "_test.pkl",
                  dirr_unlabel=unlabel_directory,
                  filename_unlabel  =unlabel_filename +str(data_flag)+ "_ALL.pkl",
                  dirr_save=save_directory,
                  filename_save=results_filename,
                  epochs=epochs, colums_min=colums_min,
                  colums_max=colums_max,max_visits=max_visits,
                   embedding_dim=embedding_dim,
                    epoch_silver=ARGS.epoch_silver, layers_incident=ARGS.layers_incident,
                    weights_prevalence=ARGS.weight_prevalence, weights_unlabel=ARGS.weight_unlabel,
                    weights_constrastive=ARGS.weight_constrastive, weight_smooth=ARGS.weight_smooth,
                    weights_additional=ARGS.weight_additional,flag_save_attention=ARGS.flag_save_attention,
                    flag_load_model=ARGS.flag_load_model,flag_prediction=ARGS.flag_prediction)
