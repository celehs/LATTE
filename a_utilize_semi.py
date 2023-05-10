import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score


##########using lo
def sigmod(x):
    return  1 / (1 + np.exp(-x))

def get_data_from_csv(mdir, dirr_data, filename, dirr_save,target_patient,
                                     colums_min, colums_max, visit_maximum,
                              dic_items, month_window=1,
                        train_mode="test", key_code="PheCode:250.2",
                         embedding_file="embedding_codified_T2D_codes_PCA80.csv",
                         flag_train_augment=0,data_flag=1,alpha_silver=0.2,temp_silver=0.8):
    print("---get data from csv:  ", dirr_data, ": ", filename)
    number_maximum =30*month_window
    dirr = dirr_data
    dirr_save = dirr_save
    df = pd.read_csv(dirr + filename)
    data_array_all = np.array(df[df.columns[colums_min:colums_max]])
    print("data_array_all.shape: ", np.array(data_array_all).shape)
    print("df.columns: ", df.columns)
    data_Time = list(df[str(dic_items["Time"])])
    Patient_num = list(df[str(dic_items["Patient"])])
    print (" Patient_num len: ", len(Patient_num))
    Y_label = list(df[str(dic_items["Label"])])
    key_codes=str(key_code).split(",")
    key_code_counts=[]
    for keycodei in key_codes:
        key_code_counts.append( list(df[str(keycodei)]))
    key_code_counts=np.array(key_code_counts)
    key_code_counts=np.sum(key_code_counts,axis=0)
    all_code_counts = np.sum(data_array_all, axis=1)
    dic_patient_utilization={}
    dic_patient_key_codes={}
    dic_label={}

    min_date = min(np.array(data_Time, dtype=np.int))
    max_date = max(np.array(data_Time, dtype=np.int))
    total_month = int(int(max_date - min_date) / month_window) + 1
    total_codes = colums_max - colums_min
    df_embedding = pd.read_csv( mdir + embedding_file)
    data_embedding_all = []
    for colmi in df.columns[colums_min:colums_max]:
        data_embedding_all.append(df_embedding[colmi])
    data_embedding_all = np.array(data_embedding_all, dtype=np.float)
    print("---data_embedding_all: ", data_embedding_all.shape)
    key_embedding = df_embedding[key_codes[0]]
    print("-----------key embedding: ", np.array(key_embedding).shape)

    # print("Patient_num len: ", len(Patient_num))
    # print("key_code_counts len: ", len(key_code_counts))
    # print("all_code_counts len: ", len(all_code_counts))
    # print("Y_label len: ", len(Y_label))
    for ID, count_key, count_all,label in zip(Patient_num,key_code_counts,all_code_counts,Y_label):
        dic_label[str(ID)]=label
        if not str(ID) in dic_patient_utilization:
            dic_patient_utilization[str(ID)]=count_all
            dic_patient_key_codes[str(ID)] = count_key
        else:
            dic_patient_utilization[str(ID)] += count_all
            dic_patient_key_codes[str(ID)] += count_key
    dic_patient_silver={}
    label_all=[]
    sivler_all=[]
    print ("dic_label len: ",len(dic_label))
    labels_total=list(dic_label.values())
    print ("labels_total mean: ",np.mean(labels_total))
    print ("labels_total[0:20]: ",labels_total[0:20])
    for ID in dic_patient_utilization:
        label_all.append(dic_label[ID])
        count=dic_patient_key_codes[ID]
        utilization = dic_patient_utilization[ID]
        silver_value=sigmod((np.log10(count + 1) - alpha_silver * np.log10(1 + utilization))/temp_silver)
        dic_patient_silver[ID]=silver_value
        sivler_all.append(silver_value)
    if train_mode == "train":
        AUC_silver = roc_auc_score(y_true=label_all,y_score=sivler_all)
        print("---------------------------------------------AUC_silver: ", AUC_silver)
    print ("dic_patient_silver len: ",len(dic_patient_silver))
    print("label_all len: ", len(label_all))

    print("key_code_counts.shape: ", np.array(key_code_counts).shape)



    patients_total = {}
    for rowi in range(len(data_Time)):
        if True:
            patient = str(Patient_num[rowi])
            #print ("patient: ",patient)
            if Patient_num[rowi] in target_patient or "ALL" in target_patient:
                if not patient in patients_total:
                    patient_col = np.ones(shape=(total_month, 1)) * int(Patient_num[rowi])
                    label_col = np.zeros(shape=(total_month, 1))
                    time_col = np.array(list(range(total_month)))
                    time_col = np.expand_dims(time_col, axis=-1)
                    patient_time_col = np.concatenate((patient_col, time_col), axis=-1)
                    patient_time_col = np.concatenate((patient_time_col, label_col), axis=-1)
                    patients_total[patient] = np.concatenate(
                        (patient_time_col, np.zeros(shape=(total_month, total_codes + 2))),
                        -1)  ###########total_codes+2: phecodes
                if patient in patients_total:
                    index_row = int(int(int(data_Time[rowi]) - min_date) / month_window)   #这里还是使用了month_window
                    patients_total[patient][index_row, 5:5 + colums_max - colums_min] += data_array_all[rowi]
                    patients_total[patient][index_row, 4] = dic_patient_silver[patient]
                    patients_total[patient][index_row, 3] = dic_patient_silver[patient]

                    patients_total[patient][index_row, 5:5 + colums_max - colums_min] = \
                        np.minimum(patients_total[patient][index_row, 5:5 + colums_max - colums_min],
                                   number_maximum)
                    patients_total[patient][index_row, 2] = int(Y_label[rowi])
                    patients_total[patient][index_row, 1] = int(data_Time[rowi])
                    patients_total[patient][index_row, 0] = int(Patient_num[rowi])
    #print(patients_total[str(Patient_num[0])])

    if train_mode == "train" and flag_train_augment==1:
        if month_window > 1:
            window_offset = range(-1, 2)
        else:
            window_offset = range(0)
        for addi in window_offset:    # 由于range是半闭半开区间，只取到2，3，4三个值
            month_window_temp = month_window + addi
            #print(month_window_temp)
            number_maximum = 3 * month_window_temp
            total_month = int(int(max_date - min_date) / month_window_temp) + 1
            #print(total_month)
            for rowi in range(len(data_Time)):
                patient_raw=str(Patient_num[rowi])
                patient = str(Patient_num[rowi]) + "_" + str(addi)
                if Patient_num[rowi] in target_patient or "ALL" in target_patient:
                    if not patient in patients_total:
                        patient_col = np.ones(shape=(total_month, 1)) * int(Patient_num[rowi])
                        label_col = np.zeros(shape=(total_month, 1))
                        time_col = np.array(list(range(total_month)))
                        time_col = np.expand_dims(time_col, axis=-1)
                        patient_time_col = np.concatenate((patient_col, time_col), axis=-1)
                        patient_time_col = np.concatenate((patient_time_col, label_col), axis=-1)
                        patients_total[patient] = np.concatenate(
                            (patient_time_col, np.zeros(shape=(total_month, total_codes + 2))),
                            -1)  ###########total_codes+2: phecodes



                    if patient in patients_total:
                        index_row = int(int(int(data_Time[rowi]) - min_date) / month_window_temp)
                        patients_total[patient][index_row, 5:5 + colums_max - colums_min] += data_array_all[rowi]
                        patients_total[patient][index_row, 4] = dic_patient_silver[patient_raw]
                        patients_total[patient][index_row, 3] = dic_patient_silver[patient_raw]
                        dic_patient_silver[patient]=dic_patient_silver[patient_raw]
                        patients_total[patient][index_row, 5:5 + colums_max - colums_min] = \
                            np.minimum(patients_total[patient][index_row, 5:5 + colums_max - colums_min],
                                       number_maximum)
                        patients_total[patient][index_row, 2] = int(Y_label[rowi])
                        patients_total[patient][index_row, 1] = int(data_Time[rowi])
                        patients_total[patient][index_row, 0] = int(Patient_num[rowi])

                    '''print(month_window_temp)
                    print(patient)
                    print(patients_total[patient])'''

    numbers_total = patients_total.keys()
    data_total = []
    data_total_counts = []
    patient_num_total = []
    label_total = []
    date_total = []
    weight_total = []
    key_feature1_total = []
    key_feature2_total = []
    squence_maximu = 0
    for patient_i in numbers_total:
        data_i = patients_total[patient_i]
        rows, cols = data_i.shape  # patient_num, date, Y, not included
        data_temp = []
        label_total_temp = []
        patient_num_total_temp = []
        date_total_temp = []
        weight_temp = []
        key_feature1_total_temp = []
        key_feature2_total_temp = []
        feature_d = colums_max - colums_min
        neighbout_stress = 1
        weight_emphsize = 1
        visit_final = 0

        for visit_i in range(rows):

            if np.sum(data_i[visit_i, 5:]) > 0 and len(data_temp) < visit_maximum:  #仅仅记录有事件出现的点？为什么
                # embeddings = np.matmul(data_i[visit_i, 3:], data_embedding_all.transpose())
                # data_temp.append(embeddings)
                visit_final = visit_i
                # 注意：这一句的位置一定不能写错！否则数据补齐错误。visit_final是最后一个
                data_temp.append(np.log10(1+data_i[visit_i, 5:]))    ####using log counts
                key_feature2_total_temp.append(data_i[visit_i, 4])
                key_feature1_total_temp.append(data_i[visit_i, 3])
                label_total_temp.append(data_i[visit_i, 2])
                date_total_temp.append(data_i[visit_i, 1])
                patient_num_total_temp.append(data_i[visit_i, 0])
                if (visit_i - neighbout_stress > 0) and (visit_i + neighbout_stress < len(data_i)):
                    if data_i[visit_i - neighbout_stress, 0] == data_i[visit_i + neighbout_stress, 0] and \
                            not data_i[visit_i - neighbout_stress, 2] == data_i[visit_i + neighbout_stress, 2]:
                        weight_temp.append(weight_emphsize)
                    else:
                        weight_temp.append(1)
                else:
                    weight_temp.append(1)


        if len(data_temp) > squence_maximu:
            squence_maximu = len(data_temp)

        for add_i in range(visit_maximum - len(data_temp)):   #超过的部分用最后一个补足;这一处实际上没有发挥作用
            # data_temp.append(np.zeros(shape=(embeddings_d)))
            '''print(data_i[visit_final, 0])
            print(data_i[visit_final, 1])'''
            data_temp.append(np.zeros(shape=(total_codes)))
            key_feature2_total_temp.append(dic_patient_silver[patient_i])
            key_feature1_total_temp.append(dic_patient_silver[patient_i])
            label_total_temp.append(data_i[visit_final, 2])
            date_total_temp.append(data_i[visit_final, 1])
            patient_num_total_temp.append(data_i[visit_final, 0])
            weight_temp.append(0)
        data_total.append(np.array(data_temp).reshape(visit_maximum, feature_d))
        key_feature2_total.append(key_feature2_total_temp)
        key_feature1_total.append(key_feature1_total_temp)

        label_total.append(label_total_temp)
        date_total.append(date_total_temp)
        patient_num_total.append(patient_num_total_temp)
        weight_total.append(weight_temp)

        '''if patient_i == str(Patient_num[0]):
            print(patient_num_total_temp)
            print(date_total_temp)'''

    label_total = np.array(label_total)
    label_total = np.expand_dims(label_total, axis=-1)
    weight_total = np.array(weight_total)
    patient_num_total = np.array(patient_num_total)
    date_total = np.array(date_total)
    data_total = np.array(data_total)
    silver_total=np.array(key_feature1_total)
    # print(date_total[0])

    print("-----------------------------squence_maximu: ", squence_maximu)
    print("--------------------------------month_window: ", month_window)
    print("----------------------------------------------run_mode: ", train_mode)

    print("----------------------------------------------total patients valid: ", len(patients_total))
    print("------data_total: ", np.array(data_total).shape)
    print("np.array(label_total).shape: ", label_total.shape)
    print("np.array(weight_total).shape: ", weight_total.shape)
    print("np.array(patient_num_total).shape: ", patient_num_total.shape)
    print("np.array(date_total).shape: ", date_total.shape)
    print("np.array(silver_total).shape: ", silver_total.shape)


    save_name = filename+str(data_flag) + "_"+train_mode+".pkl"
    with open(dirr_save + save_name, 'wb') as fid:
        pickle.dump((np.array(data_total), np.array(label_total),
                     np.array(patient_num_total), np.array(date_total),
                     np.array(weight_total), np.array(data_embedding_all),
                     np.tile(key_embedding, (colums_max-colums_min, 1)),silver_total), fid, protocol=4)


