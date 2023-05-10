# example of a cnn for image classification
import numpy as np
import pandas as pd
import random
import os
import pickle
from tensorflow.keras import layers,models
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score,accuracy_score,precision_score
from a_Transformer import MultiHeadAttention,MultiHeadAttention_sigmod

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def Attention_train(dirr_train,filename_train,dirr_test, filename_test, dirr_unlabel,filename_unlabel,
                    dirr_save, filename_save,
                  colums_min,colums_max,max_visits,
                   epochs=50,batch_size=128,epoch_show=1,feature_d=161,
                    embedding_dim=100,
                    epoch_silver=10,layers_incident="80",
                    weights_prevalence = 0.25,weights_unlabel = 0.3,
                    weights_constrastive = 0.1,weight_smooth = 0.1,weights_additional=0.1,
                    flag_save_attention = 1,flag_load_model=0, flag_prediction=0):
    if not os.path.exists(dirr_save):
        os.mkdir(dirr_save)

    if flag_prediction >0:
        print("----------------------------------predict novel data: instead of cross-validation or evaluation-----------")
    else:
        print(
            "----------------------------------cross-validation with training data or on test data -----------")
    feature_d=colums_max-colums_min
    train_epoches=epochs
    batch_size=batch_size
    max_lengh=max_visits
    optimizer = tf.keras.optimizers.Adam()
    optimizer_silver = tf.keras.optimizers.Adam()
    d_model_T=min(100,embedding_dim)
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=30):
            super(CustomSchedule, self).__init__()
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.warmup_steps = warmup_steps
        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        
    with open(dirr_train + filename_train, 'rb') as fid:
        (data_train, y_train, patient_num_train,
         date_train, weights_train,data_embedding_train,key_embedding_train,silver_train) = pickle.load(fid)
    with open(dirr_test + filename_test, 'rb') as fid:
        (data_test, y_test, patient_num_test, date_test,
         weights_test,data_embedding_test,key_embedding_test,silver_test) = pickle.load(fid)
    with open(dirr_unlabel + filename_unlabel, 'rb') as fid:
        (data_train_unlabel, y_train_unlabel, patient_num_train_unlabel,
         date_train_unlabel, weights_train_unlabel,data_embedding_unsuper,key_embedding_unlabel,silver_unsuper) = pickle.load(fid)

    os.remove(dirr_train + filename_train)
    os.remove(dirr_test + filename_test)
    os.remove(dirr_unlabel + filename_unlabel)

    print("--------------------------date_train.shape: ", np.array(date_train).shape)
    print("--------------------------patient_num_train.shape: ", np.array(patient_num_train).shape)
 
    data_embedding_all_train=[]
    data_embedding_all_test=[]
    data_embedding_all_unsuper=[]
    data_embedding_all_test_unsuper=[]
    data_unsuper=[]
    weights_unsuper = []
    data_train_new=[]
    y_train_new=[]
    patient_num_train_new=[]
    date_train_new=[]
    weights_train_new=[]
    train_num=batch_size*5
    key_features_train=[]
    key_features_unsuper=[]
    silver_train_new=[]
    silver_unsuper_new = []
    for i in range(train_num):
        data_embedding_all_train.append(data_embedding_train)
        data_embedding_all_unsuper.append(data_embedding_unsuper)
        sample_i=i%(len(data_train))
        key_features_train.append(key_embedding_train)
        data_train_new.append(data_train[sample_i])
        y_train_new.append(y_train[sample_i])
        patient_num_train_new.append(patient_num_train[sample_i])
        date_train_new.append(date_train[sample_i])
        weights_train_new.append(weights_train[sample_i])
        silver_train_new.append(silver_train[sample_i])
        sample_i_unsuper=random.randint(0, len(data_train_unlabel) - 1)
        data_unsuper.append(data_train_unlabel[sample_i_unsuper])
        weights_unsuper.append(weights_train_unlabel[sample_i_unsuper])
        key_features_unsuper.append(key_embedding_unlabel)
        silver_unsuper_new.append(silver_unsuper[sample_i_unsuper])

    data_train=np.array(data_train_new)
    y_train = np.array(y_train_new)
    patient_num_train = np.array(patient_num_train_new)
    date_train = np.array(date_train_new)
    key_features_train = np.array(key_features_train)
    weights_train = np.array(weights_train_new)
    data_unsuper = np.array(data_unsuper)
    weights_unsuper = np.array(weights_unsuper)
    key_features_unsuper = np.array(key_features_unsuper)
    silver_train = np.array(silver_train_new)
    silver_unsuper = np.array(silver_unsuper_new)

    key_features_test=[]
    for i in range(len(data_test)):
        data_embedding_all_test.append(data_embedding_train)
        data_embedding_all_test_unsuper.append(data_embedding_unsuper)
        key_features_test.append(key_embedding_test)
    data_embedding_all_train=np.array(data_embedding_all_train)
    data_embedding_all_test = np.array(data_embedding_all_test)
    data_embedding_all_unsuper=np.array(data_embedding_all_unsuper)
    data_embedding_all_test_unsuper=np.array(data_embedding_all_test_unsuper)
    key_features_test=np.array(key_features_test)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_silver = tf.keras.metrics.Mean(name='train_loss_silver')
    train_loss_classfication = tf.keras.metrics.Mean(name='train_smooth_loss_classfication')
    train_smooth_loss = tf.keras.metrics.Mean(name='train_smooth_loss')
    train_smooth_loss_unsuper = tf.keras.metrics.Mean(name='train_smooth_loss_unsuper')
    train_keyfeature_loss = tf.keras.metrics.Mean(name='train_keyfeature_loss')
    train_constrastive_loss = tf.keras.metrics.Mean(name='train_constrastive_loss')
    train_constrastive_loss_MLP = tf.keras.metrics.Mean(name='train_constrastive_loss_MLP')
    train_MLP_entropy_unsuper = tf.keras.metrics.Mean(name='train_MLP_entropy_unsuper')
    train_MLP_consistency = tf.keras.metrics.Mean(name='train_MLP_consistency')
    train_MLP_incident= tf.keras.metrics.Mean(name='train_MLP_incident')
    train_metric = tf.keras.metrics.AUC(name='train_auc', )
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_smooth_loss = tf.keras.metrics.Mean(name='valid_smooth_loss')
    valid_metric = tf.keras.metrics.AUC(name='test_auc')
    ds_train = tf.data.Dataset.from_tensor_slices((data_train, y_train,
                                                   patient_num_train, date_train, weights_train,
                                                   data_embedding_all_train, data_embedding_all_unsuper,
                                                   data_unsuper, weights_unsuper,
                                                   key_features_train,key_features_unsuper,silver_train,silver_unsuper)) \
        .shuffle(buffer_size=500).batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()

    ds_test = tf.data.Dataset.from_tensor_slices((data_test, y_test,
                                                  patient_num_test, date_test, weights_test,
                                                  data_embedding_all_test, data_embedding_all_test_unsuper,
                                                  data_test, weights_test,
                                                  key_features_test,key_features_test)) \
        .shuffle(buffer_size=500).batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()

##############learning code importance based on embedding###############
    def code_attention(input_seq,embedding):         
        dim=int(embedding_dim*0.75)
        layer1 = layers.Dense(dim, activation=tf.nn.relu,name="code_attention/fc1")
        layer11 = layers.Dense(dim, activation=tf.nn.relu, name="code_attention/fc11")
        layer3 = layers.Dense(1, activation=None,name="code_attention/prediction")
        input_seq = layer1(input_seq)
        embedding = layer11(embedding)
        data_embedding=tf.concat([input_seq,embedding],axis=-1)
        data_embedding=layer3(data_embedding)
        data_embedding=tf.squeeze(data_embedding,axis=-1)
        data_embedding=tf.nn.softmax(data_embedding,axis=-1)
        data_embedding=tf.expand_dims(data_embedding,axis=-2)
        return data_embedding
##############learning code importance based on embedding###############
    def Model_prediction():
        embedding_d=embedding_dim   ###input dim
        inputs1 = layers.Input(shape=(max_lengh,feature_d,))
        inputs1_shuffle = layers.Input(shape=(max_lengh, feature_d,))
        inputs_unsuper = layers.Input(shape=(max_lengh, feature_d,))
        inputs_unsuper_masked=layers.Input(shape=(max_lengh, feature_d,))

        inputs_data_embedding_all = layers.Input(shape=(feature_d, embedding_d,))
        inputs_data_embedding_unsuper = layers.Input(shape=(feature_d, embedding_d,))
        inputs_keyfeature = layers.Input(shape=(feature_d, embedding_d,))
        inputs_keyfeature_unsuper = layers.Input(shape=(feature_d, embedding_d,))
        inputs1_temp = inputs1*1.0
        inputs_unsuper_temp = inputs_unsuper*1.0
        inputs1_shuffle_temp = inputs1_shuffle * 1.0
        ##############the input counts is weighted
        attention_value = 10.0*code_attention(inputs_data_embedding_all, inputs_keyfeature)
        bias_value = 0.1
        inputs1_reweight=(attention_value+bias_value)*inputs1_temp
        inputs1_reweight_shuffle = (attention_value + bias_value) * inputs1_shuffle_temp
        inputs_unsuper_reweight=(attention_value+bias_value)*inputs_unsuper_temp
        inputs_unsuper_reweight_mask=(attention_value+bias_value)*inputs_unsuper_masked
        ##############the input counts is weighted

        inputs1_embedding_code_rw = tf.matmul(inputs1_reweight, inputs_data_embedding_all)

        ###################the re-weighted code utilization
        code_num = tf.reduce_sum(inputs1_reweight, axis=-1,keepdims=True)+1
        code_num_unsuper = tf.reduce_sum(inputs_unsuper_reweight, axis=-1,keepdims=True)+1
        code_num_unsuper_mask = tf.reduce_sum(inputs_unsuper_reweight_mask, axis=-1, keepdims=True) + 1
        code_num_shuffle = tf.reduce_sum(inputs1_reweight_shuffle, axis=-1,keepdims=True)+1
        ###################the re-weighted code utilization

        ###########getting the normalized counts*embedding with the counts is re-weighted
        inputs1_embedding_ori = tf.matmul(inputs1_temp, inputs_data_embedding_all)
        inputs1_embedding = tf.matmul(inputs1_reweight/code_num, inputs_data_embedding_all)
        inputs1_embedding_shuffle = tf.matmul(inputs1_reweight_shuffle/code_num_shuffle, inputs_data_embedding_all)
        inputs_embedding_unsuper = tf.matmul(inputs_unsuper_reweight/code_num_unsuper, inputs_data_embedding_unsuper)
        inputs_embedding_unsuper_mask = tf.matmul(inputs_unsuper_reweight_mask / code_num_unsuper_mask, inputs_data_embedding_unsuper)
        ###########getting the normalized counts*embedding with the counts is re-weighted


        ########using self-attention to learn visit imporance
        Attention_prevalence=MultiHeadAttention(d_model_T,num_heads=1)
        Prevalence_layer1 = layers.Dense(int(d_model_T), activation=tf.nn.relu,name="Binary/fcn1")
        Prevalence_predictor = layers.Dense(1, activation=None,name="Binary/predictor")
        Prevalence_predictor_silver = layers.Dense(1, activation=None, name="Binary/predictor_silver")
        ########using self-attention to learn visit imporance

        ####### getting Prevalence_prediction
        Prevalence_fcn,visit_weights=Attention_prevalence(inputs1_embedding,inputs1_embedding,inputs1_embedding, mask=None)
        Prevalence_fcn=Prevalence_layer1(tf.reduce_mean(Prevalence_fcn[:,:,:],axis=1))
        Prevalence_prediction = tf.nn.dropout(Prevalence_fcn, rate=0.2)
        Prevalence_prediction = Prevalence_predictor(Prevalence_prediction)

        Prevalence_fcn_unsuper,visit_weights_unsuper = Attention_prevalence(inputs_embedding_unsuper,inputs_embedding_unsuper,inputs_embedding_unsuper,None)
        Prevalence_fcn_unsuper = Prevalence_layer1(tf.reduce_mean(Prevalence_fcn_unsuper[:,:,:],axis=1))
        Prevalence_fcn_unsuper = tf.nn.dropout(Prevalence_fcn_unsuper, rate=0.2)
        Prevalence_prediction_unsuper = Prevalence_predictor(Prevalence_fcn_unsuper)
        Prevalence_prediction_silver=Prevalence_predictor_silver(Prevalence_fcn_unsuper)

        Prevalence_fcn_shuffle,visit_weights_shuffle = Attention_prevalence(inputs1_embedding_shuffle,inputs1_embedding_shuffle,inputs1_embedding_shuffle,None)
        Prevalence_fcn_shuffle = Prevalence_layer1(tf.reduce_mean(Prevalence_fcn_shuffle[:,:,:],axis=1))
        Prevalence_prediction_shffle = tf.nn.dropout(Prevalence_fcn_shuffle, rate=0.2)
        Prevalence_prediction_shffle = Prevalence_predictor(Prevalence_prediction_shffle)

        Prevalence_fcn_unsuper_mask,visit_weights_mask = Attention_prevalence(inputs_embedding_unsuper_mask,inputs_embedding_unsuper_mask,inputs_embedding_unsuper_mask,None)
        Prevalence_fcn_unsuper_mask = Prevalence_layer1(tf.reduce_mean(Prevalence_fcn_unsuper_mask[:,:,:],axis=1))
        Prevalence_prediction_unsuper_mask = tf.nn.dropout(Prevalence_fcn_unsuper_mask, rate=0.2)
        Prevalence_prediction_unsuper_mask = Prevalence_predictor(Prevalence_prediction_unsuper_mask)

        ##########getting the weights
        visit_weights_show=tf.reduce_mean(visit_weights[:,:,0,:],axis=1)
        inputs1_embedding_code_visit_rw =tf.expand_dims(tf.reduce_mean(visit_weights[:,:,0,:],axis=1),axis=-1)*inputs1_reweight
        ##########getting the weights

        visit_weights = tf.expand_dims(tf.reduce_sum(visit_weights[:, 0, :, :], axis=1), -1) + 0.1
        visit_weights_unsuper = tf.expand_dims(tf.reduce_sum(visit_weights_unsuper[:, 0, :, :], axis=1), -1) + 0.1
        visit_weights_mask = tf.expand_dims(tf.reduce_sum(visit_weights_mask[:, 0, :, :], axis=1), -1) + 0.1
        visit_weights_shuffle = tf.expand_dims(tf.reduce_sum(visit_weights_shuffle[:, 0, :, :], axis=1), -1) + 0.1


        representation_out=inputs1_embedding * visit_weights
        representation_out_unsuper = inputs_embedding_unsuper * visit_weights_unsuper
        representation_out_unsuper_mask = inputs_embedding_unsuper_mask * visit_weights_mask
        representation_out_shuffle = inputs1_embedding_shuffle * visit_weights_shuffle


        layes_num=list(str(layers_incident).split(","))
        try:
            layer_flag=-1
            print ("-------------------------------------using GRUs layers number: ",len(layes_num)," for incident ----: ", layers_incident)
            for layer_i in layes_num:
                layer_flag+=1
                layer_i=int(layer_i)
                layer_name="GRU_"+str(layer_flag)
                GRU_Bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=layer_i, return_sequences=True,
                                                                               activation=tf.nn.relu,
                                                                               recurrent_dropout=0.1), merge_mode='ave',name=layer_name)
                representation_out = GRU_Bidirectional(representation_out)
                representation_out_unsuper = GRU_Bidirectional(representation_out_unsuper)
                representation_out_unsuper_mask = GRU_Bidirectional(representation_out_unsuper_mask)
                representation_out_shuffle = GRU_Bidirectional(representation_out_shuffle)
        except:
            print ("----------------------------error of readding layers_incident: using default one layer with 80")
            layer_name = "GRU_default"
            GRU_Bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=80, return_sequences=True,
                                                                                   activation=tf.nn.relu,
                                                                                   recurrent_dropout=0.1), merge_mode='ave',name=layer_name)
            representation_out = GRU_Bidirectional(representation_out)
            representation_out_unsuper = GRU_Bidirectional(representation_out_unsuper)
            representation_out_unsuper_mask = GRU_Bidirectional(representation_out_unsuper_mask)
            representation_out_shuffle = GRU_Bidirectional(representation_out_shuffle)

        GRU_incident=tf.keras.layers.GRU(units=10, return_sequences=True,activation=tf.nn.relu)
        representation_out = tf.nn.dropout(representation_out, rate=0.1)
        representation_out_unsuper = tf.nn.dropout(representation_out_unsuper, rate=0.1)
        representation_out_unsuper_mask = tf.nn.dropout(representation_out_unsuper_mask, rate=0.1)
        representation_out_shuffle = tf.nn.dropout(representation_out_shuffle, rate=0.1)

        representation_out = GRU_incident(representation_out)
        representation_out_unsuper = GRU_incident(representation_out_unsuper)
        representation_out_unsuper_mask = GRU_incident(representation_out_unsuper_mask)
        representation_out_shuffle = GRU_incident(representation_out_shuffle)

        print ("-----representation_out: ",representation_out)
        predictor_temporal = layers.Dense(1, activation=None)
        predictor_temporal_silver = layers.Dense(1, activation=None)
        inputs1_embedding_out = representation_out#tf.nn.dropout(representation_out, rate=0.0)
        representation_out_unsuper =representation_out_unsuper# tf.nn.dropout(representation_out_unsuper, rate=0.0)
        inputs_unsuper_embedding_out_mask = representation_out_unsuper_mask#tf.nn.dropout(representation_out_unsuper_mask, rate=0.0)
        inputs1_embedding_out = predictor_temporal(inputs1_embedding_out)
        inputs_unsuper_embedding_out = predictor_temporal(representation_out_unsuper)
        prediction_silver= predictor_temporal_silver(representation_out_unsuper)
        inputs_unsuper_embedding_out_mask=predictor_temporal(inputs_unsuper_embedding_out_mask)
        print ('inputs1_embedding_out: ',inputs1_embedding_out)


        model = models.Model(inputs = [inputs1,inputs_data_embedding_all,
                                       inputs_unsuper,inputs_unsuper_masked,inputs_data_embedding_unsuper,
                                       inputs_keyfeature,inputs_keyfeature_unsuper,inputs1_shuffle],
                             outputs = [inputs1_embedding_out,inputs_unsuper_embedding_out,
                                        representation_out,representation_out_unsuper,
                                        representation_out_unsuper_mask,
                                        Prevalence_prediction,
                                        attention_value*inputs1_temp,
                                        visit_weights_show,
                                        inputs1_embedding_ori,inputs1_embedding_code_rw,
                                        inputs1_embedding_code_visit_rw,
                                        representation_out_shuffle,attention_value,
                                        Prevalence_fcn,Prevalence_fcn_shuffle,Prevalence_prediction_unsuper,
                                        Prevalence_prediction_unsuper_mask,visit_weights,prediction_silver,
                                        Prevalence_prediction_silver]) #  outputs_sque  outputs  outputs_fused
        return model
    def train_step(model, data_train, labels,weights,data_embedding_all,
                   data_embedding_unsuper,
                   data_unsuper,weights_unsuper,smooth_weight_in,
                   keyfeature,keyfeature_unsuper,patient_num,
                   silver_train,silver_unsuper,flag_silver=False):
        with tf.GradientTape(persistent=True) as tape:
            threshold_embedding=10.0
            threshold_embedding_same=5.0
            threshold_MLP=10.0
            threshold_MLP_same=5.0

            weights_value = np.sum(weights.numpy(), axis=-1)
            weights_value_unsuper = np.sum(weights_unsuper.numpy(), axis=-1)
            weights = tf.cast(weights, tf.float32)
            labels = tf.cast(labels, tf.float32)
            silver_unsuper = tf.cast(silver_unsuper, tf.float32)
            silver_train = tf.cast(silver_train, tf.float32)

            weights_unsuper = tf.cast(weights_unsuper, tf.float32)
            weights_smooth_super = tf.cast(weights, tf.float32)
            weights_unsuper = tf.cast(tf.greater(weights_unsuper,0.0), tf.float32)
            weights_smooth_super = tf.cast(tf.greater(weights_smooth_super,0.0), tf.float32)

            indices = tf.range(start=0, limit=tf.shape(data_train)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            shuffled_data_train = tf.gather(data_train, shuffled_indices)
            shuffled_weights_smooth_super = tf.gather(weights_smooth_super, shuffled_indices)
            shuffled_labels = tf.gather(labels, shuffled_indices)
            batch_num=data_unsuper.numpy().shape[0]
            if random.random() < 0.5:
                mask_unsuper = np.random.normal(loc=1, scale=0.06, size=(batch_num, max_lengh, feature_d))
            else:
                mask_unsuper = np.random.normal(loc=1, scale=0.01, size=(batch_num, max_lengh, feature_d))
            mask_dropout=[]
            for iii in range(int(batch_num*max_lengh*feature_d)):
                if random.random()<0.5:
                    if random.random()<0.95:
                        mask_dropout.append(1)
                    else:
                        mask_dropout.append(0)
                else:
                    mask_dropout.append(1)
            mask_dropout=np.array(mask_dropout).reshape((batch_num, max_lengh, feature_d))
            mask_unsuper = tf.maximum(mask_unsuper, 0.0)
            if random.random() < 0.5:
                data_unsuper_masked = data_unsuper *mask_unsuper
                data_train=data_train*mask_unsuper
            else:
                data_unsuper_masked = data_unsuper * mask_dropout
                data_train = data_train * mask_dropout


            predictions,predictions_unsuper,smooth_seq,smooth_seq_unsuper,smooth_seq_mask,Prevalence_prediction,\
            attention_value_codes,attention_value_visits, \
            embedding_ori, embedding_code_rw, embedding_code_visit_rw,shuffled_smooth_seq,\
            attention_value_code,Prevalence_fcn,Prevalence_fcn_shuffle,Prevalence_prediction_unsuper,\
            Prevalence_prediction_unsuper_mask,visit_weights,\
           prediction_silver,Prevalence_prediction_silver= \
                model([data_train,data_embedding_all,
                       data_unsuper,data_unsuper_masked,data_embedding_unsuper,
                                keyfeature,keyfeature_unsuper,shuffled_data_train,], training=True)


            prediction_incident_binary=predictions[0, weights_value[0] - 1, 0]
            prediction_incident_binary=tf.expand_dims(prediction_incident_binary,0)

            prediction_incident_binary_unsuper = predictions_unsuper[0, weights_value_unsuper[0] - 1, 0]
            prediction_incident_binary_unsuper = tf.expand_dims(prediction_incident_binary_unsuper, 0)
            prediction_incident_silver = prediction_silver[0, weights_value_unsuper[0] - 1, 0]
            prediction_incident_silver = tf.expand_dims(prediction_incident_silver, 0)

            for rowi in range(1,len(weights_value)):
                prediction_incident_binary_temp = predictions[rowi, weights_value[rowi] - 1, 0]
                prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
                prediction_incident_binary=tf.concat([prediction_incident_binary,prediction_incident_binary_temp],axis=0)

                prediction_incident_binary_temp = predictions_unsuper[rowi, weights_value_unsuper[rowi] - 1, 0]
                prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
                prediction_incident_binary_unsuper = tf.concat([prediction_incident_binary_unsuper, prediction_incident_binary_temp],
                                                       axis=0)
                prediction_incident_binary_temp = prediction_silver[rowi, weights_value_unsuper[rowi] - 1, 0]
                prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
                prediction_incident_silver = tf.concat(
                    [prediction_incident_silver, prediction_incident_binary_temp], axis=0)
            prediction_MLP_total=tf.expand_dims(prediction_incident_binary,axis=-1).numpy()
            prediction_incident_silver = tf.expand_dims(prediction_incident_silver, axis=-1)
            prediction_incident_silver=tf.nn.sigmoid(prediction_incident_silver)

            loss_MLP_incident=tf.reduce_mean(tf.abs((prediction_incident_binary)-(Prevalence_prediction)))
            loss_MLP_incident_unsuper = tf.reduce_mean(tf.abs((prediction_incident_binary_unsuper) -
                                                              (Prevalence_prediction_unsuper)))
            loss_MLP_incident=(loss_MLP_incident+loss_MLP_incident_unsuper)/2.0
            #Prevalence_prediction = tf.nn.sigmoid((tf.expand_dims(prediction_incident_binary,axis=-1)+Prevalence_prediction)/2.0)
            Prevalence_prediction = tf.nn.sigmoid(Prevalence_prediction)
            Prevalence_prediction_silver = tf.nn.sigmoid(Prevalence_prediction_silver)

            Prevalence_prediction_unsuper = tf.nn.sigmoid(Prevalence_prediction_unsuper)
            Prevalence_prediction_unsuper_mask = tf.nn.sigmoid(Prevalence_prediction_unsuper_mask)

            labels_MLP = tf.reduce_sum(labels, axis=1)
            labels_MLP = tf.cast(tf.greater_equal(labels_MLP, 1.0), tf.float32)
            silver_unsuper = tf.reduce_mean(silver_unsuper, axis=1)
            silver_train = tf.reduce_mean(silver_train, axis=1)
            AUC_silver = roc_auc_score(y_true=labels_MLP.numpy(), y_score=silver_train.numpy())
            #print ("   AUC_silver:  ",AUC_silver)

            labels_MLP_shuffle = tf.reduce_sum(shuffled_labels, axis=1)
            labels_MLP_shuffle = tf.cast(tf.greater_equal(labels_MLP_shuffle, 1.0), tf.float32)
            loss_binary_MLP = tf.keras.losses.binary_crossentropy(labels_MLP, Prevalence_prediction)
            loss_binary_MLP = tf.reduce_mean(loss_binary_MLP)
            loss_binary_MLP_silver = tf.keras.losses.binary_crossentropy(silver_unsuper, Prevalence_prediction_silver)
            loss_binary_MLP_silver = tf.reduce_mean(loss_binary_MLP_silver)
            loss_incident_silver = tf.keras.losses.binary_crossentropy(silver_unsuper, prediction_incident_silver)
            loss_incident_silver = tf.reduce_mean(loss_incident_silver)

            loss_consistency_MLP = tf.reduce_mean(tf.square(Prevalence_prediction_unsuper - Prevalence_prediction_unsuper_mask) )
            loss_consistency = tf.reduce_sum( tf.abs(smooth_seq_unsuper - smooth_seq_mask) \
                                              *tf.expand_dims(weights_unsuper, -1))/(tf.reduce_sum(weights_unsuper))

            ########smoonth loss as the non-decreasing loss
            smooth_loss=tf.abs(smooth_seq[:,0:-1,:]-smooth_seq[:,1:,:])*tf.expand_dims(weights_smooth_super[:,1:],-1)

            smooth_loss_unsuper = tf.abs(smooth_seq_unsuper[:, 0:-1, :] - smooth_seq_unsuper[:, 1:, :]) \
                                  * tf.expand_dims(weights_unsuper[:, 1:], -1)

            smooth_loss_mask = tf.abs(smooth_seq_mask[:, 0:-1, :] - smooth_seq_mask[:, 1:, :]) \
                                  * tf.expand_dims(weights_unsuper[:, 1:], -1)

            smooth_loss=tf.reduce_sum(smooth_loss,-1)
            smooth_loss_unsuper =tf.reduce_sum(smooth_loss_unsuper, -1)
            smooth_loss_mask = tf.reduce_sum(smooth_loss_mask, -1)

            predictions_unsuper=tf.nn.sigmoid(predictions_unsuper)
            entropy_unsuper = -tf.reduce_sum \
                (predictions_unsuper * tf.math.log(predictions_unsuper) + (1 - predictions_unsuper + 1e-10) * tf.math.log(
                    1 - predictions_unsuper + 1e-10)) *weights_unsuper/(tf.reduce_sum(weights_unsuper))

            entropy_unsuper_MLP = -tf.reduce_mean  (Prevalence_prediction_unsuper * tf.math.log(Prevalence_prediction_unsuper) + (
                            1 - Prevalence_prediction_unsuper + 1e-10) * tf.math.log(1 - Prevalence_prediction_unsuper + 1e-10))

            entropy_unsuper_MLP_mask = -tf.reduce_mean(Prevalence_prediction_unsuper_mask * tf.math.log(Prevalence_prediction_unsuper_mask) + (
                    1 - Prevalence_prediction_unsuper_mask + 1e-10) * tf.math.log(1 - Prevalence_prediction_unsuper_mask + 1e-10))


            ########constractive loss################
            features_half_batch=tf.reduce_sum(tf.abs(smooth_seq-shuffled_smooth_seq),axis=-1)
            weights_half_batch=weights_smooth_super+shuffled_weights_smooth_super
            weights_half_batch = tf.cast(tf.greater(weights_half_batch, 1.5), tf.float32)
            Y_half_batch=tf.reduce_mean(abs(labels-shuffled_labels),axis=-1)

            distance_same=(1-Y_half_batch)*tf.maximum(features_half_batch-threshold_embedding_same,0.0)*weights_half_batch
            distance_differ = Y_half_batch * tf.maximum(threshold_embedding-features_half_batch , 0.0)*weights_half_batch
            distance_constastive=distance_same+distance_differ
            distance_constastive =tf.reduce_sum(distance_constastive)/tf.reduce_sum(weights_half_batch)
 
            

            #################constrastive loss of MLP prediction##########
            features_half_batch =tf.reduce_sum(tf.abs(Prevalence_fcn-Prevalence_fcn_shuffle),axis=-1)
            Y_half_batch = tf.reduce_mean(abs(labels_MLP - labels_MLP_shuffle),axis=-1)
            #print("features_half_batch MLP: ", features_half_batch.numpy().shape)
            #print("Y_half_batch MLP: ", Y_half_batch.numpy().shape)
            distance_same = (1 - Y_half_batch) * tf.maximum(features_half_batch - threshold_MLP_same,
                                                            0.0)
            distance_differ = Y_half_batch * tf.maximum(threshold_MLP - features_half_batch,
                                                        0.0)
            distance_constastive_MLP = tf.reduce_mean(distance_same + distance_differ)
            #################constrastive loss of MLP prediction##########

            predictions = tf.nn.sigmoid(predictions)
            predictions = tf.cast(predictions, tf.float32)
            loss_incident = tf.keras.losses.binary_crossentropy (labels, predictions)
            loss_incident=loss_incident *weights

            #weights = tf.cast(tf.greater(weights, 0.0), tf.float32)
            weights_sum = tf.reduce_sum(weights)
            weights_sum_unsuper=tf.reduce_sum(weights_unsuper)
            loss_incident = tf.reduce_sum(loss_incident, name="sigmoid_losses")/weights_sum
            smooth_loss= tf.reduce_sum(smooth_loss, name="smooth_loss")/weights_sum
            smooth_loss_unsuper = tf.reduce_sum(smooth_loss_unsuper, name="smooth_loss_unsuper") / weights_sum_unsuper
            smooth_loss_mask = tf.reduce_sum(smooth_loss_mask, name="smooth_loss_mask") / weights_sum_unsuper

            loss_MLP =loss_binary_MLP +distance_constastive_MLP*weights_constrastive+loss_consistency_MLP*smooth_weight_in+\
                      (entropy_unsuper_MLP)*smooth_weight_in*0.5
            loss_temporal=loss_incident+(smooth_loss_unsuper+smooth_loss_mask+smooth_loss*0.5)*weight_smooth*0.5 \
                                            + distance_constastive * weights_constrastive \
                                            +loss_consistency*smooth_weight_in # +entropy_unsuper*smooth_weight_in*0.5
                           #+smooth_weight_in*(smooth_loss+smooth_loss_unsuper)/2.0+distance_constastive* weights_constrastive
                             #+entropy_unsuper*smooth_weight_in

            loss_total=loss_MLP*weights_prevalence+loss_temporal +\
                       (loss_binary_MLP_silver*weights_prevalence+loss_incident_silver)*weights_unlabel+loss_MLP_incident*smooth_weight_in
            loss_silver=loss_binary_MLP_silver*weights_prevalence+loss_incident_silver
        all_variables = model.trainable_variables

        if flag_silver==False:
            gradients = tape.gradient(loss_total, all_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, all_variables))
        else:
            gradients = tape.gradient(loss_silver, all_variables)
            optimizer_silver.apply_gradients(grads_and_vars=zip(gradients, all_variables))

        train_MLP_incident.update_state(loss_MLP_incident)
        train_loss.update_state(loss_incident)
        train_loss_classfication.update_state(loss_binary_MLP)
        train_smooth_loss.update_state(smooth_loss)
        train_smooth_loss_unsuper.update_state(smooth_loss_unsuper)
        train_keyfeature_loss.update_state(smooth_loss_unsuper)
        train_constrastive_loss.update_state(distance_constastive)
        train_constrastive_loss_MLP.update_state(distance_constastive_MLP)
        train_MLP_entropy_unsuper.update_state((entropy_unsuper+entropy_unsuper_MLP+entropy_unsuper_MLP_mask/3.0))
        train_MLP_consistency.update_state((loss_consistency+loss_consistency_MLP)*0.5)
        train_loss_silver.update_state(loss_silver)

        Prevalence_prediction_final =(Prevalence_prediction.numpy() + np.array(prediction_MLP_total)) / 2.0 #(Prevalence_prediction.numpy() + np.array(prediction_MLP_total)) / 2.0
        return loss_incident.numpy(),predictions.numpy(),Prevalence_prediction_final,\
               labels_MLP.numpy(),attention_value_codes.numpy(),\
               attention_value_visits.numpy(),embedding_ori.numpy(), \
               embedding_code_rw.numpy(), embedding_code_visit_rw.numpy(),\
               smooth_seq.numpy(),attention_value_code.numpy(),Prevalence_prediction.numpy(),np.array(prediction_MLP_total)

    def valid_step(model,data_test, labels,weights,data_embedding_test,
                   data_embedding_test_unsuper,data_unsuper,weights_unsuper,smooth_weight
                   ,keyfeature, keyfeature_unsuper,patient_num_test):
        predictions,predictions_unsuper,smooth_seq,smooth_seq_unsuper,smooth_seq_unsuper_mask,Prevalence_prediction,\
        attention_value_codes,attention_value_visits, \
            embedding_ori, embedding_code_rw, embedding_code_visit_rw,smooth_seq_shuffle,\
        attention_value_code,Prevalence_fcn,Prevalence_fcn_shuffle,Prevalence_fcn_unsuper,Prevalence_fcn_unsuper_mask,\
        visit_weights,prediction_silver,Prevalence_prediction_silver = model([data_test,
            data_embedding_test,data_unsuper,data_unsuper,data_embedding_test_unsuper,
                       keyfeature, keyfeature_unsuper,data_test])
        predictions  = tf.nn.sigmoid(predictions)

        weights_value = np.sum(weights.numpy(), axis=-1)
        prediction_incident_binary = predictions[0, weights_value[0] - 1, 0]
        prediction_incident_binary = tf.expand_dims(prediction_incident_binary, 0)

        for rowi in range(1, len(weights_value)):
            prediction_incident_binary_temp = predictions[rowi, weights_value[rowi] - 1, 0]
            prediction_incident_binary_temp = tf.expand_dims(prediction_incident_binary_temp, 0)
            prediction_incident_binary = tf.concat([prediction_incident_binary, prediction_incident_binary_temp],
                                                   axis=0)
        prediction_MLP_total = tf.expand_dims(prediction_incident_binary, axis=-1).numpy()
        Prevalence_prediction = tf.nn.sigmoid(Prevalence_prediction)
        labels = tf.cast(labels, tf.float32)
        labels_MLP = tf.reduce_sum(labels, axis=1)
        # print("labels_MLP.shape: ", labels_MLP.numpy().shape)
        labels_MLP = tf.cast(tf.greater(labels_MLP, 0.0), tf.float32)
        weights = tf.cast(weights, tf.float32)
        # weights = tf.expand_dims(weights, axis=-1)
        predictions = tf.cast(predictions, tf.float32)
        loss_binary = tf.keras.losses.binary_crossentropy(labels, predictions)
        loss_binary = loss_binary * weights
        weights_sum=tf.reduce_sum(weights)
        loss_binary = tf.reduce_sum(loss_binary, name="sigmoid_losses")/weights_sum
        loss = loss_binary
        valid_loss.update_state(loss_binary)
        Prevalence_prediction_final=(Prevalence_prediction.numpy()+np.array(prediction_MLP_total))/2.0#(Prevalence_prediction.numpy()+np.array(prediction_MLP_total))/2.0
        return loss.numpy(), predictions.numpy(),Prevalence_prediction_final,labels_MLP.numpy(),\
               attention_value_codes.numpy(),attention_value_visits.numpy(), \
               embedding_ori.numpy(), embedding_code_rw.numpy(), \
               embedding_code_visit_rw.numpy(),\
               smooth_seq.numpy(),attention_value_code.numpy(),Prevalence_prediction.numpy(),np.array(prediction_MLP_total)

    def train_model(model, ds_train, ds_valid, epochs):
        print ("--------begin model training.....")
        epoch_num=-1
        AUC_test_total=[]
        PPV_test_total=[]
        AUC_MLP_test_total=[]
        threshold = 0.5
        #flag_save = str(random.randint(1, 1500)) + "_"
        flag_save = str(1) + "_"
        flag_save_finish=False
        while(epoch_num<epochs):
            epoch_num+=1
            if len(AUC_MLP_test_total)>9:
                if np.mean(AUC_MLP_test_total[-6:-3])>np.mean(AUC_MLP_test_total[-3:]):
                    epoch_num += 2  # why here(early stop strategy?)
            if epoch_num<5:
                smooth_weight_temp=0
            else:
                smooth_weight_temp=0.03+weights_additional *(epoch_num-1 )/epochs   #  *(epoch_num-1)/epochs

            if epoch_num<epoch_silver:
                flag_silver=True
                print ("-------------------------------pre-training using silver labels-----------")
            else:
                flag_silver=False
                print("-------------------------------joint training with silver labels-----------")

            predictions_train_total=[]
            labels_train_total = []

            predictions_test_total = []
            labels_test_total = []
            weights_test_total = []
            date_test_total = []
            patient_num_test_total = []
            Prevalence_prediction_total=[]
            labels_MLP_total=[]
            Prevalence_prediction_total_test = []
            Prevalence_prediction_total_test_MLP_ONLY = []
            Prevalence_prediction_total_test_temporal_only = []

            attention_value_visit_test=[]
            attention_value_code_test=[]

            embedding_code_rw_test=[]
            embedding_code_visit_rw_test=[]

            embedding_ori_test = []
            labels_MLP_total_test = []
            embedding_code_rw_hidden_test=[]

            embedding_ori_train = []
            labels_MLP_total_train = []
            embedding_code_rw_hidden_train = []
            weights_train_total=[]
            i_number=0
            for data_train, y_train, patient_num_train, \
                date_train, weights_train,data_embedding_all_temp,\
                data_embedding_unsuper_temp,data_unsuper,weights_unsuper,\
                    keyfeature_train,keyfeature_unsuper,silver_train_batch, silver_unsuper_batch in ds_train:
                loss_out,predictions,Prevalence_prediction,labels_MLP,\
                attention_value_codes,attention_value_visits,\
                    embedding_ori, embedding_code_rw, \
                embedding_code_visit_rw,embedding_hidden,attention_value_weights,\
                    binary_MLP_only, binary_temporal_only=train_step(model,data_train, y_train ,
                                weights_train,data_embedding_all_temp,
                data_embedding_unsuper_temp,data_unsuper,weights_unsuper,smooth_weight_temp,
                                keyfeature_train,keyfeature_unsuper,patient_num_train,
                                  silver_train_batch, silver_unsuper_batch,flag_silver)


                if i_number == 0:
                    predictions_train_total = np.array(predictions)
                    labels_train_total = np.array(y_train)
                    weights_train_total = np.array(weights_train)
                    Prevalence_prediction_total= np.array(Prevalence_prediction)
                    labels_MLP_total = np.array(labels_MLP)

                    embedding_ori_train =np.array(embedding_ori)
                    labels_MLP_total_train = np.array(labels_MLP)
                    embedding_code_rw_hidden_train = np.array(embedding_hidden)
                else:
                    predictions_train_total = np.concatenate((predictions_train_total, np.array(predictions)), axis=0)
                    labels_train_total = np.concatenate((labels_train_total, np.array(y_train)), axis=0)
                    weights_train_total = np.concatenate((weights_train_total, np.array(weights_train)), axis=0)
                    Prevalence_prediction_total = np.concatenate((Prevalence_prediction_total, np.array(Prevalence_prediction)), axis=0)
                    labels_MLP_total = np.concatenate((labels_MLP_total, np.array(labels_MLP)), axis=0)

                    embedding_ori_train = np.concatenate((embedding_ori_train, np.array(embedding_ori)), axis=0)
                    labels_MLP_total_train = np.concatenate((labels_MLP_total_train, np.array(labels_MLP)), axis=0)
                    embedding_code_rw_hidden_train = np.concatenate((embedding_code_rw_hidden_train, np.array(embedding_hidden)), axis=0)

                i_number += 1
            i_number = 0
            if epoch_num%epoch_show==0 or epoch_num>epochs-2:
                for data_test, y_test, patient_num_test, date_test, \
                    weights_test,data_embedding_test_temp,data_embedding_test_temp_temp,\
                    data_unsuper,weights_unsuper,\
                    keyfeature_test,keyfeature_test_unsuper in ds_valid:
                    loss_out,predictions,Prevalence_prediction,labels_MLP,\
                    attention_value_codes,attention_value_visits,\
                        embedding_ori, embedding_code_rw, embedding_code_visit_rw,\
                    embedding_hidden,attention_value_weights_test,binary_MLP_only, binary_temporal_only=valid_step(model,
                                data_test,y_test, weights_test,
                        data_embedding_test_temp,data_embedding_test_temp_temp,data_unsuper,weights_unsuper,
                          smooth_weight_temp,keyfeature_test, keyfeature_test_unsuper,patient_num_test)

                    # Prevalence_prediction? labels_MLP?
                    # predictions, y_test, date_test, patient_num_test?
                    if i_number == 0:
                        predictions_test_total = np.array(predictions)
                        labels_test_total = np.array(y_test)
                        weights_test_total = np.array(weights_test)
                        date_test_total = np.array(date_test)
                        patient_num_test_total = np.array(patient_num_test)
                        Prevalence_prediction_total_test = np.array(Prevalence_prediction)
                        Prevalence_prediction_total_test_MLP_ONLY=np.array(binary_MLP_only)
                        Prevalence_prediction_total_test_temporal_only = np.array(binary_temporal_only)

                        attention_value_visit_test=np.array(attention_value_visits)
                        attention_value_code_test = np.array(attention_value_codes)

                        embedding_ori_test = np.array(embedding_ori)
                        embedding_code_rw_hidden_test = np.array(embedding_hidden)
                        labels_MLP_total_test = np.array(labels_MLP)

                        embedding_code_rw_test = np.array(embedding_code_rw)
                        embedding_code_visit_rw_test = np.array(embedding_code_visit_rw)

                    else:
                        predictions_test_total = np.concatenate((predictions_test_total, np.array(predictions)), axis=0)
                        labels_test_total = np.concatenate((labels_test_total, np.array(y_test)), axis=0)
                        weights_test_total = np.concatenate((weights_test_total, np.array(weights_test)), axis=0)
                        date_test_total = np.concatenate((date_test_total, np.array(date_test)), axis=0)
                        patient_num_test_total = np.concatenate((patient_num_test_total, np.array(patient_num_test)), axis=0)
                        Prevalence_prediction_total_test = np.concatenate((Prevalence_prediction_total_test, np.array(Prevalence_prediction)), axis=0)
                        Prevalence_prediction_total_test_MLP_ONLY = np.concatenate(
                            (Prevalence_prediction_total_test_MLP_ONLY, np.array(binary_MLP_only)), axis=0)
                        Prevalence_prediction_total_test_temporal_only = np.concatenate(
                            (Prevalence_prediction_total_test_temporal_only, np.array(binary_temporal_only)), axis=0)

                        labels_MLP_total_test = np.concatenate((labels_MLP_total_test, np.array(labels_MLP)),
                                                                axis=0)
                        attention_value_visit_test = np.concatenate((attention_value_visit_test, np.array(attention_value_visits)), axis=0)
                        attention_value_code_test = np.concatenate((attention_value_code_test, np.array(attention_value_codes)), axis=0)

                        embedding_ori_test = np.concatenate((embedding_ori_test, np.array(embedding_ori)), axis=0)
                        embedding_code_rw_test = np.concatenate((embedding_code_rw_test, np.array(embedding_code_rw)), axis=0)
                        embedding_code_visit_rw_test = np.concatenate((embedding_code_visit_rw_test, np.array(embedding_code_visit_rw)), axis=0)
                        embedding_code_rw_hidden_test = np.concatenate((embedding_code_rw_hidden_test, np.array(embedding_hidden)),
                                                                axis=0)
                    i_number += 1
                if flag_save_attention>0:
                    print("---saving attention values")
                    if epoch_num % 10 == 0 or epoch_num > epochs - 2:
                        savefile_name =filename_save + "Attenation_value_patient_visit_code_prediction_label_weight_test.pkl"
                        with open(dirr_save + savefile_name, 'wb') as fid:
                            pickle.dump((patient_num_test_total,attention_value_visit_test,
                                         attention_value_code_test,predictions_test_total,labels_test_total,weights_test_total), fid)

                        savefile_name =filename_save + "_embedding_patient_hiddenFCN_label_test.pkl"
                        with open(dirr_save + savefile_name, 'wb') as fid:
                            pickle.dump((patient_num_test_total, embedding_ori_test,
                                         embedding_code_rw_test,embedding_code_visit_rw_test,
                                         embedding_code_rw_hidden_test,labels_test_total,weights_test_total), fid)
                        savefile_name = filename_save + "_embedding_patient_hiddenFCN_label_train.pkl"
                        with open(dirr_save + savefile_name, 'wb') as fid:
                            pickle.dump((embedding_ori_train, embedding_code_rw_hidden_train,
                                         labels_train_total,weights_train_total), fid)
            if epoch_num % epoch_show == 0 or epoch_num > epochs - 2:
                y_true_get_train = np.array(labels_train_total,dtype=int)
                y_true_get_train=np.reshape(y_true_get_train,(-1, 1))
                # print("y_true_get_train: ", y_true_get_train.shape)

                score_get_train = np.array(predictions_train_total,dtype=float).reshape((-1, 1))
                weights_train = np.array(weights_train_total,dtype=int).reshape((-1, 1))

                try:
                    AUC_train = roc_auc_score(y_true=y_true_get_train,
                                                 y_score=score_get_train,
                                                 sample_weight=weights_train,
                                                 average='macro')

                    AUC_MLP_train=roc_auc_score(y_true=labels_MLP_total,
                                                 y_score=Prevalence_prediction_total,
                                                 average='macro')
                except:
                    AUC_train=0
                    AUC_MLP_train=0
                    print ("-----------------------------------------prevelance AUC evaluation error--------------")
                try:
                    if flag_prediction >0:
                        AUC_MLP_test=AUC_MLP_train
                        AUC_MLP_test_MLP_only=AUC_MLP_train
                        AUC_MLP_test_incident_only = AUC_MLP_train
                    else:
                        AUC_MLP_test = roc_auc_score(y_true=labels_MLP_total_test,
                                                  y_score=Prevalence_prediction_total_test,
                                                  average='macro')
                        AUC_MLP_test_MLP_only = roc_auc_score(y_true=labels_MLP_total_test,
                                                    y_score=Prevalence_prediction_total_test_MLP_ONLY,
                                                    average='macro')
                        AUC_MLP_test_incident_only = roc_auc_score(y_true=labels_MLP_total_test,
                                                    y_score=Prevalence_prediction_total_test_temporal_only,
                                                    average='macro')
                except:
                    AUC_MLP_test=0
                    AUC_MLP_test_MLP_only=0
                    AUC_MLP_test_incident_only=0
                    print ("-----------------------------------------incident AUC evaluation error--------------")
                AUC_MLP_test_total.append(AUC_MLP_test)
                Prevalence_prediction_total_binary = np.array(np.greater(Prevalence_prediction_total, threshold), dtype=np.int32)
                Prevalence_prediction_total_test_binary = np.array(np.greater(Prevalence_prediction_total_test, threshold), dtype=np.int32)
                # use threshold to decide positive & negative sample

                tn, fp, fn, tp = confusion_matrix(labels_MLP_total_test, Prevalence_prediction_total_test_binary).ravel()
                specificity_test_MLP = tn / (tn + fp)
                sensitivity_test_MLP = tp / (tp + fn)
                PPV_test_MLP = tp / (tp + fp)
                NPV_test_MLP = tn / (tn + fn)
                # Fall out or false positive rate
                FPR = fp / (fp + tn)
                # False negative rate
                FNR = fn / (tp + fn)
                acc_train_MLP=accuracy_score(labels_MLP_total,Prevalence_prediction_total_binary)
                acc_test_MLP=accuracy_score(labels_MLP_total_test,Prevalence_prediction_total_test_binary)
                logs = '---AUC_B_train: {}, AUC_B_comb:{} ,AUC_B_TF_only:{},AUC_B_incident_only:{} ' \
                       'PPV_MLP:{},' \
                       'specif_B:{},' \
                       'sensi_B:{},' \
                       'NPV_B:{} Loss_constr_B :{}, Loss_entro_B :{}, Loss_consis_B :{},' \
                       '  Consis_MLP_incident: {}, loss_silver_B: {}'
                tf.print(tf.strings.format(logs, (AUC_MLP_train,
                      AUC_MLP_test,AUC_MLP_test_MLP_only,AUC_MLP_test_incident_only,
                   PPV_test_MLP, specificity_test_MLP, sensitivity_test_MLP,NPV_test_MLP,
                    train_constrastive_loss_MLP.result(),train_MLP_entropy_unsuper.result(),train_MLP_consistency.result(),
                       train_MLP_incident.result(),train_loss_silver.result())))

                labels_test_total = np.array(labels_test_total).reshape((-1, 1))
                predictions_test_total = np.array(predictions_test_total).reshape((-1, 1))
                weights_test_total = np.array(weights_test_total).reshape((-1, 1))
                date_test_total = np.array(date_test_total).reshape((-1, 1))
                patient_num_test_total = np.array(patient_num_test_total).reshape((-1, 1))
                y_true_get_test=[]
                score_get_test=[]
                date_test=[]
                patient_num_test=[]
                for samplei in range(len(labels_test_total)):
                    if int(weights_test_total[samplei]) > 0:
                            y_true_get_test.append(labels_test_total[samplei])
                            score_get_test.append(predictions_test_total[samplei])
                            date_test.append(date_test_total[samplei])
                            patient_num_test.append(patient_num_test_total[samplei])

                y_true_get_test=np.array(y_true_get_test)
                score_get_test = np.array(score_get_test)
                date_test = np.array(date_test)
                patient_num_test = np.array(patient_num_test)

                if flag_prediction >0:
                    y_true_get_test_temp=[]
                    for ij in range(len(y_true_get_test)):
                        y_true_get_test_temp.append(random.randint(0,1))
                    y_true_get_test=np.array(y_true_get_test_temp)
                score_get_test_binary=np.array(np.greater(score_get_test, threshold), dtype=np.int32)
                AUC_test = roc_auc_score(y_true_get_test, score_get_test,  average="macro")
                tn, fp, fn, tp = confusion_matrix(y_true_get_test, score_get_test_binary).ravel()
                specificity = tn / (tn + fp)
                sensitivity = tp / (tp + fn)
                PPV = tp / (tp + fp)
                NPV = tn / (tn + fn)
                f_1 = (2 * tp) / (2 * tp + fp + fn)
                AUC_test_total.append(AUC_test)
                PPV_test_total.append(PPV)
                logs = '---Epoch={},Loss:{}, Loss_binary:{}  Loss_smooth:{},' \
                       'Loss_smooth_un:{},Loss_constra:{},' \
                       'Auc_train:{},' \
                               'AUC:{},F1:{}, PPV:{},Speci:{},Sensi:{}, NPV:{}'
                tf.print(tf.strings.format(logs,(epoch_num, train_loss.result(),
                                    train_loss_classfication.result(), train_smooth_loss.result(),
                                            train_smooth_loss_unsuper.result(),train_constrastive_loss.result(),
                                                 AUC_train,
                                                    AUC_test,f_1,PPV,specificity,sensitivity,NPV)))

                patient_num_test = np.squeeze(patient_num_test)
                date_test = np.squeeze(date_test)
                score_get_test = np.squeeze(score_get_test)
                y_true_get_test = np.squeeze(y_true_get_test)

                if epoch_num > epochs - 2 :
                    if True:
                        print("---------saving--- ", dirr_save, ": ", filename_save)
                        patient_num_test_total2, date_test_total2, y_pred_get2, y_true_get2 = \
                            zip(*sorted(zip(patient_num_test, date_test, score_get_test, y_true_get_test)))
                        dataframe = pd.DataFrame({'Patient_num': patient_num_test_total2,
                                                  'Date': date_test_total2,
                                                  "Y_pred": y_pred_get2,
                                                  "Y_label:": y_true_get2})
                        dataframe.to_csv(dirr_save + "Incident_epoch"+str(epoch_num) + filename_save +'.csv', index=True, sep=',')

                        #############saving codes weights
                        savename_weights=dirr_save +"_"+ filename_save+"_code_weights.csv"
                        if not os.path.exists(savename_weights):
                            df=pd.DataFrame({})
                        else:
                            df=pd.read_csv(savename_weights)
                        weights_save=list(attention_value_weights_test[0,0,:])
                        df["weight"+str(random.randint(0,1000))]=weights_save
                        df.to_csv(savename_weights,index=False)
                    if True:
                        print("---------saving--- ", dirr_save, ": ", filename_save)
                        print ("Prevalence_prediction_total_test: ",len(Prevalence_prediction_total_test))
                        print ("len(Prevalence_prediction_total_test): ",np.array(Prevalence_prediction_total_test).shape)
                        print("len(labels_MLP_total_test): ", np.array(labels_MLP_total_test).shape)
                        dataframe = pd.DataFrame({'Predition':np.array(Prevalence_prediction_total_test).reshape(len(Prevalence_prediction_total_test),),
                                                  'Y': np.array((labels_MLP_total_test)).reshape(len(labels_MLP_total_test),)})
                        dataframe.to_csv(dirr_save + "Prevalence_"+filename_save + '.csv', index=True, sep=',')
                        print("---------saving ends successfully--- ", dirr_save, ": ", filename_save)


                if epoch_num > epochs - 2 and flag_save_finish==False :
                    flag_save_finish=True
                    if True:
                        f = open(dirr_save + filename_save + "_incident_evaluation.txt", 'a')
                        f.write("Threshold value=0.5 AUC_train :%4f , NPV:%4f ,Specificity:%4f, F1:%4f ,"
                                " PPV:%4f , Sensitvity:%4f, AUC:%4f" % (
                            AUC_train, NPV,specificity, f_1, PPV, sensitivity, AUC_test))
                        f.write("\r")
                        f.close()
                    if True:
                        f = open(dirr_save + filename_save + "_prevalence_evaluation.txt", 'a')
                        f.write(" Threshold value=0.5 AUC_train :%4f , ACC_train:%4f, ,FPR:%4f, FNR:%4f, NPV_MLP:%4f "
                                " ,AUC_test_comb:%4f,  AUC_test_TF:%4f, AUC_test_incident:%4f,"
                                "ACC_test:%4f "
                                ", Speci:%4f ,Sensi:%4f, PPV:%4f"
                                 % (
                            AUC_MLP_train, acc_train_MLP, FPR,FNR, NPV_test_MLP,AUC_MLP_test,
                            AUC_MLP_test_MLP_only,AUC_MLP_test_incident_only,
                            acc_test_MLP,specificity_test_MLP,
                            sensitivity_test_MLP,
                            PPV_test_MLP))
                        f.write("\r")
                        f.close()
            train_loss.reset_states()
            valid_loss.reset_states()
            train_metric.reset_states()
            valid_metric.reset_states()
            train_smooth_loss.reset_states()
            train_smooth_loss_unsuper.reset_states()
            train_keyfeature_loss.reset_states()
            train_constrastive_loss.reset_states()
            train_constrastive_loss_MLP.reset_states()
            train_MLP_entropy_unsuper.reset_states()
            train_MLP_consistency.reset_states()
            train_MLP_incident.reset_states()
            train_loss_silver.reset_states()

    model = Model_prediction()
    savename_model = dirr_save + filename_save+"_model"
    
    if os.path.exists(savename_model) and flag_load_model>0:
        print("---------------------------------------------loadding saved model....................")
        #model = load_model(savename_model)
        model = tf.keras.models.load_model(savename_model)
    train_model(model, ds_train, ds_test, epochs=train_epoches)
    print("---------------------------------------------saving model....................")
    #savename_model = dirr_save + filename_save + "_model+"
    model.save(savename_model)




