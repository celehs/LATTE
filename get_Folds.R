
library(data.table)


filter_patient = function(patient.dat){

  dic_patient_firstDate[1]= label.patients[[1]]$T[1]
  dic_patient_lastDate[1] = label.patients[[1]]$T[nrow(label.patients[[1]])]

  lapply(label.patients,function(x){})
  fdate = ldate = lab.date = 0
  for (rowi in seq_len(nrow(label.dat))[-1]){

    #Patients_all[Patients[rowi]]=1
    if (Patients[rowi-1]!=Patients[rowi]){
      dic_patient_firstDate[]= Times[rowi]
    }
    dic_patient_lastDate[Patients[rowi-1]+1] = Times[rowi-1]
    if (Patients[rowi-1]==Patients[rowi] &  Labels[rowi-1]==0 & Labels[rowi]==1){
      dic_patient_labelDate[Patients[rowi]+1] = Times[rowi]
    }
    if (Patients[rowi-1]!=Patients[rowi] & Labels[rowi]==1){
      dic_patient_labelDate[Patients[rowi]+1] = Times[rowi]
    }
  }



  return(list(fDate = fDate, lDate = lDate, labelDate = labelDate, ))

}



set.seed(181694)
#phe.nm = "SimDat.1.1"; Observation_years = 5; ntrain = 300; ntest = 170
#Trimesters, years, and years. First try years, then in 3 month periods, i.e. trimesters.
getFolds_RP_v2 = function(phe.nm,Observation_years,ntrain,ntest,data_name){

  all.dat = list.files(paste0(mdir,"Simulation/SimDat/",phe.nm),full.names=T, pattern = "labeled.csv")

  label.dat = fread(all.dat[1],data.table=F);
  label.patients = split(label.dat,label.dat$ID)
  n = length(label.patients)

  dic_patient_firstDate = matrix(0,n,1)
  dic_patient_lastDate = unlist(lapply(label.patients,function(x){x$T[nrow(x)]}))
  dic_patient_labelDate = unlist(lapply(label.patients,function(x){x$T[grep(1,x$Y)[1]]}))

  Times = label.dat$T; Labels = label.dat$Y
  Labels.all = unlist(lapply(label.patients,function(x){x$Y[nrow(x)]}))
  Patients_all = unlist(lapply(label.patients,function(x){x$ID[nrow(x)]}))

  length((dic_patient_labelDate)); length((dic_patient_firstDate)); length((dic_patient_lastDate))
  print (paste0("----positive patients with a negative period: ", sum(!is.na(dic_patient_labelDate))))
  ###########remove false positives
  patients_false_positive = Patients_all[!is.na(dic_patient_labelDate) & dic_patient_labelDate <= Observation_years]

  ###########remove patients with less than Observation_years
  patients_too_small= Patients_all[dic_patient_lastDate <= Observation_years]


  print(paste0("----total patients : ", length(Patients_all)))
  print(paste0("----positive patients : ", length(dic_patient_labelDate[!is.na(dic_patient_labelDate)])-length(patients_false_positive)))
  print(paste0("----negative patients : ", length(Patients_all)-(length(dic_patient_labelDate[!is.na(dic_patient_labelDate)])
                                                                 - length(patients_false_positive))))
  Patients_final = setdiff(Patients_all,c(patients_false_positive,patients_too_small))
  print(paste0("----usable patients : ", length(Patients_final)))
  #length(Patients_final)

  summary(do.call(rbind,lapply(label.patients[match(Patients_final,Patients_all)],function(x){x$T[nrow(x)]})))

  ##############
  Final.Labels = Labels.all[match(Patients_final,Patients_all)]
  fwrite(cbind(Patients_final,Final.Labels), file = paste0(mdir,"Simulation/SimDat/",phe.nm,"/",phe.nm,"_label_patient_info.csv"))
  pos.frac = sum(Final.Labels)/length(Final.Labels)
  #npos = pos.frac*ntrain
  npos = floor(pos.frac*ntest)

  # Training set: n.train = ntest, 400, 600; Testing set: n.test = ntest
  train_patients = matrix(0,length(Final.Labels)-ntest,1)
  test_patients = matrix(0,ntest,1)

  #do.call(rbind,lapply(label.patients[test_patients[,i]+1],function(x){x[grep(1,x$Y)[1],c("ID","T","Y")]}))
  patients.neg = sample(Patients_final[Final.Labels==0],replace = T)
  patients.pos = sample(Patients_final[Final.Labels==1], replace = T)
  test.1 = patients.pos[1:npos]; test.0 = patients.neg[1:(ntest-npos)]
  train.1 = patients.pos[-c(1:npos)]; train.0 = patients.neg[-c(1:(ntest-npos))]


  train_patients[,1] = sample(c(train.1,train.0))
  test_patients[,1] = sample(c(test.1,test.0))


  train.inds = train_patients[1:ntrain,]
  test.inds = test_patients
  fwrite(data.frame(train.inds),file = paste0(mdir,"Simulation/SimDat/",phe.nm,"/train_patients_",ntrain,".csv"))

  fwrite(data.frame(test.inds),file = paste0(mdir,"Simulation/SimDat/",phe.nm,"/test_patients.csv"))


  if (dir.exists(paste0(mdir,"Simulation/SimDat/",phe.nm,"/train/")) == 0){
    dir.create(paste0(mdir,"Simulation/SimDat/",phe.nm,"/train/"))
    dir.create(paste0(mdir,"Simulation/SimDat/",phe.nm,"/test/"))
  }

  flag = 1
      for (train_inds in train.inds){
        if (flag == 1){
            train_data = data.frame(label.patients[train_inds])
            colnames(train_data) = c('V1','X','ID','Y','T','S.1','S.2','S.3','S.4','S.5','S.6','S.7','S.8','S.9','S.10')
            # print(dim(train_data))
            flag = 0
        }
        else{
            train_new_data = data.frame(label.patients[train_inds])
            colnames(train_new_data) = colnames(train_data)
            train_data = data.frame(rbind(train_data,train_new_data))
        }
      }

      for (test_inds in test.inds){
        if (flag == 0){
            test_data = data.frame(label.patients[test_inds])
            # print(dim(train_data))
            colnames(test_data) = c('V1','X','ID','Y','T','S.1','S.2','S.3','S.4','S.5','S.6','S.7','S.8','S.9','S.10')
            flag = 1
        }
        else{
            test_new_data = data.frame(label.patients[test_inds])
            colnames(test_new_data) = colnames(test_data)
            test_data = data.frame(rbind(test_data,test_new_data))
        }
      }


  #print(dim(train_data))
  #print(colnames(train_data))

  fwrite(train_data,file =  paste0(mdir,"Simulation/SimDat/",phe.nm,"/train/train_data.csv"))
  fwrite(test_data,file =  paste0(mdir,"Simulation/SimDat/",phe.nm,"/test/test_data.csv"))

  # save split data(for main method)


  ##########Unlabeled patients##########

  unlabel.dat = fread(all.dat[2],data.table=F)
  unlabel.patients = split(unlabel.dat,unlabel.dat$ID)
  n.u = length(unlabel.patients)

  dic_patient_firstDate_unlabel = matrix(0,n.u,1)
  dic_patient_lastDate_unlabel = unlist(lapply(unlabel.patients,function(x){x$T[nrow(x)]}))
  Patients_all_unlabel = unlist(lapply(unlabel.patients,function(x){x$ID[1]}))


  ###########remove patients with less than Observation_years
  patients_too_small_unlabel = Patients_all_unlabel[dic_patient_lastDate_unlabel <= Observation_years]

  sapply(unlabel.patients[dic_patient_lastDate_unlabel <= Observation_years],function(x) x$T[nrow(x)])


  print(paste0("----total patients : ", length(Patients_all_unlabel)))
  unlabeled_patients = sample(setdiff(Patients_all_unlabel,patients_too_small_unlabel),replace = T)

  summary(do.call(rbind,lapply(unlabel.patients[match(unlabeled_patients,Patients_all_unlabel)],function(x){x$T[nrow(x)]})))

  fwrite(data.frame(unlabeled_patients),file = paste0(mdir,"Simulation/SimDat/",phe.nm,"/unlabeled_patients.csv"))

}



