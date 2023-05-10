import pandas as pd
import numpy as np
import csv

d = 10           # dimension of embedding
l = 10           # covariants column numbers(eg. for 'S.10' for max, l = 10)
Sigma = np.zeros((l,l))
Norms = np.zeros(l)
dirr_data = 'C:/Users/NORTH/source/incident_phenotyping/data/'
dirr_example = 'C:/Users/NORTH/source/incident_phenotyping/example/'
embedding_file= "embedding_example.csv"
df_embedding = np.array(pd.read_csv(dirr_example + embedding_file))
emb_sele = df_embedding[0:d,1:1+l]

for i in range(0,l):
    Norms[i] = np.linalg.norm(emb_sele[:,i])
    emb_sele[:,i] = emb_sele[:,i] / Norms[i]

for i in range(0,l):
    for j in range(0,i+1):
        Sigma[i,j] = Sigma[j,i] = np.dot(emb_sele[:,i], emb_sele[:,j])

#print(Sigma)

#print(emb_sele.shape)

np.save(dirr_data+'embedding_selected',emb_sele)
np.save(dirr_data+'Sigma',Sigma)


embed_sele_csv = pd.DataFrame(emb_sele)
embed_sele_csv.to_csv(dirr_data+'embedding_selected.csv',index= False, header= ['S.1','S.2','S.3','S.4','S.5','S.6','S.7','S.8','S.9','S.10'])

Sigma_csv = pd.DataFrame(Sigma)
Sigma_csv.to_csv(dirr_data+'Sigma.csv',index= False, header= True)

# Norms = np.around(Norms * 10)
Norms_csv = pd.DataFrame(Norms)
Norms_csv.to_csv(dirr_data+'Norms.csv',index= False, header= True)
# 不要头文件，不要列索引

# 查看保存的文件，不要头文件
