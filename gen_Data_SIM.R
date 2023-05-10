# TODO: change the home dir


library(MAP)
library(ff)
library(data.table)

args = commandArgs(trailingOnly = TRUE)


total_data_num = as.numeric(args[1])
# 37736
labeled_data_num = as.numeric(args[2])
# 1000
prop.train = as.numeric(args[3])


#"cum": cumulative counts; "stacked": stacked counts
num.train = round(labeled_data_num * 0.7 * prop.train)
num.test = round(labeled_data_num * 0.7 * (1-prop.train))


#----------------------------------------
# data generation #
#----------------------------------------
source('./set_mdir.R')
source(paste0(mdir,'/simulation_data_functions_v4.R'))
source(paste0(mdir,"/get_Folds.R"))

A<-matrix(rep(c(1,-2),50),10)
pmean <- function(x,y) (x+y)/2
A[] <- pmean(A, matrix(A, nrow(A), byrow=TRUE))
s<-matrix(rep(1,100),10)-2*A+diag(rep(c(0.5,-4.5),5))


Sigma_filename = 'simdata_generate_rely/Sigma.csv'
Sigma = read.csv(paste0(mdir, Sigma_filename))
Norms_filename = 'simdata_generate_rely/Norms.csv'
Norms = read.csv(paste0(mdir, Norms_filename))

# print(Sigma)


# data.1.1


gen_data_2(total_data_num, p=10, Sigma, Norms, T.family = 'linear',
     T.b0 = -3, T.beta = 0.5*c(1,-1,-2,-2, 1,-1,-2,-2, 1,-2), T.coef = 0.05,
     C.rate = 0.5, filename = paste0(mdir,'Simulation/SimDat/SimDat.1/','SimDat.1.csv'),dataname =paste0(mdir,'Simulation/SimDat/SimDat.1/','SimDat.1.Rds'))


# data.1.2

gen_data_2(total_data_num, p=10, Sigma, Norms, T.family = 'quadratic',
     T.b0 = -30, T.beta = 0.5*c(1,-1,-2,-2, 1,-1,-2,-2, 1,-2), T.B = -1*s, T.coef = 0.05,
     C.rate = 0.5, filename = paste0(mdir,'Simulation/SimDat/SimDat.2/','SimDat.2.csv'),dataname =paste0(mdir,'Simulation/SimDat/SimDat.2/','SimDat.2.Rds'))



for (data.file in c('SimDat.1','SimDat.2')) {
   label_split(data.file,label.num = labeled_data_num, total = total_data_num)
   getFolds_RP_v2(phe.nm = data.file, Observation_years = 2, ntrain = num.train, ntest = num.test)
}

