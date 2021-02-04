mycols=c('#EC407A', 'black', '#66C2A5','#FC8D62','#8DA0CB','#E78AC3','#A6D854','#FFD92F')
set.seed(42)
file_path = '../output/RF_100estimators_10folds_30reps_2021-01-29_12-12-54/' # put path to the file created by pipeline.py
dir.create(file.path(file_path, 'recovery_predictions'))
cytof <- read.csv('../data/immune/HipValidation_cytof_Pre_stim_baseline_adjusted_all_plates.csv')
acti <- read.csv(paste(file_path, 'extracted_recovery_features/univar_surrogate/univar_surrogate.csv', sep=''))
prot <- read.csv('../data/proteomics/olink_Pre.csv')

cytof <- cytof[,-1]
acti <- acti[,-1]
prot <- prot[,-1]
cytof = cytof[seq(35),]#last plate not usable
acti = acti[seq(35),]
prot = prot[seq(35),]
##impute missing values
library(missForest)
cytof=missForest(cytof)$ximp
prot=missForest(prot)$ximp

library(RGCCA)
##initializing values for cca of cytof
index=seq(35)
A=list(as.matrix(acti[index,]), as.matrix(cytof[index,]))
result.rgcca = rgcca(A)
Ytest = matrix(NA, nrow(A[[1]]), 2)
actiweights=result.rgcca$a[[1]]
cytofweights=result.rgcca$a[[2]]
##leave one out cross-validation
for (i in 1:nrow(A[[1]])){
    B = lapply(A, function(x) x[-i, ])
    B = lapply(B, scale2)
    resB = rgcca(B, scale = FALSE, verbose = FALSE)
    for (k in 1:length(B)){
        if (cor(result.rgcca$a[[k]], resB$a[[k]]) >= 0) 
            resB$a[[k]] = resB$a[[k]] else resB$a[[k]] = -resB$a[[k]]
    }
    Btest =lapply(A, function(x) x[i, ])
    Btest[[1]]=(Btest[[1]]-attr(B[[1]],"scaled:center")) /
        (attr(B[[1]],"scaled:scale"))/sqrt(NCOL(B[[1]]))
    Btest[[2]]=(Btest[[2]]-attr(B[[2]],"scaled:center")) / 
        (attr(B[[2]],"scaled:scale"))/sqrt(NCOL(B[[2]]))
    Ytest[i, 1] = Btest[[1]]%*%resB$a[[1]]
    Ytest[i, 2] = Btest[[2]]%*%resB$a[[2]]
    actiweights=actiweights+resB$a[[1]]
    cytofweights=cytofweights+resB$a[[2]]
}
cytofweights=cytofweights/nrow(A[[1]])
actiweights=actiweights/nrow(A[[1]])
write.csv(cytofweights, paste(file_path, 'recovery_predictions/cca-cytof-acti_cytofweights.csv', sep=''))
write.csv(actiweights, paste(file_path, 'recovery_predictions/cca-cytof-acti_actiweights.csv', sep=''))
pdf(paste(file_path, 'recovery_predictions/cca-cytof-acti.pdf', sep=''))
plot(Ytest[,1], Ytest[,2], xlab='Cellular Response', ylab='Activity')
cor=cor.test(Ytest[,1], Ytest[,2], method='spearman')
legend('topleft', inset=0.05, legend=sprintf('pv=%f\nR=%f',cor$p.value, cor$estimate))
write.csv(Ytest, paste(file_path, 'recovery_predictions/cca-cytof-acti.csv', sep=''))
dev.off()

##initializing values for cca of cytof
index=seq(35)
A=list(as.matrix(acti[index,]), as.matrix(prot[index,]))
result.rgcca = rgcca(A)
Ytest = matrix(NA, nrow(A[[1]]), 2)
actiweights=result.rgcca$a[[1]]
protweights=result.rgcca$a[[2]]
##leave one out cross-validation
for (i in 1:nrow(A[[1]])){
    B = lapply(A, function(x) x[-i, ])
    B = lapply(B, scale2)
    resB = rgcca(B, scale = FALSE, verbose = FALSE)
    for (k in 1:length(B)){
        if (cor(result.rgcca$a[[k]], resB$a[[k]]) >= 0) 
            resB$a[[k]] = resB$a[[k]] else resB$a[[k]] = -resB$a[[k]]
    }
    Btest =lapply(A, function(x) x[i, ])
    Btest[[1]]=(Btest[[1]]-attr(B[[1]],"scaled:center")) /
        (attr(B[[1]],"scaled:scale"))/sqrt(NCOL(B[[1]]))
    Btest[[2]]=(Btest[[2]]-attr(B[[2]],"scaled:center")) / 
        (attr(B[[2]],"scaled:scale"))/sqrt(NCOL(B[[2]]))
    Ytest[i, 1] = Btest[[1]]%*%resB$a[[1]]
    Ytest[i, 2] = Btest[[2]]%*%resB$a[[2]]
    actiweights=actiweights+resB$a[[1]]
    protweights=protweights+resB$a[[2]]
}
protweights=protweights/nrow(A[[1]])
actiweights=actiweights/nrow(A[[1]])
write.csv(protweights, paste(file_path, 'recovery_predictions/cca-prot-acti_protweights.csv', sep=''))
write.csv(actiweights, paste(file_path, 'recovery_predictions/cca-prot-acti_actiweights.csv', sep=''))

pdf(paste(file_path, 'recovery_predictions/cca-prot-acti.pdf', sep=''))
plot(Ytest[,1], Ytest[,2], xlab='Plasma Proteins', ylab='Activity')
cor=cor.test(Ytest[,1], Ytest[,2], method='spearman')
legend('topleft', inset=0.05, legend=sprintf('pv=%f\nR=%f',cor$p.value, cor$estimate))
write.csv(Ytest, paste(file_path, 'recovery_predictions/cca-prot-acti.csv', sep=''))
dev.off()


