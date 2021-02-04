mycols=c('#EC407A', 'black', '#66C2A5','#FC8D62','#8DA0CB','#E78AC3','#A6D854','#FFD92F')
set.seed(42)
#get the univariate correlations shown in Supplementary Figure 5
file_path = '../output/RF_100estimators_10folds_30reps_2021-01-29_12-12-54/' # put path to the file created by pipeline.py
dir.create(file.path(file_path, 'recovery_predictions'))
cytof <- read.csv('../data/immune/HipValidation_cytof_Pre_stim_baseline_adjusted_all_plates.csv')
acti <- read.csv(paste(file_path, 'extracted_recovery_features/univar_surrogate/univar_surrogate.csv', sep=''))
prot <- read.csv('../data/proteomics/olink_Pre.csv')

cytof <- cytof[,-1]
acti <- acti[,-1]
prot <- prot[,-1]
##impute missing values
library(missForest)
cytof=missForest(cytof)$ximp
prot=missForest(prot)$ximp
##calculate all the edges to be drawn
index=seq(35)
acti = acti[index,]
cytof = cytof[index,]
prot = prot[index,]

index=matrix(NA, 0, 2)
# all_adjusted_pvals = matrix(1, ncol(cytof), ncol(acti))
# all_rhos = matrix(0, ncol(cytof), ncol(acti))
all_adjusted_pvals = matrix(1, ncol(acti), ncol(cytof))
all_rhos = matrix(0, ncol(acti), ncol(cytof))
for (j in seq(ncol(cytof))){
# for (j in seq(ncol(acti))){
    pvs=vector()
    rhos=vector()
    for (i in seq(ncol(acti))){
        spearman_corr = cor.test(cytof[,j], acti[,i], method='spearman')
        pvs[i]=spearman_corr$p.value
        rhos[i]=spearman_corr$estimate
    }
    all_adjusted_pvals[,j] = p.adjust(pvs, method='fdr')
    all_rhos[,j] = rhos
    temp=which(p.adjust(pvs, method='fdr')<0.05)
    # temp=which(pvs<0.05)
    # temp=which(pvs<0.01)
    index=rbind(index,cbind(temp, rep(j, length(temp))))
}
write.csv(all_adjusted_pvals, paste(file_path, 'recovery_predictions/fig5_supp_cytof_acti_pval_fdr_corrected.csv', sep=''))
write.csv(all_rhos, paste(file_path, 'recovery_predictions/fig5_supp_cytof_acti_rho.csv', sep=''))

index2=matrix(NA, 0, 2)
all_adjusted_pvals = matrix(1, ncol(acti), ncol(prot))
all_rhos = matrix(0, ncol(acti), ncol(prot))
for (j in seq(ncol(prot))){
    pvs=vector()
    rhos=vector()
    for (i in seq(ncol(acti))){
        spearman_corr = cor.test(prot[,j], acti[,i], method='spearman')
        pvs[i]=spearman_corr$p.value
        rhos[i]=spearman_corr$estimate
        }
    all_adjusted_pvals[,j] = p.adjust(pvs, method='fdr')
    all_rhos[,j] = rhos
    print(all_rhos[,j])
    print(all_adjusted_pvals[,j])
    temp=which(p.adjust(pvs, method='fdr')<0.05)
    index2=rbind(index2,cbind(temp, rep(j, length(temp))))
}
write.csv(all_adjusted_pvals, paste(file_path, 'recovery_predictions/fig5_supp_prot_acti_pval_fdr_corrected.csv', sep=''))
write.csv(all_rhos, paste(file_path, 'recovery_predictions/fig5_supp_prot_acti_rho.csv', sep=''))