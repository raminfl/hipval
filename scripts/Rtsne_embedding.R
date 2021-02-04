library(Rtsne)

set.seed(1)
file_path = '../output/RF_100estimators_10folds_30reps_2021-01-29_12-12-54/activity_predictions_preIncluded/' # put path to the file created by pipeline.py
data=read.csv(paste(file_path,'Features_vs_days_clustered.csv', sep='')) 
clusters=data[,ncol(data)]
data=data[,-ncol(data)]
names=data[,1]
data=data[,-1]
library(Rtsne)
ts=Rtsne(data, perplexity=5, check_duplicates=F)
# plot(ts$Y, pch=20, col=clusters+1, cex=2)
# print(ts$Y)
# print(ts)
write.csv(ts$Y, paste(file_path,'Actigraph_Rtsne_embedding.csv', sep=''))