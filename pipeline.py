import os
import sys 
import time
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
import multiprocessing
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, ParameterGrid, cross_validate
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
import matplotlib
from matplotlib import cm
# import networkx as nx
# sys.path.append('/home/raminf/HipVal/recovery_features/New Ramin/')  
# from recovery_segmented_regression.recovery_segmented_regression import RecoverySegmentedRegression



class recovery_prediction_pipeline(object):
    
    def __init__(self, main_save_folder, actigraph_filename, cytof_filename, olink_filename, clinical_report_filename, no_of_folds=10, no_of_reps=30, RF_no_of_estimators=100):
        self.actigraph_filename = actigraph_filename
        self.no_of_folds = no_of_folds
        self.no_of_reps = no_of_reps
        self.RF_no_of_estimators = RF_no_of_estimators
        self.cytof_filename = cytof_filename
        if not main_save_folder:
            self.main_save_folder = './output/RF_{0:}estimators_{1:}folds_{2:}reps_'.format(self.RF_no_of_estimators,self.no_of_folds,self.no_of_reps)+datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')+'/'
        else:
            self.main_save_folder = main_save_folder
        self.actigraph_predictions_save_folder = self.main_save_folder+'activity_predictions_preIncluded/'
        self.single_feat_actigraph_predictions_save_folder = self.main_save_folder+'single_feat_activity_predictions/'
        self.extract_recovery_feature_save_folder = self.main_save_folder+'extracted_recovery_features/'
        self.clinical_vs_surrogate_univ_corr_save_folder = self.main_save_folder+'clinical_vs_surrogate_univ_corr/'
        self.clinical_vs_preop_univ_corr_save_folder = self.main_save_folder+'clinical_vs_preop_univ_corr/'
        self.surrogate_vs_preop_univ_corr_save_folder = self.main_save_folder+'surrogate_vs_preop_univ_corr/'
          
        self.olink_filename = olink_filename    
        self.clinical_report_filename = clinical_report_filename    
        self.recovery_feature_predictions_save_folder = self.main_save_folder+'recovery_feature_predictions/baseline_pred/'

        os.makedirs(self.main_save_folder, exist_ok=True)
        params_filename = self.main_save_folder+'params.txt'
        if not main_save_folder:
            with open(params_filename, 'w') as filetowrite:
                pipeline_params = []
                pipeline_params.append('no_of_folds = {}\n'.format(self.no_of_folds))
                pipeline_params.append('no_of_reps = {}\n'.format(self.no_of_reps))
                pipeline_params.append('RF_no_of_estimators = {}\n'.format(self.RF_no_of_estimators))
                pipeline_params.append('main_save_folder = {}\n'.format(self.main_save_folder))
                filetowrite.writelines(pipeline_params)
                filetowrite.close()


    def repeated_k_folds_per_patient_single_feature(self, output_array,single_feat,patient,X,Y,model_selection=False, param_grid = [{'n_estimators': [50,100,200], 'loss': ['linear', 'square', 'exponential']}]):
        # print('**************************** repeated_k_folds_per_patient')

       
        # df_feat_importance_col = ['iteration', 'fold']
        # df_feat_importance_col.extend(features)
        # df_feat_importance = pd.DataFrame(columns=df_feat_importance_col)
        
        # params for rf regressor param_grid = [{'n_estimators': [50,100,200], 'max_depth': [2, 5, 10, None]}]
        for k in range(1, self.no_of_reps+1):
            df_predictions = pd.DataFrame(columns=['single_feat', 'patient_id', 'ground_truth', 'predicted'])
            # df_predictions_preIncluded = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted'])
            kf = KFold(n_splits=self.no_of_folds, shuffle=True, random_state=k)
            
            fold = 0
            X = X.reshape(-1,1)
            for train_index, test_index in kf.split(X):
                # print('rep = {}, fold = {}'.format(k,fold))
            
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                
                # X_test_combined = np.concatenate((X_test, X_pre), axis=0)
                # Y_test_combined = np.concatenate((y_test, Y_pre), axis=0)
                
                # scalerX = StandardScaler()
                # scalerX.fit(X_train)
                # X_train = scalerX.transform(X_train)
                # X_test = scalerX.transform(X_test)
                # X_test_combined = scalerX.transform(X_test_combined)

                # scalerY = StandardScaler()
                # scalerY.fit(y_train.reshape(-1, 1))
                # y_train = scalerY.transform(y_train.reshape(-1, 1))
                # y_test = scalerY.transform(y_test.reshape(-1, 1))
                # Y_test_combined = scalerY.transform(Y_test_combined.reshape(-1, 1))

                max_iter = 1000000
                if model_selection:
                    # grid = GridSearchCV(estimator=estimator(random_state=k), param_grid=param_grid, cv=5, n_jobs=10, iid=True)
                    grid = GridSearchCV(estimator=estimator(DecisionTreeRegressor(max_depth=4), random_state=k), param_grid=param_grid, cv=5, n_jobs=10, iid=True)
                    grid.fit(X_train, y_train.flatten())
                    clf = grid.best_estimator_
                else:
                    if self.estimator_str == 'RF':
                        clf = RandomForestRegressor(n_estimators=self.RF_no_of_estimators, random_state=k)
                    else: 
                        clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=self.RF_no_of_estimators, random_state=k)
                    
                    # clf = estimator(n_estimators=300, n_jobs=5)
                    # clf = estimator(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=k)
                    clf.fit(X_train, y_train.flatten())
                
                
            
                y_predicted = clf.predict(X_test)
                # y_predicted_combined = clf.predict(X_test_combined)
        
                # importances = clf.feature_importances_
                # print(importances)
                # df_feat_importance.loc[df_feat_importance.shape[0],:] = [k, fold] + list(importances)
                # print(df_feat_importance)
                # sys.exit()
                # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                #             axis=0)
                # indices = np.argsort(importances)[::-1]

                # Print the feature ranking
                # print("Feature ranking:")

                # for f in range(X.shape[1]):
                #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                # return
                # for i in range(y_test.shape[0]):
                #     df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                #     df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                #     # df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                # print(y_test.flatten())
                # print(y_predicted)
                df_fold_predictions = pd.DataFrame(columns=['single_feat', 'patient_id', 'ground_truth', 'predicted'])
                for y_ground, y_pred in zip(y_test.flatten(), y_predicted):
                    # df_predictions.loc[df_predictions.shape[0],:] = [patient, scalerY.inverse_transform([y_ground])[0], scalerY.inverse_transform([y_pred])[0]]
                    # df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient, scalerY.inverse_transform([y_ground])[0], scalerY.inverse_transform([y_pred])[0]]
                    df_predictions.loc[df_predictions.shape[0],:] = [single_feat, patient, y_ground, y_pred]
                    df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [single_feat, patient, y_ground, y_pred]
                
                # for y_ground, y_pred in zip(Y_test_combined.flatten(), y_predicted_combined):
                #     # df_predictions_preIncluded.loc[df_predictions_preIncluded.shape[0],:] = [patient, scalerY.inverse_transform([y_ground])[0], scalerY.inverse_transform([y_pred])[0]]
                #     df_predictions_preIncluded.loc[df_predictions_preIncluded.shape[0],:] = [patient, y_ground, y_pred]
                
                fold += 1
                
                rho, pval = spearmanr(df_fold_predictions['ground_truth'], df_fold_predictions['predicted'])
                # print('spearman rho = {} pval = {}'.format(rho, pval))
                rho, pval = pearsonr(df_fold_predictions['ground_truth'], df_fold_predictions['predicted'])
                # print('pearson rho = {} pval = {}'.format(rho, pval))
                
        
            
            df_predictions.sort_values(by='ground_truth', axis=0, ascending=True, inplace=True)
            df_predictions.reset_index(drop=True, inplace=True)
            df_predictions_filename = self.single_feat_actigraph_predictions_save_folder+'{}/patient_{}/'.format(single_feat, patient)
        
            os.makedirs(df_predictions_filename, exist_ok=True)
            # os.makedirs(df_predictions_preIncluded_filename, exist_ok=True)
            df_predictions.to_csv(df_predictions_filename+'prediction_rep_{}.csv'.format(k), header=True, index=False)
            # df_predictions_preIncluded.to_csv(df_predictions_preIncluded_filename+'prediction_rep_{}.csv'.format(k), header=True, index=False)

            rho, pval = spearmanr(df_predictions['ground_truth'], df_predictions['predicted'])
            # print('spearman rho = {} pval = {}'.format(rho, pval))
            pearsonr_rho, pearson_pval = pearsonr(df_predictions['ground_truth'], df_predictions['predicted'])
            # print('pearson rho = {} pval = {}'.format(pearsonr_rho, pearson_pval))
            output_array.append([single_feat, patient, rho, pval, pearsonr_rho, pearson_pval])
        #     df_repeatedkfold_predictions.loc[df_repeatedkfold_predictions.shape[0],:] = [k, rho, pval]
        # print(df_repeatedkfold_predictions)   
        # df_repeatedkfold_predictions.to_csv(df_predictions_filename+'/prediction_correlations.csv', header=True, index=False) 
        # df_feat_importance.to_csv(df_predictions_filename+'feature_importances.csv', header=True, index=False) 
        
        
        # r2 = r2_score(df_predictions['baseline'], df_predictions['predicted'])
        # print('r2 = ', r2)



    def predict_single_feature_personalized_activity_model_parallel_iterations_k_fold_per_patient(self):
        # for the sake of comparison
        time_ = time.time()
        
        df_original = pd.read_csv(self.actigraph_filename)
        print(df_original)
        output_cols = ['single_feat', 'patient_id', 'rho', 'pval','pearson rho', 'pearson pval']
        manager = multiprocessing.Manager()
        output_array = manager.list()
        features = df_original.columns.values[1:-1]

        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]

        numberOfThreads = 1
        jobs = []
        #v1 :15 done
        #v2 15:25 done
        #v3 25:40 done
        #v4 40:50 running, top
        #v5 50:50 running, down
        for single_feat in features:
            print('single feat = {}'.format(single_feat))
            for patient in patients:
                
                # print('------patient = {}'.format(patient))
                    
                if patient < 10:
                    patient_id = 'VAL0{}.agd'.format(patient)
                else:
                    patient_id = 'VAL{}.agd'.format(patient)

                # df = df_original[(df_original['Day'] > 0) & (df_original['Filename'] == patient_id)]
                df = df_original[(df_original['Day'] > 0) & (df_original['Day'] < 41) & (df_original['Filename'] == patient_id)]
                
                # df_pre = df_original[(df_original['Day'] < 1) & (df_original['Filename'] == patient_id)]
                # df_pre = df_original[(df_original['Day'] < 1) & (df_original['Day'] > -6) & (df_original['Filename'] == patient_id)]
                # df = df_original[(df_original['Day'] > 0)]
                # print(df.shape[0])
                
                df.drop(['Filename'], axis=1,inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.dropna(inplace=True)

                label = 'Day'

                X = df[single_feat].values
                Y = df[label].values
                # X_pre = df_pre[features].values
                
                # Y_pre = df_pre[label].values
                # X_pre = []
                # Y_pre = []

                p = multiprocessing.Process(target=self.repeated_k_folds_per_patient_single_feature, args=(output_array,single_feat,patient,X,Y,))
                jobs.append(p)
        for i in self.chunks(jobs,numberOfThreads):
            for j in i:
                j.start()
            for j in i:
                j.join()
    

        output_df = pd.DataFrame(np.array(output_array), columns=output_cols)
        print(output_df.shape[0])
        
        output_df.to_csv(self.single_feat_actigraph_predictions_save_folder+'predictions.csv', index=False, header=True)
        print('run time = {}'.format(time.time()-time_))

    def predict_single_feature_personalized_activity_model_boxplot_median_k_fold_per_patient(self):

        # box plot for R and p values
    
        df_actigraph = pd.read_csv(self.actigraph_filename)
        features = df_actigraph.columns.values[1:-1]

        pred_filename = self.single_feat_actigraph_predictions_save_folder+'predictions.csv'
        df_original = pd.read_csv(pred_filename)
        print(df_original)
        df_original['rho'] = df_original['rho'].clip(lower=0)
        print(df_original)
        df_original['log10_pval'] = -1*np.log10(df_original['pval'].values)
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]

        for single_feat in features:
            print('single_feat = {}'.format(single_feat))
            
            
            plot_df = pd.DataFrame(columns=['patient_id', 'rho', 'log10_pval'])
            for patient in patients:
                
                print('patient = {}'.format(patient))
                    
                # if patient < 10:
                #     patient_id = 'VAL0{}.agd'.format(patient)
                # else:
                #     patient_id = 'VAL{}.agd'.format(patient)

                df = df_original[df_original['single_feat'] == single_feat]
                df = df[df['patient_id'] == patient]

                print(df)
                
                plot_df.loc[plot_df.shape[0],:] = [patient, df['rho'].median(), df['log10_pval'].median()]
                


            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(5,5), constrained_layout=True) 
            # df.boxplot(column=['rho'], ax=ax[0], showfliers=False)
            # df.boxplot(column=['log10_pval'], ax=ax[1], showfliers=False)
            sns.boxplot(y="rho", data=plot_df, showfliers = False, ax=ax[0], color="royalblue")
            sns.swarmplot(y="rho", data=plot_df, color=".25", ax=ax[0])
            sns.boxplot(y="log10_pval", data=plot_df, showfliers = False, ax=ax[1], color="limegreen")
            sns.swarmplot(y="log10_pval", data=plot_df, color=".25", ax=ax[1])
            plt.tight_layout()
            # fig.suptitle(foldername, fontsize=8, verticalalignment='bottom')
            ax[0].set_ylim([0,1])
            ax[1].set_ylim([0.1,20])
            ax[1].axhline(y=-1*np.log10(0.05), linewidth=4, color='r')

            plt.savefig(self.single_feat_actigraph_predictions_save_folder+single_feat+'/boxplot_medians.jpg', format='jpg', pad_inches=1)
            
            # plt.show() 
            plot_df.to_csv(self.single_feat_actigraph_predictions_save_folder+single_feat+'/median_predictions.csv', header=True, index=False) 
            
    def predict_single_feat_personalized_activity_model_rmse_patient_median(self):
        # get the rmse for each patient
        df_actigraph = pd.read_csv(self.actigraph_filename)
        features = df_actigraph.columns.values[1:-1]

        for single_feat in features:
            print('single_feat = {}'.format(single_feat))
            df_rmse = pd.DataFrame(columns=['patient_id', 'RMSE', 'MAE'])
            excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
            patients = [i for i in range(1,76) if i not in excluded_file_no]
            for patient in patients:
                
                print('patient = {}'.format(patient))
                df_patient = pd.DataFrame(columns=['ground_truth', 'predicted'])
                for rep in range(1,self.no_of_reps+1):
                    # if patient < 10:
                    #     patient_id = 'VAL0{}.agd'.format(patient)
                    # else:
                    #     patient_id = 'VAL{}.agd'.format(patient)
                    pred_filename = self.single_feat_actigraph_predictions_save_folder+single_feat+'/patient_{}/prediction_rep_{}.csv'.format(patient, rep)
                    df_rep =pd.read_csv(pred_filename)
                    assert df_rep.iloc[0,0] == single_feat, 'feat names dont match'
                    assert df_rep.iloc[0,1] == patient, 'patient ids dont match'
                    # print(df_rep)
                    df_rep.drop('single_feat', axis=1, inplace=True)
                    df_rep.drop('patient_id', axis=1, inplace=True)
                    df_patient = pd.concat([df_patient, df_rep], axis=0)
                df_patient.reset_index(drop=True, inplace=True)
                print(df_patient)
                rmse = mean_squared_error(df_patient['ground_truth'],df_patient['predicted'], squared=False)
                mae = mean_absolute_error(df_patient['ground_truth'],df_patient['predicted'])
                df_rmse = pd.concat([df_rmse, pd.DataFrame([[patient, rmse, mae]], columns=['patient_id', 'RMSE', 'MAE'])], axis=0)
            print(df_rmse)
            df_rmse.to_csv(self.single_feat_actigraph_predictions_save_folder+single_feat+'/predictions_rmse.csv', index=False, header=True)
            
    def plot_predict_single_feat_vs_multi_personalized_activity_model_rmse_patient_median(self):
        #plot model performance comparison in terms of RMSE and MAE for our model vs single_feature models
        
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(15,10)) 
        
        df_plot_RMSE = pd.DataFrame()
        df_plot_MAE = pd.DataFrame()

        df = pd.read_csv(self.actigraph_predictions_save_folder+'predictions_rmse.csv')
        print(df)
        df_plot_RMSE['Multivariate Clock'] = df['RMSE']
        df_plot_MAE['Multivariate Clock'] = df['MAE']


        df_actigraph = pd.read_csv(self.actigraph_filename)
        features = df_actigraph.columns.values[1:-1]

        for single_feat in features:
            df = pd.read_csv(self.single_feat_actigraph_predictions_save_folder+single_feat+'/predictions_rmse.csv')
            df_plot_RMSE[single_feat] = df['RMSE']
            df_plot_MAE[single_feat] = df['MAE']

            
        df_plot_RMSE.plot(kind='box', ax=ax[0], showfliers=False)
        # ax[0].set_ylim([0,1])
        
        df_plot_MAE.plot(kind='box', ax=ax[1], showfliers=False)
        # ax[1].set_ylim([0,1])
        plt.tight_layout()
        ax[0].tick_params(labelrotation=90)
        ax[1].tick_params(labelrotation=90)
        plt.setp(ax[0], ylabel='RMSE')
        plt.setp(ax[1], ylabel='MAE')
        fig.tight_layout(h_pad=3)
        plt.savefig(self.single_feat_actigraph_predictions_save_folder+'comparison_boxplot_RMSE.pdf', format='pdf')

        


    def repeated_k_folds_per_patient(self, output_array,patient,X,Y,X_pre,Y_pre,features,model_selection=False, param_grid = [{'n_estimators': [50,100,200], 'loss': ['linear', 'square', 'exponential']}]):
        print('**************************** repeated_k_folds_per_patient')

       
        df_feat_importance_col = ['iteration', 'fold']
        df_feat_importance_col.extend(features)
        df_feat_importance = pd.DataFrame(columns=df_feat_importance_col)
        
        # params for rf regressor param_grid = [{'n_estimators': [50,100,200], 'max_depth': [2, 5, 10, None]}]
        for k in range(1, self.no_of_reps+1):
            df_predictions = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted'])
            df_predictions_preIncluded = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted'])
            kf = KFold(n_splits=self.no_of_folds, shuffle=True, random_state=k)
            
            fold = 0

            for train_index, test_index in kf.split(X):
                print('rep = {}, fold = {}'.format(k,fold))
            
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                
                X_test_combined = np.concatenate((X_test, X_pre), axis=0)
                Y_test_combined = np.concatenate((y_test, Y_pre), axis=0)
                
                # scalerX = StandardScaler()
                # scalerX.fit(X_train)
                # X_train = scalerX.transform(X_train)
                # X_test = scalerX.transform(X_test)
                # X_test_combined = scalerX.transform(X_test_combined)

                # scalerY = StandardScaler()
                # scalerY.fit(y_train.reshape(-1, 1))
                # y_train = scalerY.transform(y_train.reshape(-1, 1))
                # y_test = scalerY.transform(y_test.reshape(-1, 1))
                # Y_test_combined = scalerY.transform(Y_test_combined.reshape(-1, 1))

                max_iter = 1000000
                if model_selection:
                    # grid = GridSearchCV(estimator=estimator(random_state=k), param_grid=param_grid, cv=5, n_jobs=10, iid=True)
                    grid = GridSearchCV(estimator=estimator(DecisionTreeRegressor(max_depth=4), random_state=k), param_grid=param_grid, cv=5, n_jobs=10, iid=True)
                    grid.fit(X_train, y_train.flatten())
                    clf = grid.best_estimator_
                else:
                    if self.estimator_str == 'RF':
                        clf = RandomForestRegressor(n_estimators=self.RF_no_of_estimators, random_state=k)
                    else: 
                        clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=self.RF_no_of_estimators, random_state=k)
                    
                    # clf = estimator(n_estimators=300, n_jobs=5)
                    # clf = estimator(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=k)
                    clf.fit(X_train, y_train.flatten())
                
                
            
                y_predicted = clf.predict(X_test)
                y_predicted_combined = clf.predict(X_test_combined)
        
                importances = clf.feature_importances_
                # print(importances)
                df_feat_importance.loc[df_feat_importance.shape[0],:] = [k, fold] + list(importances)
                # print(df_feat_importance)
                # sys.exit()
                # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                #             axis=0)
                # indices = np.argsort(importances)[::-1]

                # Print the feature ranking
                # print("Feature ranking:")

                # for f in range(X.shape[1]):
                #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                # return
                # for i in range(y_test.shape[0]):
                #     df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                #     df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                #     # df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                # print(y_test.flatten())
                # print(y_predicted)
                df_fold_predictions = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted'])
                for y_ground, y_pred in zip(y_test.flatten(), y_predicted):
                    # df_predictions.loc[df_predictions.shape[0],:] = [patient, scalerY.inverse_transform([y_ground])[0], scalerY.inverse_transform([y_pred])[0]]
                    # df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient, scalerY.inverse_transform([y_ground])[0], scalerY.inverse_transform([y_pred])[0]]
                    df_predictions.loc[df_predictions.shape[0],:] = [patient, y_ground, y_pred]
                    df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient, y_ground, y_pred]
                
                for y_ground, y_pred in zip(Y_test_combined.flatten(), y_predicted_combined):
                    # df_predictions_preIncluded.loc[df_predictions_preIncluded.shape[0],:] = [patient, scalerY.inverse_transform([y_ground])[0], scalerY.inverse_transform([y_pred])[0]]
                    df_predictions_preIncluded.loc[df_predictions_preIncluded.shape[0],:] = [patient, y_ground, y_pred]
                
                fold += 1
                
                rho, pval = spearmanr(df_fold_predictions['ground_truth'], df_fold_predictions['predicted'])
                print('spearman rho = {} pval = {}'.format(rho, pval))
                rho, pval = pearsonr(df_fold_predictions['ground_truth'], df_fold_predictions['predicted'])
                print('pearson rho = {} pval = {}'.format(rho, pval))
                
        
            
            df_predictions.sort_values(by='ground_truth', axis=0, ascending=True, inplace=True)
            df_predictions.reset_index(drop=True, inplace=True)
            df_predictions_filename = self.actigraph_predictions_save_folder+'patient_{}/'.format(patient)
            df_predictions_preIncluded_filename = self.actigraph_predictions_save_folder+'patient_{}_preIncluded/'.format(patient)
        
            os.makedirs(df_predictions_filename, exist_ok=True)
            os.makedirs(df_predictions_preIncluded_filename, exist_ok=True)
            df_predictions.to_csv(df_predictions_filename+'prediction_rep_{}.csv'.format(k), header=True, index=False)
            df_predictions_preIncluded.to_csv(df_predictions_preIncluded_filename+'prediction_rep_{}.csv'.format(k), header=True, index=False)

            rho, pval = spearmanr(df_predictions['ground_truth'], df_predictions['predicted'])
            print('spearman rho = {} pval = {}'.format(rho, pval))
            pearsonr_rho, pearson_pval = pearsonr(df_predictions['ground_truth'], df_predictions['predicted'])
            print('pearson rho = {} pval = {}'.format(pearsonr_rho, pearson_pval))
            output_array.append([patient, rho, pval, pearsonr_rho, pearson_pval])
        #     df_repeatedkfold_predictions.loc[df_repeatedkfold_predictions.shape[0],:] = [k, rho, pval]
        # print(df_repeatedkfold_predictions)   
        # df_repeatedkfold_predictions.to_csv(df_predictions_filename+'/prediction_correlations.csv', header=True, index=False) 
        df_feat_importance.to_csv(df_predictions_filename+'feature_importances.csv', header=True, index=False) 
        
        
        # r2 = r2_score(df_predictions['baseline'], df_predictions['predicted'])
        # print('r2 = ', r2)


        

    def predict_personalized_activity_model_parallel_iterations_k_fold_per_patient(self):
        time_ = time.time()
        
        # pred_filename = self.actigraph_predictions_save_folder+'predictions.csv'
        
        df_original = pd.read_csv(self.actigraph_filename)
        print(df_original)
        output_cols = ['patient_id', 'rho', 'pval','pearson rho', 'pearson pval']
        manager = multiprocessing.Manager()
        output_array = manager.list()
        

        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]

        numberOfThreads = 5
        jobs = []
        for patient in patients:
            
            print('patient = {}'.format(patient))
                
            if patient < 10:
                patient_id = 'VAL0{}.agd'.format(patient)
            else:
                patient_id = 'VAL{}.agd'.format(patient)

            # df = df_original[(df_original['Day'] > 0) & (df_original['Filename'] == patient_id)]
            df = df_original[(df_original['Day'] > 0) & (df_original['Day'] < 41) & (df_original['Filename'] == patient_id)]
            
            # df_pre = df_original[(df_original['Day'] < 1) & (df_original['Filename'] == patient_id)]
            df_pre = df_original[(df_original['Day'] < 1) & (df_original['Day'] > -6) & (df_original['Filename'] == patient_id)]
            # df = df_original[(df_original['Day'] > 0)]
            # print(df.shape[0])
            
            df.drop(['Filename'], axis=1,inplace=True)
            df.reset_index(inplace=True)
            df.dropna(inplace=True)

            label = 'Day'
            features = df.columns.values[1:-1] #remove index and label columns

            X = df[features].values
            Y = df[label].values
            X_pre = df_pre[features].values
            
            Y_pre = df_pre[label].values
            # X_pre = []
            # Y_pre = []

            p = multiprocessing.Process(target=self.repeated_k_folds_per_patient, args=(output_array,patient,X,Y,X_pre,Y_pre,features,))
            jobs.append(p)
        for i in self.chunks(jobs,numberOfThreads):
            for j in i:
                j.start()
            for j in i:
                j.join()
    

        output_df = pd.DataFrame(np.array(output_array), columns=output_cols)
        print(output_df.shape[0])
        
        output_df.to_csv(self.actigraph_predictions_save_folder+'predictions.csv', index=False, header=True)
        print('run time = {}'.format(time.time()-time_))
        
        


    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def predict_personalized_activity_model_boxplot_median_k_fold_per_patient(self):
        pred_filename = self.actigraph_predictions_save_folder+'predictions.csv'
        df_original =pd.read_csv(pred_filename)
        print(df_original)
        # df_original['rho'] = df_original['rho'].clip_lower(0)
        df_original['rho'] = df_original['rho'].clip(lower=0)
        print(df_original)
        df_original['log10_pval'] = -1*np.log10(df_original['pval'].values)
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]
        plot_df = pd.DataFrame(columns=['patient_id', 'rho', 'log10_pval'])
        for patient in patients:
            
            print('patient = {}'.format(patient))
                
            # if patient < 10:
            #     patient_id = 'VAL0{}.agd'.format(patient)
            # else:
            #     patient_id = 'VAL{}.agd'.format(patient)

            df = df_original[df_original['patient_id'] == patient]

            print(df)
            
            plot_df.loc[plot_df.shape[0],:] = [patient, df['rho'].median(), df['log10_pval'].median()]
            


        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(5,5), constrained_layout=True) 
        # df.boxplot(column=['rho'], ax=ax[0], showfliers=False)
        # df.boxplot(column=['log10_pval'], ax=ax[1], showfliers=False)
        sns.boxplot(y="rho", data=plot_df, showfliers = False, ax=ax[0], color="royalblue")
        sns.swarmplot(y="rho", data=plot_df, color=".25", ax=ax[0])
        sns.boxplot(y="log10_pval", data=plot_df, showfliers = False, ax=ax[1], color="limegreen")
        sns.swarmplot(y="log10_pval", data=plot_df, color=".25", ax=ax[1])
        plt.tight_layout()
        # fig.suptitle(foldername, fontsize=8, verticalalignment='bottom')
        ax[0].set_ylim([0,1])
        ax[1].set_ylim([0.1,20])
        ax[1].axhline(y=-1*np.log10(0.05), linewidth=4, color='r')

        plt.savefig(self.actigraph_predictions_save_folder+'boxplot_medians.jpg', format='jpg', pad_inches=1)
        
        # plt.show() 
        plot_df.to_csv(self.actigraph_predictions_save_folder+'median_predictions.csv', header=True, index=False) 


    def predict_personalized_activity_model_boxplot_k_fold_each_patient(self):
        print('### we are here')
        pred_filename = self.actigraph_predictions_save_folder+'predictions.csv'
        df_original =pd.read_csv(pred_filename)
        print(df_original)
        # df_original['rho'] = df_original['rho'].clip_lower(0)
        df_original['rho'] = df_original['rho'].clip(lower=0)
        print(df_original)
        df_original['log10_pval'] = -1*np.log10(df_original['pval'].values)
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]
        plot_df_rho = [pd.DataFrame() for _ in range(5)]
        plot_df_pval = [pd.DataFrame() for _ in range(5)]
        fig_rho, ax_rho = plt.subplots(nrows=5, ncols=1, figsize=(14,10)) 
        fig_pval, ax_pval = plt.subplots(nrows=5, ncols=1, figsize=(14,10)) 
        for idx, patient in enumerate(patients):
            
            print('patient = {}'.format(patient))
                
            # if patient < 10:
            #     patient_id = 'VAL0{}.agd'.format(patient)
            # else:
            #     patient_id = 'VAL{}.agd'.format(patient)
        
            df = df_original[df_original['patient_id'] == patient]
            df.reset_index(drop=True, inplace=True)
            df['log10_pval'] = -1*np.log10(df['pval'])
            # print(df)
            # print(plot_df_rho[idx//10])
            # print(plot_df_pval[idx//4])
            this_plot_rho = plot_df_rho[idx//10]
            this_plot_pval = plot_df_pval[idx//10]
            # this_plot['patient_id'] = df['patient_id']
            this_plot_rho['Patient {}'.format(int(idx))] = df['rho']
            # print(this_plot_rho)


            this_plot_pval['Patient {}'.format(int(idx))] = df['log10_pval']
            # print(this_plot_pval)
            # print(plot_df_rho[idx//10])
            # print(plot_df_pval[idx//4])



            # ax1[idx//(len(patients)//3),idx%(len(patients)//3)].boxplot(df['rho'])
            # ax2[idx//(len(patients)//3),idx%(len(patients)//3)].boxplot(df['log10_pval'])
            # plt.show()
            # sys.exit()
            # plot_df.loc[plot_df.shape[0],:] = [patient, df['rho'].median(), df['log10_pval'].median()]
            
        for i in range(5):
            # print(plot_df_rho[i])
            # print(plot_df_pval[i])
        
            plot_df_rho[i].boxplot(ax=ax_rho[i])
            plot_df_pval[i].boxplot(ax=ax_pval[i])
        # plt.show()
        plt.savefig(self.actigraph_predictions_save_folder+'temp_copy.jpg', format='jpg', pad_inches=1)
        
    def predict_personalized_activity_model_rmse_patient_median(self):
        # get the rmse for each patient
        df_rmse = pd.DataFrame(columns=['patient_id', 'RMSE', 'MAE'])
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]
        for patient in patients:
            
            print('patient = {}'.format(patient))
            df_patient = pd.DataFrame(columns=['ground_truth', 'predicted'])
            for rep in range(1,self.no_of_reps+1):
                # if patient < 10:
                #     patient_id = 'VAL0{}.agd'.format(patient)
                # else:
                #     patient_id = 'VAL{}.agd'.format(patient)
                pred_filename = self.actigraph_predictions_save_folder+'patient_{}/prediction_rep_{}.csv'.format(patient, rep)
                df_rep =pd.read_csv(pred_filename)
                assert df_rep.iloc[0,0] == patient, 'patient ids dont match'
                # print(df_rep)
                df_rep.drop('patient_id', axis=1, inplace=True)
                df_patient = pd.concat([df_patient, df_rep], axis=0)
            df_patient.reset_index(drop=True, inplace=True)
            print(df_patient)
            rmse = mean_squared_error(df_patient['ground_truth'],df_patient['predicted'], squared=False)
            mae = mean_absolute_error(df_patient['ground_truth'],df_patient['predicted'])
            df_rmse = pd.concat([df_rmse, pd.DataFrame([[patient, rmse, mae]], columns=['patient_id', 'RMSE', 'MAE'])], axis=0)
        print(df_rmse)
        df_rmse.to_csv(self.actigraph_predictions_save_folder+'predictions_rmse.csv', index=False, header=True)
            
            
    
    def plot_personalized_activity_model_lineplot_median_k_fold_per_patient(self):
        
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]
        confidence_df_col = ['patient_id']
        confidence_df_col.extend([d for d in range(-5,41) if d!=0])
        # confidence_df_col.extend([d for d in range(-5,43) if d!=0])
        confidence_df = pd.DataFrame(columns=confidence_df_col)
        # print(patients)
        
        x1 = np.linspace(-5, 46, 500)
        # x1 = np.linspace(-5, 48, 500)
        for patient in patients: 
            patient_filename = self.actigraph_predictions_save_folder+'median_predictions/Predictions_patient_{}.csv'.format(patient)
            df_patient_pred = pd.read_csv(patient_filename)
            df_patient_pred = df_patient_pred[df_patient_pred['ground_truth']>=-5]
            row = [patient]
            row.extend(list(df_patient_pred['predicted_value'].values))
            confidence_df.loc[confidence_df.shape[0]] = row
            print(df_patient_pred)
        print(confidence_df)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5)) 
        for i in range(confidence_df.shape[0]):
            x = []
            y = []
            for idx, (x0, y0) in enumerate(zip(confidence_df.columns.values[1:],confidence_df.iloc[i,1:].values)):
                if idx<=6 or idx%4==0 or idx==len(confidence_df.iloc[i,1:].values)-1:
                    x.append(x0)
                    y.append(y0)
            tck1 = interpolate.splrep(x,y)
            y1 = interpolate.splev(x1, tck1)
            ax.plot(x1,y1,color='darkgrey', linewidth=0.5, alpha=0.5)
        # confidence=0.95
        confidence=0.9
        # confidence=0.99
        average = []
        upper = []
        lower = []
        for day in confidence_df.columns.values[1:]:
            a = 1.0 * np.array(confidence_df[day].values)
            n = len(a)
            m, se = np.mean(a), stats.sem(a)
            h = se * stats.t.ppf((1 + confidence) / 2., n-1)
            average.append(m)
            upper.append(m+h)
            lower.append(m-h)
        tck2 = interpolate.splrep(confidence_df.columns.values[1:],average)
        y2 = interpolate.splev(x1, tck2)
        tck3 = interpolate.splrep(confidence_df.columns.values[1:],upper)
        y3 = interpolate.splev(x1, tck3)
        tck4 = interpolate.splrep(confidence_df.columns.values[1:],lower)
        y4 = interpolate.splev(x1, tck4)  
        ax.plot(x1,y2,color='red', linewidth=2)
        # ax.plot(x1,y3,color='green', linewidth=1)
        # ax.plot(x1,y4,color='blue', linewidth=1)
        ax.fill_between(x1, y3, y4, color='red', alpha=0.5)

        ax.set_xlim([-5,40])
        # ax.set_xlim([-5,42])
        ax.set_ylim([0,40])
        # ax.set_ylim([0,42])
        plt.xticks([-5,0,15,30,40])
        # plt.xticks([-5,0,15,30])
        plt.title('avg spline (actual vs predicted day since surgery) over all patient models, {} CI'.format(confidence), fontsize=7)
        # plt.show()
        plt.savefig(self.actigraph_predictions_save_folder+'Preds_lineplot_only_avg_spline_CI_{}_medians.pdf'.format(confidence), format='pdf', dpi=600)
        

    def plot_clustered_personalized_activity_model_lineplot_median_k_fold_per_patient(self, CI, n_c):


        # a = np.array([1, 2, 3, 4, 4])
        # print(np.interp(a, (0, 3), (-1, +1)))
        # sys.exit()

        df_med_preds = pd.read_csv(self.recovery_feature_predictions_save_folder+'median_predictions.csv')
        df_med_preds.columns = ['patient_id', self.target_recovery_feature, 'predicted']
        kmeans = KMeans(n_clusters=n_c, random_state=42)
        kmeans.fit(df_med_preds[self.target_recovery_feature].values.reshape(-1,1))
        print(kmeans.labels_)
        df_med_preds['cluster_labels'] = kmeans.labels_
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        cm_subsection = np.linspace(0, 1, n_c) 
        colors = [cm.jet(x) for x in cm_subsection]
        node_colors = []
        for i in range(df_med_preds.shape[0]):
            node_colors.append(colors[df_med_preds.loc[i,'cluster_labels']])
        df_med_preds.plot(kind='scatter', x=self.target_recovery_feature, y='predicted', color=node_colors, ax=ax)
        plt.title('Basline prediction model ground truth value vs predicted by cytof. colored by cluster id'.format(n_c), fontsize=7)
        # ax.set_ylim([5,35])
        # ax.set_xlim([5,40])
        # plt.show()
        os.makedirs(self.recovery_feature_predictions_save_folder+'clustered_surrogates/{}/'.format(n_c), exist_ok=True)
        plt.savefig(self.recovery_feature_predictions_save_folder+'clustered_surrogates/{}/scatter_plot_predictions_{}_surrogate_colored_by_clusters.pdf'.format(n_c, self.target_recovery_feature), format='pdf', dpi=600)
        plt.close()
        df_med_preds.to_csv(self.recovery_feature_predictions_save_folder+'clustered_surrogates/{0:}/median_predictions_gt_{0:}_clusters.csv'.format(n_c), header=True, index=False)
        
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]
        
        print(patients)

        min_pred = np.inf
        max_pred = -1*np.inf
        for patient in patients:
            patient_filename = self.actigraph_predictions_save_folder+'median_predictions/Predictions_patient_{}.csv'.format(patient)
            df_patient_pred = pd.read_csv(patient_filename)
            df_patient_pred = df_patient_pred[df_patient_pred['ground_truth']>=-5]
            # df_patient_pred['predicted_value_scaled'] = np.interp(df_patient_pred['predicted_value'], (0, 39), (0, 100))
            # print(df_patient_pred)
          
            # print(df_patient_pred['predicted_value'].max())
            # print(df_patient_pred['predicted_value'].min())
            max_pred = max(max_pred, df_patient_pred['predicted_value'].max())
            min_pred = min(min_pred, df_patient_pred['predicted_value'].min())
        print(max_pred)
        print(min_pred)
 

        patient_id_clusters = []
        for c_id in range(n_c):
            c_id_patients = []
            for p in patients:
                if df_med_preds[df_med_preds['patient_id']==p]['cluster_labels'].values[0] == c_id:
                    c_id_patients.append(p)
            patient_id_clusters.append(c_id_patients)
        print(patient_id_clusters)
        
        cm_lines = np.linspace(0, 1, n_c) 
        colors = [cm.jet(x) for x in cm_lines]
        
        x1 = np.linspace(-5, 42, 500)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5)) 
        for c_idx, c_id_patients in enumerate(patient_id_clusters):

            # min_pred = np.inf
            # max_pred = -1*np.inf
            # for patient in c_id_patients:
            #     patient_filename = self.actigraph_predictions_save_folder+'median_predictions/Predictions_patient_{}.csv'.format(patient)
            #     df_patient_pred = pd.read_csv(patient_filename)
            #     df_patient_pred = df_patient_pred[df_patient_pred['ground_truth']>=-5]
            #     # df_patient_pred['predicted_value_scaled'] = np.interp(df_patient_pred['predicted_value'], (0, 39), (0, 100))
            #     # print(df_patient_pred)
            
            #     # print(df_patient_pred['predicted_value'].max())
            #     # print(df_patient_pred['predicted_value'].min())
            #     max_pred = max(max_pred, df_patient_pred['predicted_value'].max())
            #     min_pred = min(min_pred, df_patient_pred['predicted_value'].min())
            # print(max_pred)
            # print(min_pred)

            print(c_id_patients)
            confidence_df_col = ['patient_id']
            confidence_df_col.extend([d for d in range(-5,41) if d!=0])
            # confidence_df_col.extend([d for d in range(-5,43) if d!=0])
            confidence_df = pd.DataFrame(columns=confidence_df_col)
            for patient in c_id_patients: 
                patient_filename = self.actigraph_predictions_save_folder+'median_predictions/Predictions_patient_{}.csv'.format(patient)
                df_patient_pred = pd.read_csv(patient_filename)
                df_patient_pred = df_patient_pred[df_patient_pred['ground_truth']>=-5]
                # df_patient_pred['predicted_value_scaled'] = np.interp(df_patient_pred['predicted_value'], (0, max_pred), (0, 100))
                # print(df_patient_pred)
                row = [patient]
                # row.extend(list(df_patient_pred['predicted_value_scaled'].values))
                row.extend(list(df_patient_pred['predicted_value'].values))
                confidence_df.loc[confidence_df.shape[0]] = row
                print(df_patient_pred)
            print(confidence_df)
            
            for i in range(confidence_df.shape[0]):
                x = []
                y = []
                for idx, (x0, y0) in enumerate(zip(confidence_df.columns.values[1:],confidence_df.iloc[i,1:].values)):
                    if idx<=6 or idx%4==0 or idx==len(confidence_df.iloc[i,1:].values)-1:
                        x.append(x0)
                        y.append(y0)
                tck1 = interpolate.splrep(x,y)
                y1 = interpolate.splev(x1, tck1)
                # ax.plot(x1,y1,color='darkgrey', linewidth=0.5, alpha=0.5)
            
            average = []
            upper = []
            lower = []
            for day in confidence_df.columns.values[1:]:
                a = 1.0 * np.array(confidence_df[day].values)
                n = len(a)
                m, se = np.mean(a), stats.sem(a)
                h = se * stats.t.ppf((1 + CI) / 2., n-1)
                average.append(m)
                upper.append(m+h)
                lower.append(m-h)
            pre_op_avg = np.mean(average[:5])
            # print(average[:5])
            # print(np.mean(average[:5]))
            # print(average)
            # print([item-np.mean(average[:5]) for item in upper])
            # max_pred = np.max(upper[:5])
            # max_pred = max_pred - np.mean(average[:5])
            for idx in range(len(upper)):
                upper[idx] = 100/pre_op_avg*upper[idx]
            # upper = [lambda:(100/(np.mean(average[:5])*item))(item) for item in upper]
            # upper = np.interp(upper, (0-np.mean(average[:5]), max_pred), (0, 100))
        
            # print([item-np.mean(average[:5]) for item in lower])
            # lower = [item-np.mean(average[:5]) for item in lower]
            for idx in range(len(lower)):
                lower[idx] = 100/pre_op_avg*lower[idx]
            # lower = [lambda:(100/(np.mean(average[:5])*item))(item) for item in lower]
            # lower = np.interp(lower, (0-np.mean(average[:5]), max_pred), (0, 100))
            # print(lower)
            
            # print([item-np.mean(average[:5]) for item in average])
            # average = [item-np.mean(average[:5]) for item in average]
            for idx in range(len(average)):
                average[idx] = 100/pre_op_avg*average[idx]
            # average = [lambda:(100/(np.mean(average[:5])*item))(item) for item in average]
            # average = np.interp(average, (0-np.mean(average[:5]), max_pred), (0, 100))
            
            # sys.exit()
            tck2 = interpolate.splrep(confidence_df.columns.values[1:],average)
            y2 = interpolate.splev(x1, tck2)
            tck3 = interpolate.splrep(confidence_df.columns.values[1:],upper)
            y3 = interpolate.splev(x1, tck3)
            tck4 = interpolate.splrep(confidence_df.columns.values[1:],lower)
            y4 = interpolate.splev(x1, tck4)  
            ax.plot(x1,y2,color=colors[c_idx], linewidth=2)
            # ax.plot(x1,y3,color='green', linewidth=1)
            # ax.plot(x1,y4,color='blue', linewidth=1)
            ax.fill_between(x1, y3, y4, color=colors[c_idx], alpha=0.3)

        ax.set_xlim([-5,40])
        # ax.set_xlim([-5,42])
        ax.set_ylim([0,140])
        # ax.set_ylim([0,42])
        plt.xticks([-5,0,15,30])
        plt.yticks([20,40,60,80,100])
        legend_elements = []
        for c_id in range(n_c):
            legend_elements.append(mpatches.Patch(facecolor=colors[c_id],label='cluster_{}'.format(c_id)))
        plt.legend(handles=legend_elements,
               scatterpoints=1,
               loc='best',
               ncol=4,
               fontsize=4)   
        plt.title('avg spline {} clusters (actual vs predicted day since surgery) over all patient models, {} CI'.format(n_c, CI), fontsize=7)
        # plt.show()
        plt.savefig(self.recovery_feature_predictions_save_folder+'clustered_surrogates/{0:}/Preds_{0:}_clusters_lineplot_only_avg_spline_CI_{1:}_medians_new.pdf'.format(n_c, CI), format='pdf', dpi=600)



    def predict_personalized_activity_model_corr_network(self, use_median=True, dimensionality_reduction='precomputed_tsne'):
        
        perp = 30
        lr= 200
        ee = 12
        last_day = 40

        df_actigraph = pd.read_csv(self.actigraph_filename)
        # df_actigraph = df_actigraph[df_actigraph['Day']>0]
        df_actigraph = df_actigraph[(df_actigraph['Day']>=-5) & (df_actigraph['Day']<41)]
        df_actigraph.reset_index(drop=True, inplace=True)
        print(df_actigraph)
        
        features = df_actigraph.columns.values[1:-1]
    
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(2,76) if i not in excluded_file_no]
        df_feat_importances = pd.read_csv(self.actigraph_predictions_save_folder+'patient_{}/'.format(1)+'feature_importances.csv')
        
        for patient in patients:
            df_feat_importances = pd.concat([df_feat_importances,pd.read_csv(self.actigraph_predictions_save_folder+'patient_{}/'.format(patient)+'feature_importances.csv')])
            # print(df_feat_importances)
            # print(len(df_feat_importances[features].mean(axis=0).values))
        df_feat_importances.reset_index(drop=True, inplace=True)
        print(df_feat_importances)  
       
        if use_median:
            df_feat_importances.loc[df_feat_importances.shape[0],2:] = df_feat_importances[features].median(axis=0).values
        else:
            df_feat_importances.loc[df_feat_importances.shape[0],2:] = df_feat_importances[features].mean(axis=0).values
        
        feature_importances = df_feat_importances.loc[df_feat_importances.shape[0]-1, features]
        # print(feature_importances)
        # feature_importances = feature_importances.sort_values(ascending=True)
        # print(feature_importances.index.values)
        # print(df_cytof)
        # old_features = df_actigraph.columns.values[1:]
        # new_df_actigraph_cols = ['Filename']
        # new_df_actigraph_cols.extend(feature_importances.index.values)
        # print(new_cytof_df_cols)
        # df_actigraph = df_actigraph[new_df_actigraph_cols]
        # features = df_actigraph.columns.values[1:]
        print(df_actigraph)
        
        
        # print(df_feat_importances)
        # print(feature_importances.max())
        
        
        # corr = df_cytof[features].corr(method ='spearman') 
        # # corr = df_cytof[features].corr(method ='pearson') 
        # print(corr)
        
        # p_values[where_are_NaNs] = 1
        # print(corr)

        # df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        # df_cytof['recovery_feat'] = df_recovery_features[self.target_recovery_feature]
        
        univar_corr = []
        univar_pval = []
        for col in features:
            # print(col)
            for patient in patients:
                univar_corr_patient = []
                univar_pval_patient = []
                if patient < 10:
                    patient_id = 'VAL0{}.agd'.format(patient)
                else:
                    patient_id = 'VAL{}.agd'.format(patient)
                gt_day_labels = [i for i in range(-5,0)]
                gt_day_labels.extend([i for i in range(1,41)])
                # print(gt_day_labels)
                # print(df_actigraph[df_actigraph['Filename']==patient_id])
                # print(df_actigraph[df_actigraph['Filename']==patient_id][col])
                # c, p = pearsonr(df_actigraph[df_actigraph['Filename']==patient_id][col], gt_day_labels)
                c, p = spearmanr(df_actigraph[df_actigraph['Filename']==patient_id][col], gt_day_labels)

                # c, p = pearsonr(df_actigraph[df_actigraph['Filename']==patient_id][col], [i for i in range(1,41)])
                # c, p = pearsonr(df_actigraph[df_actigraph['Filename']==patient_id][col], [i for i in range(1,43)])

                univar_corr_patient.append(c)
                univar_pval_patient.append(p)
            if use_median:
                univar_corr.append(np.median(univar_corr_patient))
                univar_pval.append(-1*np.log10(np.median(univar_pval_patient)))
            else:
                univar_corr.append(np.mean(univar_corr_patient))
                univar_pval.append(-1*np.log10(np.mean(univar_pval_patient)))
        
        # for i in range(len(univar_corr)):
        #     if univar_pval[i] < 0.05:
        #         print(univar_corr[i])
        #         print(univar_pval[i])
        
        # where_are_NaNs_univar_corr = np.isnan(univar_corr)
        # print('where_are_NaNs', where_are_NaNs_univar_corr.any())
        # univar_corr[where_are_NaNs_univar_corr] = 0
        # print(univar_corr)
        
        
        # print(p_values)
        
        
        # distance_matrix = 1 - np.abs(corr)
        for col in df_actigraph.columns[1:]:
            df_actigraph[col].fillna(0, inplace=True)
        df_actigraph.drop(['Filename', 'Day'], axis=1, inplace=True)
        print(df_actigraph)
    
        # corr = df_actigraph.corr(method ='pearson').values
        corr = df_actigraph.corr(method ='spearman').values
        # corr, p_values = spearmanr(df_cytof.values)
        where_are_NaNs = np.isnan(corr)
        print('where_are_NaNs', where_are_NaNs.any())
        
        corr[where_are_NaNs] = 0

        # node_sizes = np.arcsinh(feature_importances.values)*4000
        # node_sizes = []
        # for feature_imp in feature_importances.values:
        #     if feature_imp < 0.0005:
        #         node_sizes.append(0.0005*4000)
        #     else:
        #         node_sizes.append(feature_imp*4000)
        # print(min(node_sizes))
        # print(max(node_sizes))

        # print(node_sizes)
        # print(len(node_sizes))
        
        # jet = plt.get_cmap('Reds')
        # cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=5)
        # scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
        # colors = []
        # for i in plot_df['log_pval_mean'].values:
        #     colorVal = scalarMap.to_rgba(i)
        #     colors.append(colorVal)
        
        corr = np.clip(corr, -1, 1)
        distance_matrix = 1 - np.abs(corr)
        # distance_matrix = (1 - corr)/2
        # print(embedded.shape)
        # print(distance_matrix)
        # print(distance_matrix)
        # print(distance_matrix.shape)
        
        
        
        X = df_actigraph.values.transpose()
        # print(X)
        scalerX = StandardScaler()
        scalerX.fit(X)
        X_scaled = scalerX.transform(X)
        # print(X_scaled)
        # print(X_scaled.shape)
        # X_embedded = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(X)
        if dimensionality_reduction == 'tsne':
            embedded = TSNE(n_components=2, perplexity=20, random_state=42).fit_transform(X_scaled)
            # print(embedded)
            # print(embedded.shape)
           
        elif dimensionality_reduction == 'precomputed_tsne':
            # embedded = TSNE(n_components=2, metric='precomputed', random_state=42).fit_transform(distance_matrix)
            embedded = TSNE(n_components=2, metric='precomputed', perplexity=perp, early_exaggeration=ee, learning_rate=lr, random_state=42).fit_transform(distance_matrix)
            # print(embedded)
            # print(embedded.shape)
        # elif dimensionality_reduction == 'umap':
        #     embedded = umap.UMAP(n_neighbors=20, min_dist=0.3, metric='correlation').fit_transform(X_scaled)
        #     print(embedded)
        #     print(embedded.shape)
        else:
            embedded = []

        df_embedded = pd.DataFrame(data=embedded, columns=['a', 'b'])

        G = nx.Graph()

        eps = 1e-11
        node_colors = []
        gray_cm = plt.get_cmap('hot')
        # print(feature_importances.max())
        # print(-1*np.log10(feature_importances.max()+eps))
        # print(feature_importances.min())
        # print(-1*np.log10(feature_importances.min()+eps))
        
        cNorm  = matplotlib.colors.Normalize(vmin=-0.05, vmax=max(feature_importances.max(),0.05))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=gray_cm)
        for feature_imp in feature_importances.values:
            if feature_imp != 0 and feature_imp < eps:
                print(feature_imp)
                sys.exit()
            node_colors.append(scalarMap.to_rgba(feature_imp))
            # node_colors.append(scalarMap.to_rgba(-1*np.log10(feature_imp+eps)))
            
        
        node_sizes = univar_pval
        for idx in range(len(node_sizes)):
            node_sizes[idx] = 100*node_sizes[idx]
            # if np.abs(node_sizes[idx]) < 0.1:
            #     node_sizes[idx] = 0.1*200
            # else:
            #     node_sizes[idx] = np.abs(node_sizes[idx])*200

        # edge_colors = []
        # gray_cm = plt.get_cmap('gray')
        # cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=100)
        # scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=gray_cm)
        labels={}
        for i in range(distance_matrix.shape[0]):
            if len(embedded) != 0:
                G.add_node(i, pos=(embedded[i][0],embedded[i][1]))
            else:
                G.add_node(i)
            
            # labels[i] = col_labels[i]
            # if feature_importances[i] > 0.005:
            #     labels[i] = features[i]
            # else:
            #     labels[i] = ''
            # labels[i] = features[i]    
            labels[i] = ''
        # print(embedding.shape[0])
        print(distance_matrix)
        # for i in range(distance_matrix.shape[0]):
        #     for j in range(distance_matrix.shape[0]):

        #         # if (j > i) and (euclidean_distances[i][j] < 0.07):
        #         # if (j > i) and (distance_matrix[i][j] < 0.05):
        #         if (j > i) and (distance_matrix[i][j] < 0.20):
        #         # if (j > i) and (p_values[i][j] < 0.01):
        #         # if (j > i) and (p_values[i][j] < 0.001/embedding.shape[0]):
        #         # if (j > i) :
        #             G.add_edge(i,j)
        print(df_actigraph)
        for i, col1 in enumerate(df_actigraph.columns.values):
            for j, col2 in enumerate(df_actigraph.columns.values):
                if j > i:
                    # rho, pval = pearsonr(df_actigraph[col1], df_actigraph[col2])
                    rho, pval = spearmanr(df_actigraph[col1], df_actigraph[col2])
                    if (pval*1891) < 0.01:
                        G.add_edge(i,j)
                    # else:
                        # print('************************* not edge *************')
    
    
        
        

        
        

    
        print(node_sizes)
        print(len(node_sizes))
        
        print(df_embedded)
        df_embedded.to_csv(self.actigraph_predictions_save_folder+'Actigraph_features_df_tsne_embedded_v3_perplexity{}_lr{}_ee{}_spearman.csv'.format(perp, lr, ee), header=False, index=False)

        fig, ax = plt.subplots(1,1,figsize=(5,5))
        plt.title('repeat. 10-fold cv,\n node_size=univariate pearson corr. pval, color_intensity: median actigraph feat importance., edge:= featurewise pearson pval bonf corrected <0.01 (show top labels)', fontsize= 5)
        # plt.title('Timepoint = {}, repeat. 10-fold cv, {},\n size:= {} cytof feat importance., edge:= featuresize spearman corr>0.9'.format(timepoint, 'METs', 'Mean'), fontsize= 10)
        # red_patch = mpatches.Patch(color='red', label='High-level Act Ft')
        # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
        # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
        # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
        # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
        # plt.legend(handles=[red_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, 0.2),
        #        ncol=3,
        #        fontsize=8)
        # nx.draw(G, pos, node_color='black', node_size=20, alpha=0.8, edgecolors='lightgray', linewidths=.4, edge_color='lightgray')
        if len(embedded) != 0:
            pos = nx.get_node_attributes(G,'pos')
        else:
            # pos = nx.spring_layout(G,scale=4)
            # pos = nx.spectral_layout(G)
            # pos = nx.random_layout(G)
            # pos = nx.spectral_layout(G)
            pos = nx.circular_layout(G)
        
        nx.draw(G, pos, ax=ax, node_color=node_colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, edge_color='darkgrey', width=0.1)
        # nx.draw(G, pos, ax=ax, node_color=colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, linewidths=0.1, edgecolors='lightgray', edge_color='black', width=0.1)

        # red_patch = mpatches.Patch(color=cmap[0], label='High-level Act Ft')
        # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
        # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
        # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
        # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
        # plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch, black_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, -0.2),
        #        ncol=3,
        #        fontsize=8)

        # plt.title('tSNE cytof features')
        # plt.legend(handles=[red_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, -0.2),
        #        ncol=3,
        #        fontsize=8)
        # plt.xlim((-5, 5))
        # plt.ylim((-5, 5))
        # G2 = nx.Graph()
        # labels2={}
        # for i in range(2):
        #     G2.add_node(i, pos=(100, -15*i))
        #     labels2[i] = ['rho=0.25', 'rho=0.5'][i]
        # node_sizes2 = [label*400 for label in [0.25, 0.5]]
        # nx.draw(G2, nx.get_node_attributes(G2,'pos'), node_color='white', labels=labels2, font_size=8, node_size=node_sizes2, alpha=1, edgecolors='black', linewidths=.4)
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin = 0, vmax=5))
        # sm._A = []
        # plt.colorbar(scalarMap, shrink=.5)
        # plt.show()
        plt.savefig(self.actigraph_predictions_save_folder+'Actigraph_corr_network_univariate_and_median_feat_importance_tsne.pdf', format='pdf', dpi=600)



    def grid_optimized_RecoverySegmentedRegression(self,  patient_id):
        ''' use sklearn grid optimization to find the model with optimal breaking point '''

        patient_filename = self.actigraph_predictions_save_folder+'median_predictions/Predictions_patient_{}.csv'.format(patient_id)
        df = pd.read_csv(patient_filename)
        # print(df)
        if self.target_recovery_feature_type == 'min':
            df_pre_feat = df[(df['ground_truth']<0) & (df['ground_truth']>=self.first_day)]['predicted_value'].min()
        elif self.target_recovery_feature_type == '0.20_quantile':
            df_pre_feat = df[(df['ground_truth']<0) & (df['ground_truth']>=self.first_day)]['predicted_value'].quantile(0.2)
        elif self.target_recovery_feature_type == '0.25_quantile':
            df_pre_feat = df[(df['ground_truth']<0) & (df['ground_truth']>=self.first_day)]['predicted_value'].quantile(0.25)
        else:
            df_pre_feat = df[(df['ground_truth']<0) & (df['ground_truth']>=self.first_day)]['predicted_value'].mean()
        patient_pre_save_filename = self.actigraph_predictions_save_folder+'pre_predictions/'
        os.makedirs(patient_pre_save_filename, exist_ok=True)
        

        df[(df['ground_truth']<0) & (df['ground_truth']>=self.first_day)].to_csv(patient_pre_save_filename+'Pre_predictions_patient_{}.csv'.format(patient_id), header=True, index=False)
        
        # df_pre_min = df[(df['ground_truth']<0) & (df['ground_truth']>=self.first_day)]['predicted_value'].mean()
        # df_pre_min = np.percentile(df[(df['ground_truth']<0) & (df['ground_truth']>=self.first_day)]['predicted_value'], 25)
        # df_pre_min = df[df['ground_truth']<0]['predicted_value'].min()
        # df.loc[:6, 'predicted_value'] = [df_pre_min for _ in range(7)]
        
        df = df[df['ground_truth']>=self.first_day]
        df.reset_index(drop=True, inplace=True)
        print(df)
        X = df['ground_truth'].values
        X = np.asarray(X.reshape(-1,1))
        y = df['predicted_value'].values
        y = np.asarray(y.flatten())

        
        print('################## {}'.format(patient_filename))
        if np.any(np.isnan(df)):
            print(df)
        
        param_grid = [{'breaking_point': range(7,41)}]
        # param_grid = [{'breaking_point': range(7,43)}]
        # # param_grid = [{'breaking_point': range(2,41)}]
        best_score = np.inf
        best_grid, best_mode = None, None
        for g in ParameterGrid(param_grid):
            clf = RecoverySegmentedRegression(LinearRegression, LinearRegression, self.horizontal_recovered, **g)
            clf.fit(X, y)
            y_fitted = clf.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_fitted))
            
            
            # save if best
            if rmse < best_score:
                best_score = rmse
                best_grid = g
                best_model = clf
    
        
        
        return {'patient_id':patient_id, 'opt_breaking_point':best_model.breaking_point-4, 'recovery_slope':best_model.recovering_model.coef_[0], 'baseline_pred':df_pre_feat}
        # return {'patient_id':patient_id, 'opt_breaking_point':best_model.breaking_point-6, 'recovery_slope':best_model.recovering_model.coef_[0], 'baseline_pred':df_pre_feat}
        # return {'patient_id':patient_id, 'opt_breaking_point':0, 'recovery_slope':0, 'baseline_pred':df_pre_feat}
        


    def create_df_opt_breaking_points(self):

        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]

        # get median predictions of reps 
        
        
        for patient in [i for i in range(1,76) if i not in excluded_file_no]:
            patient_med_preds = pd.DataFrame(columns=['ground_truth', 'predicted_value'])
            patient_df = pd.DataFrame()
            for k in range(1,self.no_of_reps+1):
                patient_rep_df = pd.read_csv(self.actigraph_predictions_save_folder+'patient_{0:}_preIncluded/prediction_rep_{1:}.csv'.format(patient, k))
                patient_df = pd.concat([patient_df, patient_rep_df])
                # print(patient_df)
            patient_df['ground_truth'] = patient_df['ground_truth'].round(0)
            days = list(set(patient_df['ground_truth'].values))
            for day in days:
                patient_df_day = patient_df[patient_df['ground_truth']==day]
                patient_med_preds = pd.concat([patient_med_preds, pd.DataFrame(data=[[patient_df_day['ground_truth'].median(), patient_df_day['predicted'].median()]], columns=['ground_truth', 'predicted_value'])])
            patient_med_preds.sort_values(by='ground_truth', axis=0, ascending=True, inplace=True)
            patient_med_preds.reset_index(drop=True, inplace=True)
            os.makedirs(self.actigraph_predictions_save_folder+'median_predictions/', exist_ok=True)
            patient_med_preds.to_csv(self.actigraph_predictions_save_folder+'median_predictions/Predictions_patient_{}.csv'.format(patient), header=True, index=False)
        
        output_df = pd.DataFrame()
        for patient in [i for i in range(1,76) if i not in excluded_file_no]:
            
                row = self.grid_optimized_RecoverySegmentedRegression(patient)
                
                output_df = pd.concat([output_df, pd.DataFrame(data=np.asarray(list(row.values())).reshape(1,-1), columns=list(row.keys()))])
                # print(pd.DataFrame(data=np.asarray(list(row.values())).reshape(1,-1), columns=list(row.keys())))
            
        print(output_df)
        
        # output_path = '/'.join(filepath.split('/')[:-1]) + '/Opt_breaking_points'
        output_path = self.extract_recovery_feature_save_folder + 'Opt_breaking_points/'
        os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_df.to_csv(output_path+'opt_breaking_points.csv', header=True, index=False)
       



    def extract_recovery_feature(self):

        
        self.create_df_opt_breaking_points()

        
        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        df_org = pd.read_csv(self.org_slope_filename)
        print('+++++++++++++++ recovery slope correlation with org +++++++++++++++')
        print(spearmanr(df_recovery_features['recovery_slope'], df_org['recovery_slope']))
        print(pearsonr(df_recovery_features['recovery_slope'], df_org['recovery_slope']))
        print(mean_absolute_error(df_recovery_features['recovery_slope'], df_org['recovery_slope']))
        print('+++++++++++++++ opt breaking point correlation with org +++++++++++++++')
        print(spearmanr(df_recovery_features['opt_breaking_point'], df_org['opt_breaking_point']))
        print(pearsonr(df_recovery_features['opt_breaking_point'], df_org['opt_breaking_point']))
        print(mean_absolute_error(df_recovery_features['opt_breaking_point'], df_org['opt_breaking_point']))
        # print('+++++++++++++++ min baseline pred correlation with org +++++++++++++++')
        # print(spearmanr(df_recovery_features['baseline_pred'], df_org['baseline_pred_min']))
        # print(pearsonr(df_recovery_features['baseline_pred'], df_org['baseline_pred_min']))
        # print(mean_absolute_error(df_recovery_features['baseline_pred'], df_org['baseline_pred_min']))

        corrs_filename = self.extract_recovery_feature_save_folder+'correlations_with_org_slope.txt'
        with open(corrs_filename, 'w') as filetowrite:
            corrs_with_org = []
            corrs_with_org.append('+++++++++++++++ recovery slope correlation with org +++++++++++++++\n')
            corrs_with_org.append('spearman = {}\n'.format(spearmanr(df_recovery_features['recovery_slope'], df_org['recovery_slope'])))
            corrs_with_org.append('pearson = {}\n'.format(pearsonr(df_recovery_features['recovery_slope'], df_org['recovery_slope'])))
            corrs_with_org.append('mean_absolute_error = {}\n'.format(mean_absolute_error(df_recovery_features['recovery_slope'], df_org['recovery_slope'])))
            corrs_with_org.append('+++++++++++++++ opt breaking point correlation with org +++++++++++++++\n')
            corrs_with_org.append('spearman = {}\n'.format(spearmanr(df_recovery_features['opt_breaking_point'], df_org['opt_breaking_point'])))
            corrs_with_org.append('pearson = {}\n'.format(pearsonr(df_recovery_features['opt_breaking_point'], df_org['opt_breaking_point'])))
            corrs_with_org.append('mean_absolute_error = {}\n'.format(mean_absolute_error(df_recovery_features['opt_breaking_point'], df_org['opt_breaking_point'])))
            # corrs_with_org.append('+++++++++++++++ min baseline pred correlation with org +++++++++++++++\n')
            # corrs_with_org.append('spearman = {}\n'.format(spearmanr(df_recovery_features['baseline_pred'], df_org['baseline_pred_min'])))
            # corrs_with_org.append('pearson = {}\n'.format(pearsonr(df_recovery_features['baseline_pred'], df_org['baseline_pred_min'])))
            # corrs_with_org.append('mean_absolute_error = {}\n'.format(mean_absolute_error(df_recovery_features['baseline_pred'], df_org['baseline_pred_min'])))
            filetowrite.writelines(corrs_with_org)
            filetowrite.close()



    def predict_recovery_feature_repeated_k_folds(self):

        os.makedirs(self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/', exist_ok=True)
        corrs_filename = self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/prediction_correlations.txt'
        pred_corrs = []
        



        print('**************************** repeated_k_folds_recovery')
        cytof_filename = self.cytof_dict[self.cytof_dataset]
        df_cytof = pd.read_csv(cytof_filename)
        # df_recovery_features = pd.read_csv(self.org_slope_filename)
        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        # print(df_cytof)
        
        # df_cytof.fillna(0, inplace=True)
        

        df_cytof['recovery_feature'] = df_recovery_features[self.target_recovery_feature]
        df_cytof = df_cytof.iloc[:35,:] # only VAL<54
        print(df_cytof)
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer
        from sklearn.neighbors import KNeighborsRegressor
        # imp = IterativeImputer(missing_values=np.nan, sample_posterior=False, 
        #                          max_iter=10, tol=0.001, 
        #                          n_nearest_features=4, initial_strategy='median')
        imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), random_state=42)
        imp_data = imp.fit_transform(df_cytof.iloc[:,1:-1].values)
        df_cytof.iloc[:,1:-1] = imp_data
        print(df_cytof)
    

        
        
        
        df_repeatedkfold_predictions = pd.DataFrame(columns=['random_state','spearman_rho', 'spearman_pval','pearson_rho', 'pearson_pval'])
        
        df_feat_importance_col = ['iteration', 'fold']
        df_feat_importance_col.extend(df_cytof.columns.values[1:-1])
        df_feat_importance = pd.DataFrame(columns=df_feat_importance_col)
        df_all_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
        # print(df_feat_importance)
        for k in range(1,no_of_reps+1):
            df_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
            kf = KFold(n_splits=no_of_folds, shuffle=True, random_state=k)
            X = df_cytof.iloc[:, 1:-1].values
            y = df_cytof.iloc[:, -1].values
            patient_ids = df_recovery_features['patient_id'].values
            fold = 0
            for train_index, test_index in kf.split(X):
                print('rep = {}, fold = {}'.format(k,fold))

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                patient_ids_train, patient_ids_test = patient_ids[train_index], patient_ids[test_index]
        

                # scalerX = StandardScaler()
                # scalerX.fit(X_train)
                # X_train = scalerX.transform(X_train)
                # X_test = scalerX.transform(X_test)

                # scalerY = StandardScaler()
                # scalerY.fit(y_train.reshape(-1, 1))
                # y_train = scalerY.transform(y_train.reshape(-1, 1))
                # y_test = scalerY.transform(y_test.reshape(-1, 1))

                max_iter = 1000000
                if self.estimator_str == 'RF':
                    clf = RandomForestRegressor(n_estimators=self.RF_no_of_estimators, random_state=k)
                else: 
                    clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=self.RF_no_of_estimators, random_state=k)
                
                # clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))
                clf.fit(X_train, y_train.flatten())
                # param_grid=[{'n_estimators':range(10,301,10), 'loss':['linear', 'square', 'exponential'] }]
                # param_grid = [{'n_estimators': [50,100,200], 'max_depth': [2, 5, 10, None]}]
                # grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=20, iid=True)
                # grid.fit(X_train, y_train.flatten())
            
                # clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=n_estimators, loss=loss)
                # clf = RandomForestRegressor(n_estimators=200,  n_jobs=20, random_state=k)
                # clf.fit(X_train,y_train.flatten())
                
                # clf = grid.best_estimator_
                
            
                y_predicted = clf.predict(X_test)
                
                # Y_predicted = clf.predict(X_test_final)
                # clf_params = clf.get_params()
            
                importances = clf.feature_importances_
                
                
                df_feat_importance.loc[df_feat_importance.shape[0],:] = [k, fold] + list(importances)
                
                # Print the feature ranking
                # print("Feature ranking:")
                # for f in range(X.shape[1]):
                #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                
                df_fold_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
                for i in range(y_test.shape[0]):
                    # df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                    # df_all_predictions.loc[df_all_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                    # df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                    df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                    df_all_predictions.loc[df_all_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                    df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                fold += 1
                # rho, pval = spearmanr(df_fold_predictions['recovery_feature'], df_fold_predictions['predicted'])
                # print('spearman rho = {} pval = {}'.format(rho, pval))
                # rho, pval = pearsonr(df_fold_predictions['recovery_feature'], df_fold_predictions['predicted'])
                # print('pearson rho = {} pval = {}'.format(rho, pval))

            
            
            df_predictions.sort_values(by='patient_id', axis=0, ascending=True, inplace=True)
            df_predictions.reset_index(drop=True, inplace=True)
            
            # df_predictions_filename = '/home/raminf/HipVal/recovery_features/Ramin_BaselinePrediction/Predictions_repeated_10folds_for_Franck_AGD_segmented_recovery_pre_stim_adjusted/RF_no_grid_n_estimators_500/'+feature
            # df_predictions_filename = '/home/raminf/HipVal/recovery_features/Ramin_BaselinePrediction/Predictions_repeated_10folds_for_Franck/old_Clustered_segmented_recovery_pre_stim_adjusted/'+feature
            # os.makedirs(df_predictions_filename, exist_ok=True)
            # df_predictions.to_csv(df_predictions_filename+'/prediction_rep_{}.csv'.format(k), header=True, index=False)

            rho, pval = spearmanr(df_predictions['recovery_feature'], df_predictions['predicted'])
            print('spearman rho = {} pval = {}'.format(rho, pval))
            rho_pearson, pval_pearson = pearsonr(df_predictions['recovery_feature'], df_predictions['predicted'])
            print('pearson rho = {} pval = {}'.format(rho_pearson, pval_pearson))
            df_repeatedkfold_predictions.loc[df_repeatedkfold_predictions.shape[0],:] = [k, rho, pval, rho_pearson, pval_pearson]
        print(df_repeatedkfold_predictions)   
        df_repeatedkfold_predictions.to_csv(self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/reps_prediction_correlations.csv', header=True, index=False) 
        df_feat_importance.to_csv(self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/feature_importances.csv', header=True, index=False) 
        df_all_predictions.sort_values(by='patient_id', axis=0, ascending=True, inplace=True)
        df_all_predictions.reset_index(drop=True, inplace=True)
        print(df_all_predictions)
        df_all_predictions.to_csv(self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/all_predictions.csv', header=True, index=False)
        rho, pval = spearmanr(df_all_predictions['recovery_feature'], df_all_predictions['predicted'])
        print('spearman rho = {} pval = {}'.format(rho, pval))
        pred_corrs.append('+++++++++++++++ all_predictions correlation +++++++++++++++\n')
        pred_corrs.append('spearman rho = {} pval = {}\n'.format(rho, pval))
        rho, pval = pearsonr(df_all_predictions['recovery_feature'], df_all_predictions['predicted'])
        print('pearson rho = {} pval = {}'.format(rho, pval))
        pred_corrs.append('pearson rho = {} pval = {}\n'.format(rho, pval))
        

        patients = list(set(df_all_predictions['patient_id'].values))
        df_med_preds = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
        for patient in patients:
            patient_df = df_all_predictions[df_all_predictions['patient_id']==patient]
            df_med_preds.loc[df_med_preds.shape[0],:] = [patient, patient_df['recovery_feature'].values[0], patient_df['predicted'].median()]
        print(df_med_preds)
        df_med_preds.to_csv(self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/median_predictions.csv', header=True, index=False)
        rho, pval = spearmanr(df_med_preds['recovery_feature'], df_med_preds['predicted'])
        print('spearman rho = {} pval = {}'.format(rho, pval))
        pred_corrs.append('+++++++++++++++ median_predictions correlation +++++++++++++++\n')
        pred_corrs.append('spearman rho = {} pval = {}\n'.format(rho, pval))
        rho, pval = pearsonr(df_med_preds['recovery_feature'], df_med_preds['predicted'])
        print('pearson rho = {} pval = {}'.format(rho, pval))
        pred_corrs.append('pearson rho = {} pval = {}\n'.format(rho, pval))


        with open(corrs_filename, 'w') as filetowrite:
            filetowrite.writelines(pred_corrs)
            filetowrite.close()
        
        # r2 = r2_score(df_predictions['baseline'], df_predictions['predicted'])
        # print('r2 = ', r2)
        # 


    def plot_recovery_feature_scatter_plot_predictions(self):

        
        df_med_preds = pd.read_csv(self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/median_predictions.csv')
        df_med_preds.columns = ['patient_id', self.target_recovery_feature, 'predicted']
        print(df_med_preds)
        rho, pval = pearsonr(df_med_preds[self.target_recovery_feature], df_med_preds['predicted'])
        print('pearson rho = {} pval = {}'.format(rho, pval))
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        # df_med_preds.plot(kind='scatter', x=self.target_recovery_feature, y='predicted', color='black', ax=ax)
        sns.regplot(x=self.target_recovery_feature, y='predicted', data=df_med_preds, ax=ax, robust=True, color='black', x_ci='ci', ci=90)
        plt.title('Basline prediction model ground truth value vs predicted by cytof.\npearson rho = {:.2f} pval = {:.4f}'.format(rho, pval), fontsize=7)
        # ax.set_ylim([5,35])
        # ax.set_xlim([5,40])
        # plt.show()
        plt.savefig(self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/scatter_plot_predictions_{}_surrogate_v5.pdf'.format(self.target_recovery_feature), format='pdf', dpi=600)
        plt.close()

    def predict_recovery_feature_repeated_k_folds_proteomics(self):
        recovery_feature_predictions_save_folder = self.recovery_feature_predictions_save_folder+'proteomics_prediction/'
        os.makedirs(recovery_feature_predictions_save_folder, exist_ok=True)
        corrs_filename = recovery_feature_predictions_save_folder+'prediction_correlations.txt'
        pred_corrs = []
        



        print('**************************** repeated_k_folds_recovery')
        df_olink = pd.read_csv(self.olink_filename)
        # df_recovery_features = pd.read_csv(self.org_slope_filename)
        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        # print(df_cytof)
        
        
        

        df_olink['recovery_feature'] = df_recovery_features[self.target_recovery_feature]
        df_olink = df_olink.iloc[:35,:] # only VAL<54
        # print(df_olink)
        
        #impute missing values using ExtraTreesRegressor similar to misforest in R
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer
        from sklearn.neighbors import KNeighborsRegressor
        # imp = IterativeImputer(missing_values=np.nan, sample_posterior=False, 
        #                          max_iter=10, tol=0.001, 
        #                          n_nearest_features=4, initial_strategy='median')
        # imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), random_state=42)
        # imp = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=15), random_state=42)
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # print(df_olink.iloc[:,1:-1].values)
        # print(df_olink.iloc[:,1:-1].values.shape)
        # imp_data = imp.fit_transform(df_olink.iloc[:,1:-1].values)
        # print(imp_data)
        # print(imp_data.shape)
        # print(imp.transform(df_olink.iloc[:,1:-1].values))
        # print(imp.transform(df_olink.iloc[:,1:-1].values).shape)
        # print(df_olink.iloc[:,1:-1].values.shape)
        # df_olink.fillna(0, inplace=True)
        print(df_olink)
       
        
        df_repeatedkfold_predictions = pd.DataFrame(columns=['random_state','spearman_rho', 'spearman_pval','pearson_rho', 'pearson_pval'])
        
        df_feat_importance_col = ['iteration', 'fold']
        df_feat_importance_col.extend(df_olink.columns.values[1:-1])
        df_feat_importance = pd.DataFrame(columns=df_feat_importance_col)
        df_all_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
        # print(df_feat_importance)
        for k in range(1,no_of_reps+1):
            df_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
            kf = KFold(n_splits=no_of_folds, shuffle=True, random_state=k)
            X = df_olink.iloc[:, 1:-1].values
            y = df_olink.iloc[:, -1].values
            patient_ids = df_recovery_features['patient_id'].values
            fold = 0
            for train_index, test_index in kf.split(X):
                print('rep = {}, fold = {}'.format(k,fold))

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                patient_ids_train, patient_ids_test = patient_ids[train_index], patient_ids[test_index]
        

                # scalerX = StandardScaler()
                # scalerX.fit(X_train)
                # X_train = scalerX.transform(X_train)
                # X_test = scalerX.transform(X_test)

                # scalerY = StandardScaler()
                # scalerY.fit(y_train.reshape(-1, 1))
                # y_train = scalerY.transform(y_train.reshape(-1, 1))
                # y_test = scalerY.transform(y_test.reshape(-1, 1))

                max_iter = 1000000
                if self.estimator_str == 'RF':
                    clf = RandomForestRegressor(n_estimators=self.RF_no_of_estimators, random_state=k)
                else: 
                    clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=self.RF_no_of_estimators, random_state=k)
                
                # clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))
                clf.fit(X_train, y_train.flatten())
                # param_grid=[{'n_estimators':range(10,301,10), 'loss':['linear', 'square', 'exponential'] }]
                # param_grid = [{'n_estimators': [50,100,200], 'max_depth': [2, 5, 10, None]}]
                # grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=20, iid=True)
                # grid.fit(X_train, y_train.flatten())
            
                # clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=n_estimators, loss=loss)
                # clf = RandomForestRegressor(n_estimators=200,  n_jobs=20, random_state=k)
                # clf.fit(X_train,y_train.flatten())
                
                # clf = grid.best_estimator_
                
            
                y_predicted = clf.predict(X_test)
                
                # Y_predicted = clf.predict(X_test_final)
                # clf_params = clf.get_params()
            
                importances = clf.feature_importances_
                
                
                df_feat_importance.loc[df_feat_importance.shape[0],:] = [k, fold] + list(importances)
                
                # Print the feature ranking
                # print("Feature ranking:")
                # for f in range(X.shape[1]):
                #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                
                df_fold_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
                for i in range(y_test.shape[0]):
                    # df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                    # df_all_predictions.loc[df_all_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                    # df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                    df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                    df_all_predictions.loc[df_all_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                    df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                fold += 1
                # rho, pval = spearmanr(df_fold_predictions['recovery_feature'], df_fold_predictions['predicted'])
                # print('spearman rho = {} pval = {}'.format(rho, pval))
                # rho, pval = pearsonr(df_fold_predictions['recovery_feature'], df_fold_predictions['predicted'])
                # print('pearson rho = {} pval = {}'.format(rho, pval))

            
            
            df_predictions.sort_values(by='patient_id', axis=0, ascending=True, inplace=True)
            df_predictions.reset_index(drop=True, inplace=True)
            
            # df_predictions_filename = '/home/raminf/HipVal/recovery_features/Ramin_BaselinePrediction/Predictions_repeated_10folds_for_Franck_AGD_segmented_recovery_pre_stim_adjusted/RF_no_grid_n_estimators_500/'+feature
            # df_predictions_filename = '/home/raminf/HipVal/recovery_features/Ramin_BaselinePrediction/Predictions_repeated_10folds_for_Franck/old_Clustered_segmented_recovery_pre_stim_adjusted/'+feature
            # os.makedirs(df_predictions_filename, exist_ok=True)
            # df_predictions.to_csv(df_predictions_filename+'/prediction_rep_{}.csv'.format(k), header=True, index=False)

            rho, pval = spearmanr(df_predictions['recovery_feature'], df_predictions['predicted'])
            print('spearman rho = {} pval = {}'.format(rho, pval))
            rho_pearson, pval_pearson = pearsonr(df_predictions['recovery_feature'], df_predictions['predicted'])
            print('pearson rho = {} pval = {}'.format(rho_pearson, pval_pearson))
            df_repeatedkfold_predictions.loc[df_repeatedkfold_predictions.shape[0],:] = [k, rho, pval, rho_pearson, pval_pearson]
        print(df_repeatedkfold_predictions)   
        df_repeatedkfold_predictions.to_csv(recovery_feature_predictions_save_folder+'reps_prediction_correlations.csv', header=True, index=False) 
        df_feat_importance.to_csv(recovery_feature_predictions_save_folder+'feature_importances.csv', header=True, index=False) 
        df_all_predictions.sort_values(by='patient_id', axis=0, ascending=True, inplace=True)
        df_all_predictions.reset_index(drop=True, inplace=True)
        print(df_all_predictions)
        df_all_predictions.to_csv(recovery_feature_predictions_save_folder+'all_predictions.csv', header=True, index=False)
        rho, pval = spearmanr(df_all_predictions['recovery_feature'], df_all_predictions['predicted'])
        print('spearman rho = {} pval = {}'.format(rho, pval))
        pred_corrs.append('+++++++++++++++ all_predictions correlation +++++++++++++++\n')
        pred_corrs.append('spearman rho = {} pval = {}\n'.format(rho, pval))
        rho, pval = pearsonr(df_all_predictions['recovery_feature'], df_all_predictions['predicted'])
        print('pearson rho = {} pval = {}'.format(rho, pval))
        pred_corrs.append('pearson rho = {} pval = {}\n'.format(rho, pval))
        

        patients = list(set(df_all_predictions['patient_id'].values))
        df_med_preds = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
        for patient in patients:
            patient_df = df_all_predictions[df_all_predictions['patient_id']==patient]
            df_med_preds.loc[df_med_preds.shape[0],:] = [patient, patient_df['recovery_feature'].values[0], patient_df['predicted'].median()]
        print(df_med_preds)
        df_med_preds.to_csv(recovery_feature_predictions_save_folder+'median_predictions.csv', header=True, index=False)
        rho, pval = spearmanr(df_med_preds['recovery_feature'], df_med_preds['predicted'])
        print('spearman rho = {} pval = {}'.format(rho, pval))
        pred_corrs.append('+++++++++++++++ median_predictions correlation +++++++++++++++\n')
        pred_corrs.append('spearman rho = {} pval = {}\n'.format(rho, pval))
        rho, pval = pearsonr(df_med_preds['recovery_feature'], df_med_preds['predicted'])
        print('pearson rho = {} pval = {}'.format(rho, pval))
        pred_corrs.append('pearson rho = {} pval = {}\n'.format(rho, pval))


        with open(corrs_filename, 'w') as filetowrite:
            filetowrite.writelines(pred_corrs)
            filetowrite.close()
        
        # r2 = r2_score(df_predictions['baseline'], df_predictions['predicted'])
        # print('r2 = ', r2)
    
    
    def plot_recovery_feature_scatter_plot_predictions_proteomics(self):

        recovery_feature_predictions_save_folder = self.recovery_feature_predictions_save_folder+'proteomics_prediction/'
        df_med_preds = pd.read_csv(recovery_feature_predictions_save_folder+'median_predictions.csv')
        df_med_preds.columns = ['patient_id', self.target_recovery_feature, 'predicted']
        print(df_med_preds)
        rho, pval = pearsonr(df_med_preds[self.target_recovery_feature], df_med_preds['predicted'])
        print('pearson rho = {} pval = {}'.format(rho, pval))
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        # df_med_preds.plot(kind='scatter', x=self.target_recovery_feature, y='predicted', color='black', ax=ax)
        sns.regplot(x=self.target_recovery_feature, y='predicted', data=df_med_preds, ax=ax, robust=True, color='black', x_ci='ci', ci=90)
        plt.title('Basline prediction model ground truth value vs predicted by proteomics.\npearson rho = {:.2f} pval = {:.4f}'.format(rho, pval), fontsize=7)
        # ax.set_ylim([5,35])
        # ax.set_xlim([5,40])
        # plt.show()
        plt.savefig(recovery_feature_predictions_save_folder+'scatter_plot_predictions_{}_surrogate_v1.pdf'.format(self.target_recovery_feature), format='pdf', dpi=600)
        plt.close()



    def plot_cytof_simple_proteomics_network(self, dimensionality_reduction='precomputed_tsne'):

    
        df_olink = pd.read_csv(self.olink_filename)
        print(df_olink)
        
        features = df_olink.columns.values[1:]

        
        # print(df_cytof)
        
       
        # for col in df_cytof.columns[1:]:
        #     df_cytof[col].fillna(0, inplace=True)
        # df_olink = df_olink.iloc[:35,:] # only VAL<54
        # print(df_olink)
       
        # from sklearn.ensemble import ExtraTreesRegressor
        # from sklearn.experimental import enable_iterative_imputer
        # from sklearn.impute import IterativeImputer, SimpleImputer
        # from sklearn.neighbors import KNeighborsRegressor
        # # imp = IterativeImputer(missing_values=np.nan, sample_posterior=False, 
        # #                          max_iter=10, tol=0.001, 
        # #                          n_nearest_features=4, initial_strategy='median')
        # imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), random_state=42)
        # imp_data = imp.fit_transform(df_cytof.iloc[:,1:].values)
        # df_cytof.iloc[:,1:] = imp_data
        # print(df_cytof)
        
        df_olink.drop(['patient_id'], axis=1, inplace=True)
        # corr = df_cytof.corr(method ='pearson').values
        corr = df_olink.corr(method ='spearman').values
        # corr, p_values = spearmanr(df_cytof.values)
        where_are_NaNs = np.isnan(corr)
        print('where_are_NaNs', where_are_NaNs.any())
        
        corr[where_are_NaNs] = 0
        

       
        
        corr = np.clip(corr, -1, 1)
        distance_matrix = 1 - np.abs(corr)
        # distance_matrix = (1 - corr)/2
        # print(embedded.shape)
        # print(distance_matrix)
        print(distance_matrix)
        print(distance_matrix.shape)
        
        
        
            
           
        if dimensionality_reduction == 'precomputed_tsne':
            embedded = TSNE(n_components=2, metric='precomputed', random_state=42).fit_transform(distance_matrix)
            print(embedded)
            print(embedded.shape)
            df_embedded = pd.DataFrame(data=embedded, columns=['d1', 'd2'])
            df_embedded['node_label'] = df_olink.columns.values
            print(df_embedded)
            # sys.exit()
            # df_embedded.to_csv(self.recovery_feature_predictions_save_folder+'/simple_univariate_cytof_corr_networks/Cytof_features_df_tsne_embedded.csv', header=True, index=False)
            # os.makedirs(self.recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/', exist_ok=True)
            # df_embedded.to_csv(self.recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/Cytof_features_df_tsne_embedded_spearman.csv', header=True, index=False)
            print('here')
            # sys.exit()
        # elif dimensionality_reduction == 'umap':
        #     embedded = umap.UMAP(n_neighbors=20, min_dist=0.3, metric='correlation').fit_transform(X_scaled)
        #     print(embedded)
        #     print(embedded.shape)
        else:
            embedded = []

        print('here')
        
        col_labels= features.copy()
        embedding = embedded.copy()

        G = nx.Graph()

        eps = 1e-11
        node_colors = []
        node_colors_idx = []
        # node_sizes = univar_pval

        # if self.cytof_simple_corr_network_color_by == 'univar_corr':
        #     gray_cm = plt.get_cmap('seismic')
        #     cNorm  = matplotlib.colors.Normalize(vmin=-1*max(univar_pval), vmax=max(univar_pval))
        #     scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=gray_cm)
        #     for rho,p in zip(univar_corr,univar_pval):
        #         if self.target_recovery_feature == 'recovery_slope':
        #             if rho < 0:
        #                 node_colors.append(scalarMap.to_rgba(-1*p))
        #             else:
        #                 node_colors.append(scalarMap.to_rgba(p))
        #         elif self.target_recovery_feature == 'baseline_pred':
        #             if rho >= 0:
        #                 node_colors.append(scalarMap.to_rgba(-1*p))
        #             else:
        #                 node_colors.append(scalarMap.to_rgba(p))
        #         else:
        #             print('Error Unknown target_recovery_feature')
        #             sys.exit()

        
        
        df_prot_labels = pd.read_csv('/home/raminf/HipVal/proteomics/protlabels.csv')
        print(df_prot_labels)
        
        subsets = list(set(df_prot_labels['protlabels'].values))
        print(subsets)
        
        number_of_subsets= len(subsets)
        # cm_subsection = np.linspace(0, 1, number_of_subsets) 
        # colors = [cm.jet(x) for x in cm_subsection]
        cm_subsection = np.linspace(0.01, 0.75, number_of_subsets) 
        colors = [cm.terrain(x) for x in cm_subsection]
        # cm_subsection = np.linspace(0.01, 0.99, number_of_subsets) 
        # colors = [cm.gnuplot(x) for x in cm_subsection]
            
        # print(colors)

        for feat in df_olink.columns.values:
            #fix mismatch between feat names in two files
            feat = feat.replace('-','.')
            feat = feat.replace(' ','.')
            feat = feat.replace('/','.')
            feat = feat.replace("'",'.')
            feat = feat.replace('4E.BP1','X4E.BP1') #inflammation 
            feat = feat.replace('5..NT','X5..NT') #oncology label
            assert df_prot_labels[df_prot_labels['feature']==feat].shape[0] == 1, 'Multiple or no "{}" not found in prot label file'.format(feat)
            prot_label = df_prot_labels[df_prot_labels['feature']==feat]['protlabels'].values[0]
            # print(feat)
            # print(prot_label)
            # print(subsets.index(prot_label))
            # print('---------')
            node_colors.append(colors[subsets.index(prot_label)])
        legend_elements = []
        for ct,c in zip(subsets,colors):
            legend_elements.append(mpatches.Patch(facecolor=c,label=ct))
        # if self.cytof_simple_corr_network_color_by == 'marker':
        #     legend_elements.append(mpatches.Patch(facecolor='black',label='freq'))
      
        
        df_embedded['color'] = [matplotlib.colors.rgb2hex(i) for i in node_colors]
        recovery_feature_predictions_save_folder = self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/'
        os.makedirs(recovery_feature_predictions_save_folder+'simple_olink_corr_networks/', exist_ok=True)
        df_embedded.to_csv(recovery_feature_predictions_save_folder+'simple_olink_corr_networks/Olink_features_df_tsne_embedded_spearman.csv', header=True, index=False)
        
        # use uivar pval with overal recovery surrogate as size
        # for idx in range(len(node_sizes)):
        #     # node_sizes[idx] = 50*node_sizes[idx]
        #     if np.abs(node_sizes[idx]) < 0.1:
        #         node_sizes[idx] = 0.1*100
        #     else:
        #         node_sizes[idx] = np.abs(node_sizes[idx])*100
        #     if node_sizes[idx] > 300:
        #         node_sizes[idx] = 300
        # use cca weights as size
        # node_sizes = list(pd.read_csv(self.main_save_folder+'Rcca/cca-cytof-acti_cytofweights_v1.csv')['V1'].values)
   
        # print(max(node_sizes))
        # print(min(node_sizes))
        # for idx in range(len(node_sizes)):
        #     # node_sizes[idx] = 50*node_sizes[idx]
        #     if np.abs(node_sizes[idx]) < 0.003:
        #         node_sizes[idx] = 0.003*1000
        #     else:
        #         node_sizes[idx] = np.abs(node_sizes[idx])*1000
        #     # if node_sizes[idx] > 300:
        #     #     node_sizes[idx] = 300

        labels={}
        for i in range(distance_matrix.shape[0]):
            if len(embedding) != 0:
                G.add_node(i, pos=(embedding[i][0],embedding[i][1]))
            else:
                G.add_node(i)
            
            # labels[i] = col_labels[i]
            labels[i] = ''
            # if feature_importances[i] > 0.005:
            #     labels[i] = col_labels[i]
            # else:
            #     labels[i] = ''
            
        # print(embedding.shape[0])
        print(distance_matrix)
        # for i in range(distance_matrix.shape[0]):
        #     for j in range(distance_matrix.shape[0]):

        #         # if (j > i) and (euclidean_distances[i][j] < 0.07):
        #         # if (j > i) and (distance_matrix[i][j] < 0.05):
        #         if (j > i) and (distance_matrix[i][j] < 0.20):
        #         # if (j > i) and (p_values[i][j] < 0.01):
        #         # if (j > i) and (p_values[i][j] < 0.001/embedding.shape[0]):
        #         # if (j > i) :
        #             G.add_edge(i,j)
        

        network_saved = True
        if not network_saved:
            all_comparisons = df_olink.shape[1]*(df_olink.shape[1]-1)/2
            print(all_comparisons)
            for i, col1 in enumerate(df_olink.columns.values):
                print(i)
                for j, col2 in enumerate(df_olink.columns.values):
                    if j > i:
                        # rho, pval = pearsonr(df_cytof[col1], df_cytof[col2])
                        rho, pval = spearmanr(df_olink[col1], df_olink[col2])
                        if (pval*all_comparisons) < 0.05:
                            G.add_edge(i,j)
                        # else:
                            # print('************************* not edge *************')
    

        if network_saved:
            import pickle
            print(pickle.load(open(recovery_feature_predictions_save_folder+'simple_olink_corr_networks/G.pkl','rb')))
            G = pickle.load(open(recovery_feature_predictions_save_folder+'simple_olink_corr_networks/G.pkl','rb'))['G']
        else:
            import pickle
            pickle.dump( {'G':G}, open(recovery_feature_predictions_save_folder+'simple_olink_corr_networks/G.pkl', 'wb' ) )
        
        # if self.cytof_simple_corr_network_color_by == 'community_detection':
        #     comm_id = 1
        #     print('computing communities')
        #     no_of_isolated_nodes = 0
        #     no_of_isolated_nodes2 = 0
        #     for i in range(len(G.nodes())):
        #         # print(G.degree[i])
        #         if G.degree[i] ==0:
        #             no_of_isolated_nodes +=1 
        #     print('no of isolated nodes: {}'.format(no_of_isolated_nodes))
           

        #     import community

        #     # fig, ax = plt.subplots(figsize=(5,5))
        #     # plt.title('{}, simple corr. network,\n node_size=univariate pearson corr. pval , edge:= featurewise pearson pval bonf corrected <0.01, colored by = {}'.format(timepoint, self.cytof_simple_corr_network_color_by), fontsize= 5)
          
        #     pos = nx.get_node_attributes(G,'pos')
           
        #     #first compute the best partition
        #     partition = community.best_partition(G, resolution=1, random_state=42)
        #     # print(partition.keys())
        #     # print(partition.values())
        #     # print(set(partition.values()))
        #     total_no_of_communities = len(set(partition.values()))
        #     number_of_subsets= total_no_of_communities - no_of_isolated_nodes
        #     print('number_of_subsets: {}'.format(number_of_subsets))
        #     # cm_subsection = np.linspace(0, 1, number_of_subsets) 
        #     # colors = [cm.jet(x) for x in cm_subsection]
        #     cm_subsection = np.linspace(0.01, 0.99, number_of_subsets) 
        #     # cm_subsection = np.linspace(0.01, 0.75, number_of_subsets) 
        #     colors = [cm.gist_ncar(x) for x in cm_subsection]
        #     # colors = [cm.terrain(x) for x in cm_subsection]
        #     # colors = [cm.hsv(x) for x in cm_subsection]
        #     # cm_subsection = np.linspace(0.01, 0.99, number_of_subsets) 
        #     # colors = [cm.gnuplot(x) for x in cm_subsection]
                
        #     # print(colors)
        #     print(len(colors))
        #     new_coms = []
        #     new_coms_sizes = []
        #     comm_id_node_sizes = node_sizes.copy()
        #     for i, node in enumerate(range(len(G.nodes()))):
        #         this_partition = partition[node]
        #         this_partition_nodes = [nodes for nodes in partition.keys() if partition[nodes] == this_partition]
        #         if len(this_partition_nodes) > 1:
        #             if this_partition not in new_coms:
        #                 new_coms_sizes.append(len(this_partition_nodes))
        #                 new_coms.append(this_partition)
        #             com_index = new_coms.index(this_partition)
        #             if com_index != comm_id:
        #                 comm_id_node_sizes[i] = 0
        #             # print(partition[node])
        #             # print(no_of_isolated_nodes)
        #             # print(partition[node]-no_of_isolated_nodes)
        #             node_colors.append(colors[com_index])
        #             node_colors_idx.append(com_index)
        #         else:
        #             # node_colors.append('darkgray')
        #             node_colors.append(cm.Greys(0.3))
        #             node_colors_idx.append(1000)
        #             no_of_isolated_nodes2 +=1 
        #     print('no_of_isolated_nodes2 = {}'.format(no_of_isolated_nodes2))
        #     # ax.bar(new_coms,new_coms_sizes)
        #     # plt.show()
        #     # plt.close()
        #     legend_elements = []
        #     for ct,c in zip(range(number_of_subsets),colors):
        #         legend_elements.append(mpatches.Patch(facecolor=c,label=ct))
        #     legend_elements.append(mpatches.Patch(facecolor='darkgray',label='isolated nodes id=1000'))

        #     df_embedded['color'] = [matplotlib.colors.rgb2hex(i) for i in node_colors]
        #     os.makedirs(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/', exist_ok=True)
        #     df_embedded.to_csv(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/Cytof_features_df_tsne_embedded_spearman_comm_colors.csv', header=True, index=False)


        # print(node_sizes)
        # print(len(node_sizes))
    
        # print(node_sizes)
        # print(len(node_sizes))
        
        fig, ax = plt.subplots(figsize=(12,10))
        # fig, ax = plt.subplots()
        # if self.cytof_simple_corr_network_color_by == 'univar_corr':
        #     # plt.title('{}, simple corr. network,\n node_size and color intensity=univariate pearson corr. pval (blue=negative, red=positive), edge:= featurewise pearson pval bonf corrected <0.01'.format(timepoint), fontsize= 5)
        #     plt.title('{}, simple corr. network,\n node_size and color intensity=univariate spearman corr. pval (blue=negative, red=positive), edge:= featurewise spearman pval bonf corrected <0.01'.format(timepoint), fontsize= 5)
        # else:
        #     plt.title('{}, simple corr. network,\n node_size=univariate pearson corr. pval , edge:= featurewise pearson pval bonf corrected <0.01, colored by = {}'.format(timepoint, self.cytof_simple_corr_network_color_by), fontsize= 5)
        # plt.title('Timepoint = {}, repeat. 10-fold cv, {},\n size:= {} cytof feat importance., edge:= featuresize spearman corr>0.9'.format(timepoint, 'METs', 'Mean'), fontsize= 10)
        # red_patch = mpatches.Patch(color='red', label='High-level Act Ft')
        # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
        # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
        # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
        # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
        # plt.legend(handles=[red_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, 0.2),
        #        ncol=3,
        #        fontsize=8)
        # nx.draw(G, pos, node_color='black', node_size=20, alpha=0.8, edgecolors='lightgray', linewidths=.4, edge_color='lightgray')
        if len(embedding) != 0:
            pos = nx.get_node_attributes(G,'pos')
        else:
            # pos = nx.spring_layout(G,scale=4)
            # pos = nx.spectral_layout(G)
            # pos = nx.random_layout(G)
            # pos = nx.spectral_layout(G)
            pos = nx.circular_layout(G)
        
        nx.draw(G, pos, ax=ax, node_color=node_colors, labels=labels, font_size=1, node_size=30, alpha=1, edge_color='darkgray', width=0.1)
        # nx.draw(G, pos, ax=ax, node_color='hotpink', labels=labels, font_size=1, node_size=30, alpha=1, edge_color='darkgray', width=0.1)
        # nx.draw(G, pos, ax=ax, node_color=node_colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, edge_color='darkgray', width=0.1)
        # nx.draw(G, pos, ax=ax, node_color=colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, linewidths=0.1, edgecolors='lightgray', edge_color='black', width=0.1)

        # red_patch = mpatches.Patch(color=cmap[0], label='High-level Act Ft')
        # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
        # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
        # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
        # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
        # plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch, black_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, -0.2),
        #        ncol=3,
        #        fontsize=8)

        # plt.title('tSNE cytof features')
        # plt.legend(handles=[red_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, -0.2),
        #        ncol=3,
        #        fontsize=8)
        # plt.xlim((-5, 5))
        # plt.ylim((-5, 5))
        # G2 = nx.Graph()
        # labels2={}
        # for i in range(2):
        #     G2.add_node(i, pos=(100, -15*i))
        #     labels2[i] = ['rho=0.25', 'rho=0.5'][i]
        # node_sizes2 = [label*400 for label in [0.25, 0.5]]
        # nx.draw(G2, nx.get_node_attributes(G2,'pos'), node_color='white', labels=labels2, font_size=8, node_size=node_sizes2, alpha=1, edgecolors='black', linewidths=.4)
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin = 0, vmax=5))
        # sm._A = []
        # if self.cytof_simple_corr_network_color_by == 'univar_corr':
        #     plt.colorbar(scalarMap, shrink=.5)
        # else:
        #     plt.legend(handles=legend_elements,
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(-0.1, -0.1),
        #        ncol=8,
        #        fontsize=4)
        # plt.show()
        plot_save_dir = recovery_feature_predictions_save_folder+'/simple_olink_corr_networks/'
        os.makedirs(plot_save_dir, exist_ok=True)
        # plt.show()
        plt.savefig(plot_save_dir+'Olink_corr_network_colored_tsne_no_legend.pdf', format='pdf', dpi=600)
        # plt.savefig(plot_save_dir+'Cytof_corr_network_colored_by_{0:}_tsne_no_legend_labels_spearman.pdf'.format(self.cytof_simple_corr_network_color_by), format='pdf', dpi=600)
        plt.close()
        fig, ax = plt.subplots(figsize=(12,10))
        # if self.cytof_simple_corr_network_color_by == 'univar_corr':
        #     plt.colorbar(scalarMap, shrink=.5)
        # else:
        plt.legend(handles=legend_elements,
            scatterpoints=1,
            loc='lower left',
            bbox_to_anchor=(-0.1, -0.1),
            ncol=8,
            fontsize=4)

         
        # plt.show()
        # print(plot_save_dir)
        plt.savefig(plot_save_dir+'Olink_corr_network_colored_tsne_only_legend.pdf', format='pdf', dpi=600)
       

        


            
    def predict_recovery_feature_repeated_k_folds_subset_of_features(self, predict_by):

        df_cytof_org = pd.read_csv(self.cytof_filename)
        df_cytof_org = df_cytof_org.iloc[:35,:] # only VAL<54
        # df_recovery_features = pd.read_csv(self.org_slope_filename)
        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        # print(df_cytof_org)
        
        df_cytof_org.fillna(0, inplace=True)

        subsets = []
        if predict_by == 'celltype':
            position_in_col_label = 2
        elif predict_by == 'stim':
            position_in_col_label=1
        elif predict_by == 'marker':
            position_in_col_label=3
        else:
            print('Undefined \'predict_by\' variable')
            sys.exit()
        
        for feat in df_cytof_org.columns.values[1:]:
            subset = feat.split('.')[position_in_col_label]
            if predict_by == 'marker' and feat.split('.')[0]=='freq':
                continue
            if subset not in subsets:
                subsets.append(subset)
        for subset in subsets:
            print('************************** Running for subset {} *******************'.format(subset))
            self.recovery_feature_predictions_subset_save_folder = self.recovery_feature_predictions_save_folder+'predict_by/{}/{}/'.format(predict_by,subset)
            os.makedirs(self.recovery_feature_predictions_subset_save_folder, exist_ok=True)


            corrs_filename = self.recovery_feature_predictions_subset_save_folder+'prediction_correlations.txt'
            pred_corrs = []
            



            print('**************************** repeated_k_folds_recovery')
            cols_to_drop = []
            for col in df_cytof_org.columns.values[1:]:
                if col.split('.')[position_in_col_label] != subset:
                    cols_to_drop.append(col)
            df_cytof = df_cytof_org.drop(cols_to_drop,axis=1)
            

            df_cytof['recovery_feature'] = df_recovery_features[self.target_recovery_feature]
            print(df_cytof)
           
            
            df_repeatedkfold_predictions = pd.DataFrame(columns=['random_state','spearman_rho', 'spearman_pval','pearson_rho', 'pearson_pval'])
            
            df_feat_importance_col = ['iteration', 'fold']
            df_feat_importance_col.extend(df_cytof.columns.values[1:-1])
            df_feat_importance = pd.DataFrame(columns=df_feat_importance_col)
            df_all_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
            # print(df_feat_importance)
            for k in range(1,no_of_reps+1):
                df_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
                kf = KFold(n_splits=no_of_folds, shuffle=True, random_state=k)
                X = df_cytof.iloc[:, 1:-1].values
                y = df_cytof.iloc[:, -1].values
                patient_ids = df_recovery_features['patient_id'].values
                fold = 0
                for train_index, test_index in kf.split(X):
                    print('rep = {}, fold = {}'.format(k,fold))

                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    patient_ids_train, patient_ids_test = patient_ids[train_index], patient_ids[test_index]
            

                    # scalerX = StandardScaler()
                    # scalerX.fit(X_train)
                    # X_train = scalerX.transform(X_train)
                    # X_test = scalerX.transform(X_test)

                    # scalerY = StandardScaler()
                    # scalerY.fit(y_train.reshape(-1, 1))
                    # y_train = scalerY.transform(y_train.reshape(-1, 1))
                    # y_test = scalerY.transform(y_test.reshape(-1, 1))

                    max_iter = 1000000
                    if self.estimator_str == 'RF':
                        clf = RandomForestRegressor(n_estimators=self.RF_no_of_estimators, random_state=k)
                    else: 
                        clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=self.RF_no_of_estimators, random_state=k)
                    
                    # clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))
                    clf.fit(X_train, y_train.flatten())
                    # param_grid=[{'n_estimators':range(10,301,10), 'loss':['linear', 'square', 'exponential'] }]
                    # param_grid = [{'n_estimators': [50,100,200], 'max_depth': [2, 5, 10, None]}]
                    # grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=20, iid=True)
                    # grid.fit(X_train, y_train.flatten())
                
                    # clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=n_estimators, loss=loss)
                    # clf = RandomForestRegressor(n_estimators=200,  n_jobs=20, random_state=k)
                    # clf.fit(X_train,y_train.flatten())
                    
                    # clf = grid.best_estimator_
                    
                
                    y_predicted = clf.predict(X_test)
                    
                    # Y_predicted = clf.predict(X_test_final)
                    # clf_params = clf.get_params()
                
                    importances = clf.feature_importances_
                    
                    
                    df_feat_importance.loc[df_feat_importance.shape[0],:] = [k, fold] + list(importances)
                    
                    # Print the feature ranking
                    # print("Feature ranking:")
                    # for f in range(X.shape[1]):
                    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                    
                    df_fold_predictions = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
                    for i in range(y_test.shape[0]):
                        # df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                        # df_all_predictions.loc[df_all_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                        # df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient_ids_test[i], scalerY.inverse_transform(y_test)[i][0], scalerY.inverse_transform(y_predicted)[i]]
                        df_predictions.loc[df_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                        df_all_predictions.loc[df_all_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                        df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient_ids_test[i], y_test[i], y_predicted[i]]
                    fold += 1
                    # rho, pval = spearmanr(df_fold_predictions['recovery_feature'], df_fold_predictions['predicted'])
                    # print('spearman rho = {} pval = {}'.format(rho, pval))
                    # rho, pval = pearsonr(df_fold_predictions['recovery_feature'], df_fold_predictions['predicted'])
                    # print('pearson rho = {} pval = {}'.format(rho, pval))

                
                
                df_predictions.sort_values(by='patient_id', axis=0, ascending=True, inplace=True)
                df_predictions.reset_index(drop=True, inplace=True)
                
                # df_predictions_filename = '/home/raminf/HipVal/recovery_features/Ramin_BaselinePrediction/Predictions_repeated_10folds_for_Franck_AGD_segmented_recovery_pre_stim_adjusted/RF_no_grid_n_estimators_500/'+feature
                # df_predictions_filename = '/home/raminf/HipVal/recovery_features/Ramin_BaselinePrediction/Predictions_repeated_10folds_for_Franck/old_Clustered_segmented_recovery_pre_stim_adjusted/'+feature
                # os.makedirs(df_predictions_filename, exist_ok=True)
                # df_predictions.to_csv(df_predictions_filename+'/prediction_rep_{}.csv'.format(k), header=True, index=False)

                rho, pval = spearmanr(df_predictions['recovery_feature'], df_predictions['predicted'])
                print('spearman rho = {} pval = {}'.format(rho, pval))
                rho_pearson, pval_pearson = pearsonr(df_predictions['recovery_feature'], df_predictions['predicted'])
                print('pearson rho = {} pval = {}'.format(rho_pearson, pval_pearson))
                df_repeatedkfold_predictions.loc[df_repeatedkfold_predictions.shape[0],:] = [k, rho, pval, rho_pearson, pval_pearson]
            print(df_repeatedkfold_predictions)   
            df_repeatedkfold_predictions.to_csv(self.recovery_feature_predictions_subset_save_folder+'reps_prediction_correlations.csv', header=True, index=False) 
            df_feat_importance.to_csv(self.recovery_feature_predictions_subset_save_folder+'feature_importances.csv', header=True, index=False) 
            df_all_predictions.sort_values(by='patient_id', axis=0, ascending=True, inplace=True)
            df_all_predictions.reset_index(drop=True, inplace=True)
            print(df_all_predictions)
            df_all_predictions.to_csv(self.recovery_feature_predictions_subset_save_folder+'all_predictions.csv', header=True, index=False)
            rho, pval = spearmanr(df_all_predictions['recovery_feature'], df_all_predictions['predicted'])
            print('spearman rho = {} pval = {}'.format(rho, pval))
            pred_corrs.append('+++++++++++++++ all_predictions correlation +++++++++++++++\n')
            pred_corrs.append('spearman rho = {} pval = {}\n'.format(rho, pval))
            rho, pval = pearsonr(df_all_predictions['recovery_feature'], df_all_predictions['predicted'])
            print('pearson rho = {} pval = {}'.format(rho, pval))
            pred_corrs.append('pearson rho = {} pval = {}\n'.format(rho, pval))
            

            patients = list(set(df_all_predictions['patient_id'].values))
            df_med_preds = pd.DataFrame(columns=['patient_id', 'recovery_feature', 'predicted'])
            for patient in patients:
                patient_df = df_all_predictions[df_all_predictions['patient_id']==patient]
                df_med_preds.loc[df_med_preds.shape[0],:] = [patient, patient_df['recovery_feature'].values[0], patient_df['predicted'].median()]
            print(df_med_preds)
            df_med_preds.to_csv(self.recovery_feature_predictions_subset_save_folder+'median_predictions.csv', header=True, index=False)
            rho, pval = spearmanr(df_med_preds['recovery_feature'], df_med_preds['predicted'])
            print('spearman rho = {} pval = {}'.format(rho, pval))
            pred_corrs.append('+++++++++++++++ median_predictions correlation +++++++++++++++\n')
            pred_corrs.append('spearman rho = {} pval = {}\n'.format(rho, pval))
            rho, pval = pearsonr(df_med_preds['recovery_feature'], df_med_preds['predicted'])
            print('pearson rho = {} pval = {}'.format(rho, pval))
            pred_corrs.append('pearson rho = {} pval = {}\n'.format(rho, pval))


            with open(corrs_filename, 'w') as filetowrite:
                filetowrite.writelines(pred_corrs)
                filetowrite.close()
            
            # r2 = r2_score(df_predictions['baseline'], df_predictions['predicted'])
            # print('r2 = ', r2)



    def plot_recovery_feature_predictions_repeated_k_folds_subset_of_features(self, predict_by):

        df_cytof_org = pd.read_csv(self.cytof_filename)
        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        # print(df_cytof_org)
        
        df_cytof_org.fillna(0, inplace=True)

        subsets = []
        if predict_by == 'celltype':
            position_in_col_label = 2
        elif predict_by == 'stim':
            position_in_col_label=1
        elif predict_by == 'marker':
            position_in_col_label=3
        else:
            print('Undefined \'predict_by\' variable')
            sys.exit()
        
        for feat in df_cytof_org.columns.values[1:]:
            subset = feat.split('.')[position_in_col_label]
            if predict_by == 'marker' and feat.split('.')[0]=='freq':
                continue
            if subset not in subsets:
                subsets.append(subset)
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(10,8), constrained_layout=True)
        plot_df = pd.DataFrame(columns=[predict_by, 'pearson_rho', 'log_p_val'])
        for subset in subsets:
            print(subset)
            df_med_preds = pd.read_csv(self.recovery_feature_predictions_save_folder+'predict_by/{}/{}/median_predictions.csv'.format(predict_by,subset))
            # print(df_med_preds)
            rho, pval = pearsonr(df_med_preds['recovery_feature'], df_med_preds['predicted'])
            print('pearson rho = {} pval = {}'.format(rho, pval))
            plot_df =  pd.concat([plot_df, pd.DataFrame(data=[[subset, rho, -1*np.log10(pval)]], columns=[predict_by, 'pearson_rho', 'log_p_val'])])

        ############# last bar: all of them
        df_med_preds = pd.read_csv(self.recovery_feature_predictions_save_folder+'median_predictions.csv')
        # print(df_med_preds)
        rho, pval = pearsonr(df_med_preds['recovery_feature'], df_med_preds['predicted'])
        print('all pearson rho = {} pval = {}'.format(rho, pval))
        plot_df =  pd.concat([plot_df, pd.DataFrame(data=[['All', rho, -1*np.log10(pval)]], columns=[predict_by, 'pearson_rho', 'log_p_val'])])
        #############
        plot_df.plot(kind='bar',x=predict_by, y='pearson_rho', color='mediumblue', ax=ax[0])
        plot_df.plot(kind='bar',x=predict_by, y='log_p_val', color='green', ax=ax[1])
        ax[1].axhline(y=-1*np.log10(0.05), linewidth=4, color='r')
        fig.suptitle('prediction by {}'.format(predict_by))
        # plt.show()
        plt.savefig(self.recovery_feature_predictions_save_folder+'predict_by/{0:}/predictions_plot_by_{0:}.pdf'.format(predict_by), format='pdf', dpi=600)
        print(plot_df)
      

    def plot_cytof_corr_network(self, use_median=True, timepoint='pre_stim_adjusted', dimensionality_reduction='precomputed_tsne'):
        
        # ******work in progress

        df_cytof = pd.read_csv(self.cytof_dict[self.cytof_dataset])
        print(df_cytof)
        
        features = df_cytof.columns.values[1:]
        
        # df_feat_importances = pd.read_csv(predictions_filepath+'{}/feature_importances.csv'.format(baseline_feat))
        recovery_feature_predictions_save_folder = self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/'
        df_feat_importances = pd.read_csv(recovery_feature_predictions_save_folder+'feature_importances.csv')
        print(len(df_feat_importances[features].mean(axis=0).values))
        
        # df_cytof = df_cytof.iloc[:35,:] # only VAL<54
        # print(df_cytof)
        # from sklearn.ensemble import ExtraTreesRegressor
        # from sklearn.experimental import enable_iterative_imputer
        # from sklearn.impute import IterativeImputer, SimpleImputer
        # from sklearn.neighbors import KNeighborsRegressor
        # # imp = IterativeImputer(missing_values=np.nan, sample_posterior=False, 
        # #                          max_iter=10, tol=0.001, 
        # #                          n_nearest_features=4, initial_strategy='median')
        # imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), random_state=42)
        # imp_data = imp.fit_transform(df_cytof.iloc[:,1:-1].values)
        # df_cytof.iloc[:,1:-1] = imp_data
        # print(df_cytof)
    

        if use_median:
            df_feat_importances.loc[df_feat_importances.shape[0],2:] = df_feat_importances[features].median(axis=0).values
        else:
            df_feat_importances.loc[df_feat_importances.shape[0],2:] = df_feat_importances[features].mean(axis=0).values
        
        feature_importances = df_feat_importances.loc[df_feat_importances.shape[0]-1, features]
        # print(feature_importances)
        # feature_importances = feature_importances.sort_values(ascending=True)
        # print(feature_importances.index.values)
        # print(df_cytof)
        new_cytof_df_cols = ['patient_id']
        new_cytof_df_cols.extend(feature_importances.index.values)
        # print(new_cytof_df_cols)
        # df_cytof_sorted_by_importance = df_cytof[new_cytof_df_cols]
        # df_cytof_sorted_by_importance.drop(['patient_id'], axis=1, inplace=True)
        features = df_cytof.columns.values[1:]
        print(df_cytof)
        # print(df_cytof_sorted_by_importance)
       
        
        # print(df_feat_importances)
        print(feature_importances.max())
        
        
        # corr = df_cytof[features].corr(method ='spearman') 
        # # corr = df_cytof[features].corr(method ='pearson') 
        # print(corr)
        for col in df_cytof.columns[1:]:
            df_cytof[col].fillna(0, inplace=True)
        df_cytof.drop(['patient_id'], axis=1, inplace=True)
        corr = df_cytof.corr(method ='pearson').values
        # corr, p_values = spearmanr(df_cytof.values)
        where_are_NaNs = np.isnan(corr)
        print('where_are_NaNs', where_are_NaNs.any())
        
        corr[where_are_NaNs] = 0
        # p_values[where_are_NaNs] = 1
        # print(corr)

        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        # df_cytof['recovery_feat'] = df_recovery_features[self.target_recovery_feature]
        univar_corr = []
        univar_pval = []
        for col in df_cytof.columns:
            c, p = pearsonr(df_cytof[col], df_recovery_features[self.target_recovery_feature])
            univar_corr.append(c)
            univar_pval.append(-1*np.log10(p))
        # print(univar_pval)
        
        # for i in range(len(univar_corr)):
        #     if univar_pval[i] < 0.05:
        #         print(univar_corr[i])
        #         print(univar_pval[i])
        
        # where_are_NaNs_univar_corr = np.isnan(univar_corr)
        # print('where_are_NaNs', where_are_NaNs_univar_corr.any())
        # univar_corr[where_are_NaNs_univar_corr] = 0
        # print(univar_corr)
        
        
        # print(p_values)
        
        
        # distance_matrix = 1 - np.abs(corr)


        # node_sizes = np.arcsinh(feature_importances.values)*4000
        # node_sizes = []
        # for feature_imp in feature_importances.values:
        #     if feature_imp < 0.0005:
        #         node_sizes.append(0.0005*4000)
        #     else:
        #         node_sizes.append(feature_imp*4000)
        # print(min(node_sizes))
        # print(max(node_sizes))

        # print(node_sizes)
        # print(len(node_sizes))
        
        # jet = plt.get_cmap('Reds')
        # cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=5)
        # scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
        # colors = []
        # for i in plot_df['log_pval_mean'].values:
        #     colorVal = scalarMap.to_rgba(i)
        #     colors.append(colorVal)
        
        corr = np.clip(corr, -1, 1)
        distance_matrix = 1 - np.abs(corr)
        # distance_matrix = (1 - corr)/2
        # print(embedded.shape)
        # print(distance_matrix)
        
        print(distance_matrix)
        print(distance_matrix.shape)
        print(np.min(distance_matrix))
        # sys.exit()
        # print(np.sign(np.min(distance_matrix)))
    #    for dm in distance_matrix:
        #    print()
        
        X = df_cytof.values.transpose()
        # print(X)
        scalerX = StandardScaler()
        scalerX.fit(X)
        X_scaled = scalerX.transform(X)
        # print(X_scaled)
        # print(X_scaled.shape)
        # X_embedded = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(X)
        if dimensionality_reduction == 'tsne':
            embedded = TSNE(n_components=2, perplexity=20, random_state=42).fit_transform(X_scaled)
            print(embedded)
            print(embedded.shape)
           
        elif dimensionality_reduction == 'precomputed_tsne':
            embedded = TSNE(n_components=2, metric='precomputed', random_state=42).fit_transform(distance_matrix)
            print(embedded)
            print(embedded.shape)
        # elif dimensionality_reduction == 'umap':
        #     embedded = umap.UMAP(n_neighbors=20, min_dist=0.3, metric='correlation').fit_transform(X_scaled)
        #     print(embedded)
        #     print(embedded.shape)
        else:
            embedded = []

        
        


        

        # self.draw_network_from_scratch(embedded, distance_matrix, None, features, node_sizes, None, timepoint)
        self.draw_network_from_scratch_univariate_corr(embedded, distance_matrix, feature_importances, univar_corr, univar_pval, features, timepoint, df_cytof)



    # def draw_network_from_scratch(self, embedding, distance_matrix, p_values, col_labels, node_sizes, node_colors, timepoint):


    #     G = nx.Graph()

    #     # edge_colors = []
    #     # gray_cm = plt.get_cmap('gray')
    #     # cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=100)
    #     # scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=gray_cm)
    #     labels={}
    #     for i in range(distance_matrix.shape[0]):
    #         if len(embedding) != 0:
    #             G.add_node(i, pos=(embedding[i][0],embedding[i][1]))
    #         else:
    #             G.add_node(i)
            
    #         labels[i] = col_labels[i]
    #         # if node_sizes[i] > 50:
    #         #     labels[i] = col_labels[i]
    #         # else:
    #         #     labels[i] = ''
    #     # print(embedding.shape[0])
    #     print(distance_matrix)
    #     for i in range(distance_matrix.shape[0]):
    #         for j in range(distance_matrix.shape[0]):
    #             # if (j > i) and (euclidean_distances[i][j] < 0.07):
    #             # if (j > i) and (distance_matrix[i][j] < 0.05):
    #             if (j > i) and (distance_matrix[i][j] < 0.20):
    #             # if (j > i) and (p_values[i][j] < 0.01):
    #             # if (j > i) and (p_values[i][j] < 0.001/embedding.shape[0]):
    #             # if (j > i) :
    #                 G.add_edge(i,j)
    
        
    #     if node_colors:
    #         colors = node_colors
    #     else:
    #         colors = 'mediumseagreen'
        
        

        
        

    #     print(colors)
    #     print(len(colors))
    #     print(node_sizes)
    #     print(len(node_sizes))
        
        
    #     plt.title('{}, repeat. 10-fold cv,\n size:= {} cytof feat importance., edge:= featurewise pearson corr>0.8 (show top labels)'.format(timepoint, 'Median'), fontsize= 5)
    #     # plt.title('Timepoint = {}, repeat. 10-fold cv, {},\n size:= {} cytof feat importance., edge:= featuresize spearman corr>0.9'.format(timepoint, 'METs', 'Mean'), fontsize= 10)
    #     # red_patch = mpatches.Patch(color='red', label='High-level Act Ft')
    #     # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
    #     # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
    #     # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
    #     # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
    #     # plt.legend(handles=[red_patch],
    #     #        scatterpoints=1,
    #     #        loc='lower left',
    #     #        bbox_to_anchor=(0.0, 0.2),
    #     #        ncol=3,
    #     #        fontsize=8)
    #     # nx.draw(G, pos, node_color='black', node_size=20, alpha=0.8, edgecolors='lightgray', linewidths=.4, edge_color='lightgray')
    #     if len(embedding) != 0:
    #         pos = nx.get_node_attributes(G,'pos')
    #     else:
    #         # pos = nx.spring_layout(G,scale=4)
    #         # pos = nx.spectral_layout(G)
    #         # pos = nx.random_layout(G)
    #         # pos = nx.spectral_layout(G)
    #         pos = nx.circular_layout(G)
    #     nx.draw(G, pos, node_color=colors, labels=labels, font_size=0.01, node_size=node_sizes, alpha=0.8, linewidths=.2, edgecolors='lightgray', edge_color='gray', width=0.4)

    #     # red_patch = mpatches.Patch(color=cmap[0], label='High-level Act Ft')
    #     # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
    #     # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
    #     # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
    #     # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
    #     # plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch, black_patch],
    #     #        scatterpoints=1,
    #     #        loc='lower left',
    #     #        bbox_to_anchor=(0.0, -0.2),
    #     #        ncol=3,
    #     #        fontsize=8)

    #     # plt.title('tSNE cytof features')
    #     # plt.legend(handles=[red_patch],
    #     #        scatterpoints=1,
    #     #        loc='lower left',
    #     #        bbox_to_anchor=(0.0, -0.2),
    #     #        ncol=3,
    #     #        fontsize=8)
    #     # plt.xlim((-5, 5))
    #     # plt.ylim((-5, 5))
    #     # G2 = nx.Graph()
    #     # labels2={}
    #     # for i in range(2):
    #     #     G2.add_node(i, pos=(100, -15*i))
    #     #     labels2[i] = ['rho=0.25', 'rho=0.5'][i]
    #     # node_sizes2 = [label*400 for label in [0.25, 0.5]]
    #     # nx.draw(G2, nx.get_node_attributes(G2,'pos'), node_color='white', labels=labels2, font_size=8, node_size=node_sizes2, alpha=1, edgecolors='black', linewidths=.4)
    #     # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin = 0, vmax=5))
    #     # sm._A = []
    #     # plt.colorbar(sm, shrink=.5)
    #     # plt.show()
    #     plt.savefig(self.recovery_feature_predictions_save_folder+'Cytof_corr_network_feature_importances_tsne_{0:}_top_labels.pdf'.format(timepoint), format='pdf', dpi=600)



    def draw_network_from_scratch_univariate_corr(self, embedding, distance_matrix, feature_importances, univar_corr, univar_pval, col_labels, timepoint, df_cytof):


        G = nx.Graph()
        G2 = nx.Graph()
        G3 = nx.Graph()
        G4 = nx.Graph()

        eps = 1e-11
        node_colors = []
        gray_cm = plt.get_cmap('hot')
        # print(feature_importances.max())
        # print(-1*np.log10(feature_importances.max()+eps))
        # print(feature_importances.min())
        # print(-1*np.log10(feature_importances.min()+eps))
        
        cNorm  = matplotlib.colors.Normalize(vmin=-0.05, vmax=max(feature_importances.max(),0.08))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=gray_cm)
        for feature_imp in feature_importances.values:
            if feature_imp != 0 and feature_imp < eps:
                print(feature_imp)
                sys.exit()
            node_colors.append(scalarMap.to_rgba(feature_imp))
            # node_colors.append(scalarMap.to_rgba(-1*np.log10(feature_imp+eps)))
            
        
        node_sizes = univar_pval.copy()
        node_sizes2 = univar_pval.copy()
        node_sizes3 = univar_pval.copy()
        node_sizes4 = univar_pval.copy()
        for idx in range(len(node_sizes)):
            # node_sizes[idx] = 50*node_sizes[idx]
            if np.abs(node_sizes[idx]) < 0.1:
                node_sizes[idx] = 0.1*200
                node_sizes2[idx] = 0.1*200
                node_sizes3[idx] = 0.1*200
                node_sizes4[idx] = 0.1*200
            else:
                node_sizes[idx] = np.abs(node_sizes[idx])*200
                node_sizes2[idx] = np.abs(node_sizes2[idx])*200
                node_sizes3[idx] = np.abs(node_sizes3[idx])*200
                node_sizes4[idx] = np.abs(node_sizes4[idx])*200
            print(idx)
            print(feature_importances[idx])
            if feature_importances[idx] <= 0.01:
                node_sizes2[idx] = 0
                node_sizes3[idx] = 0
                node_sizes4[idx] = 0
            elif feature_importances[idx] <= 0.02:
                node_sizes[idx] = 0
                node_sizes3[idx] = 0
                node_sizes4[idx] = 0
            elif feature_importances[idx] <= 0.05:
                node_sizes[idx] = 0
                node_sizes2[idx] = 0
                node_sizes4[idx] = 0
            else:
                node_sizes[idx] = 0
                node_sizes2[idx] = 0
                node_sizes3[idx] = 0
                # print('big')
        # edge_colors = []
        # gray_cm = plt.get_cmap('gray')
        # cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=100)
        # scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=gray_cm)
        labels={}
        
        for i in range(distance_matrix.shape[0]):
            
            G.add_node(i, pos=(embedding[i][0],embedding[i][1]))
            G2.add_node(i, pos=(embedding[i][0],embedding[i][1]))
            G3.add_node(i, pos=(embedding[i][0],embedding[i][1]))
            G4.add_node(i, pos=(embedding[i][0],embedding[i][1]))
            
            labels[i] = col_labels[i]
            # if feature_importances[i] > 0.005:
            #     labels[i] = col_labels[i]
            # else:
            #     labels[i] = ''
            
        # for i in range(distance_matrix.shape[0]):
        #     if len(embedding) != 0:
        #         G.add_node(i, pos=(embedding[i][0],embedding[i][1]))
        #     else:
        #         G.add_node(i)
            
            
            
        # print(embedding.shape[0])
        # print(distance_matrix)
        # for i in range(distance_matrix.shape[0]):
        #     for j in range(distance_matrix.shape[0]):

        #         # if (j > i) and (euclidean_distances[i][j] < 0.07):
        #         # if (j > i) and (distance_matrix[i][j] < 0.05):
        #         if (j > i) and (distance_matrix[i][j] < 0.20):
        #         # if (j > i) and (p_values[i][j] < 0.01):
        #         # if (j > i) and (p_values[i][j] < 0.001/embedding.shape[0]):
        #         # if (j > i) :
        #             G.add_edge(i,j)
        # print(df_cytof)
        all_comparisons = df_cytof.shape[1]*(df_cytof.shape[1]-1)/2
        for i, col1 in enumerate(df_cytof.columns.values):
            for j, col2 in enumerate(df_cytof.columns.values):
                if j > i:
                    rho, pval = pearsonr(df_cytof[col1], df_cytof[col2])
                    if (pval*all_comparisons) < 0.01:
                        G.add_edge(i,j)
                    # else:
                        # print('************************* not edge *************')
    
        
        

        if node_colors:
            colors = node_colors
        else:
            colors = 'mediumseagreen'
        
        

        
        

        # print(colors)
        # print(len(colors))
        print(node_sizes)
        print(node_sizes2)
        print(node_sizes3)
        print(node_sizes4)
        print(len(node_sizes))
        print(np.max(feature_importances))
        
        fig, ax = plt.subplots(figsize=(12,10))
        plt.title('{}, repeat. 10-fold cv,\n node_size=univariate pearson corr. pval, color_intensity:= {} cytof feat importance., edge:= featurewise pearson pval bonf corrected <0.01 (show top labels)'.format(timepoint, 'Median'), fontsize= 5)
        # plt.title('Timepoint = {}, repeat. 10-fold cv, {},\n size:= {} cytof feat importance., edge:= featuresize spearman corr>0.9'.format(timepoint, 'METs', 'Mean'), fontsize= 10)
        # red_patch = mpatches.Patch(color='red', label='High-level Act Ft')
        # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
        # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
        # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
        # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
        # plt.legend(handles=[red_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, 0.2),
        #        ncol=3,
        #        fontsize=8)
        # nx.draw(G, pos, node_color='black', node_size=20, alpha=0.8, edgecolors='lightgray', linewidths=.4, edge_color='lightgray')
        if len(embedding) != 0:
            pos = nx.get_node_attributes(G,'pos')
            pos2 = nx.get_node_attributes(G2,'pos')
            pos3 = nx.get_node_attributes(G3,'pos')
            pos4 = nx.get_node_attributes(G4,'pos')
        else:
            # pos = nx.spring_layout(G,scale=4)
            # pos = nx.spectral_layout(G)
            # pos = nx.random_layout(G)
            # pos = nx.spectral_layout(G)
            pos = nx.circular_layout(G)
        
        nx.draw(G, pos, ax=ax, node_color=colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, edge_color='black', width=0.1)
        nx.draw(G2, pos2, ax=ax, node_color=colors, font_size=1, node_size=node_sizes2, alpha=1, edge_color='black', width=0.1)
        nx.draw(G3, pos3, ax=ax, node_color=colors, font_size=1, node_size=node_sizes3, alpha=1, edge_color='black', width=0.1)
        nx.draw(G4, pos4, ax=ax, node_color=colors, font_size=1, node_size=node_sizes4, alpha=1, edge_color='black', width=0.1)
        # nx.draw(G, pos, ax=ax, node_color=colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, linewidths=0.1, edgecolors='lightgray', edge_color='black', width=0.1)

        # red_patch = mpatches.Patch(color=cmap[0], label='High-level Act Ft')
        # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
        # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
        # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
        # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
        # plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch, black_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, -0.2),
        #        ncol=3,
        #        fontsize=8)

        recovery_feature_predictions_save_folder = self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/'

        # plt.title('tSNE cytof features')
        # plt.legend(handles=[red_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, -0.2),
        #        ncol=3,
        #        fontsize=8)
        # plt.xlim((-5, 5))
        # plt.ylim((-5, 5))
        # G2 = nx.Graph()
        # labels2={}
        # for i in range(2):
        #     G2.add_node(i, pos=(100, -15*i))
        #     labels2[i] = ['rho=0.25', 'rho=0.5'][i]
        # node_sizes2 = [label*400 for label in [0.25, 0.5]]
        # nx.draw(G2, nx.get_node_attributes(G2,'pos'), node_color='white', labels=labels2, font_size=8, node_size=node_sizes2, alpha=1, edgecolors='black', linewidths=.4)
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin = 0, vmax=5))
        # sm._A = []
        # plt.colorbar(scalarMap, shrink=.5)
        # plt.show()
        # plt.savefig(self.recovery_feature_predictions_save_folder+'Cytof_corr_network_univariate_and_feat_importance_tsne_{0:}_top_labels3.pdf'.format(timepoint), format='pdf', dpi=600)
        plt.savefig(recovery_feature_predictions_save_folder+'Cytof_corr_network_univariate_and_feat_importance_tsne_{0:}_top_labels3_labels.pdf'.format(timepoint), format='pdf', dpi=600)
        plt.close()
        # fig, ax = plt.subplots(figsize=(12,10))
        # plt.colorbar(scalarMap, shrink=.5)
        # plt.savefig(self.recovery_feature_predictions_save_folder+'Cytof_corr_network_univariate_and_feat_importance_tsne_{0:}_top_labels3_legend.pdf'.format(timepoint), format='pdf', dpi=600)

    def plot_cytof_simple_corr_network(self, timepoint='pre_stim_adjusted', dimensionality_reduction='precomputed_tsne', color_by='celltype'):

        self.cytof_simple_corr_network_color_by = color_by
        df_cytof = pd.read_csv(self.cytof_dict[self.cytof_dataset])
        print(df_cytof)
        
        features = df_cytof.columns.values[1:]

        recovery_feature_predictions_save_folder = self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/'
        # df_feat_importances = pd.read_csv(predictions_filepath+'{}/feature_importances.csv'.format(baseline_feat))
        df_feat_importances = pd.read_csv(recovery_feature_predictions_save_folder+'feature_importances.csv')
        print(len(df_feat_importances[features].mean(axis=0).values))
        
        
        df_feat_importances.loc[df_feat_importances.shape[0],2:] = df_feat_importances[features].median(axis=0).values
        
        
        feature_importances = df_feat_importances.loc[df_feat_importances.shape[0]-1, features]
        # print(feature_importances)
        feature_importances = feature_importances.sort_values(ascending=True)
        # print(feature_importances.index.values)
        # print(df_cytof)
        # new_cytof_df_cols = ['patient_id']
        # new_cytof_df_cols.extend(feature_importances.index.values)
        # print(new_cytof_df_cols)
        # df_cytof = df_cytof[new_cytof_df_cols]
        features = df_cytof.columns.values[1:]
        # print(df_cytof)
        
       
        # for col in df_cytof.columns[1:]:
        #     df_cytof[col].fillna(0, inplace=True)
        df_cytof = df_cytof.iloc[:35,:] # only VAL<54
        print(df_cytof)
       
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer, SimpleImputer
        from sklearn.neighbors import KNeighborsRegressor
        # imp = IterativeImputer(missing_values=np.nan, sample_posterior=False, 
        #                          max_iter=10, tol=0.001, 
        #                          n_nearest_features=4, initial_strategy='median')
        imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), random_state=42)
        imp_data = imp.fit_transform(df_cytof.iloc[:,1:].values)
        df_cytof.iloc[:,1:] = imp_data
        print(df_cytof)
        
        df_cytof.drop(['patient_id'], axis=1, inplace=True)
        # corr = df_cytof.corr(method ='pearson').values
        corr = df_cytof.corr(method ='spearman').values
        # corr, p_values = spearmanr(df_cytof.values)
        where_are_NaNs = np.isnan(corr)
        print('where_are_NaNs', where_are_NaNs.any())
        
        corr[where_are_NaNs] = 0
        

        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        df_recovery_features = df_recovery_features.iloc[:35,:] # only VAL<54
        # df_cytof['recovery_feat'] = df_recovery_features[self.target_recovery_feature]
        univar_corr = []
        univar_pval = []
        for col in df_cytof.columns:
            # c, p = pearsonr(df_cytof[col], df_recovery_features[self.target_recovery_feature])
            c, p = spearmanr(df_cytof[col], df_recovery_features[self.target_recovery_feature])
            univar_corr.append(c)
            univar_pval.append(-1*np.log10(p))
        df_unicvar_corrs = pd.DataFrame()
        df_unicvar_corrs['cytof_feature'] = df_cytof.columns.values
        # df_unicvar_corrs['pearson_rho'] = univar_corr
        df_unicvar_corrs['spearman_rho'] = univar_corr
        # df_unicvar_corrs['log10_pearson_pval'] = univar_pval
        df_unicvar_corrs['log10_spearman_pval'] = univar_pval
        # df_unicvar_corrs.to_csv(self.recovery_feature_predictions_save_folder+'cytof_vs_recovery_univar_corrs.csv', header=True, index=False)
        df_unicvar_corrs.to_csv(recovery_feature_predictions_save_folder+'cytof_vs_recovery_univar_corrs_spearman.csv', header=True, index=False)
        # print(univar_pval)
        
        # for i in range(len(univar_corr)):
        #     if univar_pval[i] < 0.05:
        #         print(univar_corr[i])
        #         print(univar_pval[i])
        
        # where_are_NaNs_univar_corr = np.isnan(univar_corr)
        # print('where_are_NaNs', where_are_NaNs_univar_corr.any())
        # univar_corr[where_are_NaNs_univar_corr] = 0
        # print(univar_corr)
        
        
        # print(p_values)
        
        
        # distance_matrix = 1 - np.abs(corr)
        
        # jet = plt.get_cmap('Reds')
        # cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=5)
        # scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
        # colors = []
        # for i in plot_df['log_pval_mean'].values:
        #     colorVal = scalarMap.to_rgba(i)
        #     colors.append(colorVal)
        
        corr = np.clip(corr, -1, 1)
        distance_matrix = 1 - np.abs(corr)
        # distance_matrix = (1 - corr)/2
        # print(embedded.shape)
        # print(distance_matrix)
        print(distance_matrix)
        print(distance_matrix.shape)
        
        
        X = df_cytof.values.transpose()
        print(X)
        scalerX = StandardScaler()
        scalerX.fit(X)
        X_scaled = scalerX.transform(X)
        print(X_scaled)
        print(X_scaled.shape)
        # X_embedded = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(X)
        if dimensionality_reduction == 'tsne':
            embedded = TSNE(n_components=2, perplexity=20, random_state=42).fit_transform(X_scaled)
            print(embedded)
            print(embedded.shape)
            
           
        elif dimensionality_reduction == 'precomputed_tsne':
            embedded = TSNE(n_components=2, metric='precomputed', random_state=42).fit_transform(distance_matrix)
            print(embedded)
            print(embedded.shape)
            df_embedded = pd.DataFrame(data=embedded, columns=['d1', 'd2'])
            df_embedded['node_label'] = df_cytof.columns.values
            print(df_embedded)
            # sys.exit()
            # df_embedded.to_csv(self.recovery_feature_predictions_save_folder+'/simple_univariate_cytof_corr_networks/Cytof_features_df_tsne_embedded.csv', header=True, index=False)
            # os.makedirs(self.recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/', exist_ok=True)
            # df_embedded.to_csv(self.recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/Cytof_features_df_tsne_embedded_spearman.csv', header=True, index=False)
            print('here')
            # sys.exit()
        # elif dimensionality_reduction == 'umap':
        #     embedded = umap.UMAP(n_neighbors=20, min_dist=0.3, metric='correlation').fit_transform(X_scaled)
        #     print(embedded)
        #     print(embedded.shape)
        else:
            embedded = []

        print('here')
        


        self.draw_network_from_scratch_simple_corr(embedded, distance_matrix, univar_corr, univar_pval, features, timepoint, df_cytof, df_embedded)

    def draw_network_from_scratch_simple_corr(self, embedding, distance_matrix, univar_corr, univar_pval, col_labels, timepoint, df_cytof, df_embedded):


        G = nx.Graph()

        eps = 1e-11
        node_colors = []
        node_colors_idx = []
        node_sizes = univar_pval

        if self.cytof_simple_corr_network_color_by == 'univar_corr':
            gray_cm = plt.get_cmap('seismic')
            cNorm  = matplotlib.colors.Normalize(vmin=-1*max(univar_pval), vmax=max(univar_pval))
            scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=gray_cm)
            for rho,p in zip(univar_corr,univar_pval):
                if self.target_recovery_feature == 'recovery_slope':
                    if rho < 0:
                        node_colors.append(scalarMap.to_rgba(-1*p))
                    else:
                        node_colors.append(scalarMap.to_rgba(p))
                elif self.target_recovery_feature == 'baseline_pred':
                    if rho >= 0:
                        node_colors.append(scalarMap.to_rgba(-1*p))
                    else:
                        node_colors.append(scalarMap.to_rgba(p))
                else:
                    print('Error Unknown target_recovery_feature')
                    sys.exit()

        elif self.cytof_simple_corr_network_color_by != 'community_detection':
            
            if self.cytof_simple_corr_network_color_by == 'celltype':
                position_in_col_label=2
            elif self.cytof_simple_corr_network_color_by == 'stim':
                position_in_col_label=1
            elif self.cytof_simple_corr_network_color_by == 'marker':
                position_in_col_label=3
            else:
                print('Undefined variable')
                sys.exit()
            subsets = []
            for feat in df_cytof.columns.values:
                subset = feat.split('.')[position_in_col_label]
                if subset not in subsets:
                    if feat.split('.')[0] != 'freq':
                        subsets.append(subset)
            

            
            number_of_subsets= len(subsets)
            # cm_subsection = np.linspace(0, 1, number_of_subsets) 
            # colors = [cm.jet(x) for x in cm_subsection]
            cm_subsection = np.linspace(0.01, 0.75, number_of_subsets) 
            colors = [cm.terrain(x) for x in cm_subsection]
            # cm_subsection = np.linspace(0.01, 0.99, number_of_subsets) 
            # colors = [cm.gnuplot(x) for x in cm_subsection]
                
            # print(colors)
            for feat in df_cytof.columns.values:
                subset = feat.split('.')[position_in_col_label]
                if feat.split('.')[0] != 'freq' or self.cytof_simple_corr_network_color_by != 'marker':
                    node_colors.append(colors[subsets.index(subset)])
                else:
                    node_colors.append('black')
            legend_elements = []
            for ct,c in zip(subsets,colors):
                legend_elements.append(mpatches.Patch(facecolor=c,label=ct))
            # if self.cytof_simple_corr_network_color_by == 'marker':
            #     legend_elements.append(mpatches.Patch(facecolor='black',label='freq'))
      
        
        df_embedded['color'] = [matplotlib.colors.rgb2hex(i) for i in node_colors]
        recovery_feature_predictions_save_folder = self.recovery_feature_predictions_save_folder+self.cytof_dataset+'/'
        os.makedirs(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/', exist_ok=True)
        df_embedded.to_csv(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/Cytof_features_df_tsne_embedded_spearman_comm_colors.csv', header=True, index=False)
        
        # use uivar pval with overal recovery surrogate as size
        # for idx in range(len(node_sizes)):
        #     # node_sizes[idx] = 50*node_sizes[idx]
        #     if np.abs(node_sizes[idx]) < 0.1:
        #         node_sizes[idx] = 0.1*100
        #     else:
        #         node_sizes[idx] = np.abs(node_sizes[idx])*100
        #     if node_sizes[idx] > 300:
        #         node_sizes[idx] = 300
        # use cca weights as size
        node_sizes = list(pd.read_csv(self.main_save_folder+'Rcca/cca-cytof-acti_cytofweights_v1.csv')['V1'].values)
        print(max(node_sizes))
        print(min(node_sizes))
        for idx in range(len(node_sizes)):
            # node_sizes[idx] = 50*node_sizes[idx]
            if np.abs(node_sizes[idx]) < 0.003:
                node_sizes[idx] = 0.003*1000
            else:
                node_sizes[idx] = np.abs(node_sizes[idx])*1000
            # if node_sizes[idx] > 300:
            #     node_sizes[idx] = 300

        labels={}
        for i in range(distance_matrix.shape[0]):
            if len(embedding) != 0:
                G.add_node(i, pos=(embedding[i][0],embedding[i][1]))
            else:
                G.add_node(i)
            
            labels[i] = col_labels[i]
            # labels[i] = ''
            # if feature_importances[i] > 0.005:
            #     labels[i] = col_labels[i]
            # else:
            #     labels[i] = ''
            
        # print(embedding.shape[0])
        print(distance_matrix)
        # for i in range(distance_matrix.shape[0]):
        #     for j in range(distance_matrix.shape[0]):

        #         # if (j > i) and (euclidean_distances[i][j] < 0.07):
        #         # if (j > i) and (distance_matrix[i][j] < 0.05):
        #         if (j > i) and (distance_matrix[i][j] < 0.20):
        #         # if (j > i) and (p_values[i][j] < 0.01):
        #         # if (j > i) and (p_values[i][j] < 0.001/embedding.shape[0]):
        #         # if (j > i) :
        #             G.add_edge(i,j)
        print(df_cytof)

        network_saved = True
        if not network_saved:
            all_comparisons = df_cytof.shape[1]*(df_cytof.shape[1]-1)/2
            print(all_comparisons)
            for i, col1 in enumerate(df_cytof.columns.values):
                print(i)
                for j, col2 in enumerate(df_cytof.columns.values):
                    if j > i:
                        # rho, pval = pearsonr(df_cytof[col1], df_cytof[col2])
                        rho, pval = spearmanr(df_cytof[col1], df_cytof[col2])
                        if (pval*all_comparisons) < 0.05:
                            G.add_edge(i,j)
                        # else:
                            # print('************************* not edge *************')
    

        if network_saved:
            import pickle
            print(pickle.load(open(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/G.pkl','rb')))
            G = pickle.load(open(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/G.pkl','rb'))['G']
        else:
            import pickle
            pickle.dump( {'G':G}, open(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/G.pkl', 'wb' ) )
        
        if self.cytof_simple_corr_network_color_by == 'community_detection':
            comm_id = 1
            print('computing communities')
            no_of_isolated_nodes = 0
            no_of_isolated_nodes2 = 0
            for i in range(len(G.nodes())):
                # print(G.degree[i])
                if G.degree[i] ==0:
                    no_of_isolated_nodes +=1 
            print('no of isolated nodes: {}'.format(no_of_isolated_nodes))
           

            import community

            # fig, ax = plt.subplots(figsize=(5,5))
            # plt.title('{}, simple corr. network,\n node_size=univariate pearson corr. pval , edge:= featurewise pearson pval bonf corrected <0.01, colored by = {}'.format(timepoint, self.cytof_simple_corr_network_color_by), fontsize= 5)
          
            pos = nx.get_node_attributes(G,'pos')
           
            #first compute the best partition
            partition = community.best_partition(G, resolution=1, random_state=42)
            # print(partition.keys())
            # print(partition.values())
            # print(set(partition.values()))
            total_no_of_communities = len(set(partition.values()))
            number_of_subsets= total_no_of_communities - no_of_isolated_nodes
            print('number_of_subsets: {}'.format(number_of_subsets))
            # cm_subsection = np.linspace(0, 1, number_of_subsets) 
            # colors = [cm.jet(x) for x in cm_subsection]
            cm_subsection = np.linspace(0.01, 0.99, number_of_subsets) 
            # cm_subsection = np.linspace(0.01, 0.75, number_of_subsets) 
            colors = [cm.gist_ncar(x) for x in cm_subsection]
            # colors = [cm.terrain(x) for x in cm_subsection]
            # colors = [cm.hsv(x) for x in cm_subsection]
            # cm_subsection = np.linspace(0.01, 0.99, number_of_subsets) 
            # colors = [cm.gnuplot(x) for x in cm_subsection]
                
            # print(colors)
            print(len(colors))
            new_coms = []
            new_coms_sizes = []
            comm_id_node_sizes = node_sizes.copy()
            for i, node in enumerate(range(len(G.nodes()))):
                this_partition = partition[node]
                this_partition_nodes = [nodes for nodes in partition.keys() if partition[nodes] == this_partition]
                if len(this_partition_nodes) > 1:
                    if this_partition not in new_coms:
                        new_coms_sizes.append(len(this_partition_nodes))
                        new_coms.append(this_partition)
                    com_index = new_coms.index(this_partition)
                    if com_index != comm_id:
                        comm_id_node_sizes[i] = 0
                    # print(partition[node])
                    # print(no_of_isolated_nodes)
                    # print(partition[node]-no_of_isolated_nodes)
                    node_colors.append(colors[com_index])
                    node_colors_idx.append(com_index)
                else:
                    # node_colors.append('darkgray')
                    node_colors.append(cm.Greys(0.3))
                    node_colors_idx.append(1000)
                    no_of_isolated_nodes2 +=1 
            print('no_of_isolated_nodes2 = {}'.format(no_of_isolated_nodes2))
            # ax.bar(new_coms,new_coms_sizes)
            # plt.show()
            # plt.close()
            legend_elements = []
            for ct,c in zip(range(number_of_subsets),colors):
                legend_elements.append(mpatches.Patch(facecolor=c,label=ct))
            legend_elements.append(mpatches.Patch(facecolor='darkgray',label='isolated nodes id=1000'))

            df_embedded['color'] = [matplotlib.colors.rgb2hex(i) for i in node_colors]
            os.makedirs(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/', exist_ok=True)
            df_embedded.to_csv(recovery_feature_predictions_save_folder+'simple_univariate_cytof_corr_networks/Cytof_features_df_tsne_embedded_spearman_comm_colors.csv', header=True, index=False)


        # print(node_sizes)
        # print(len(node_sizes))
    
        # print(node_sizes)
        # print(len(node_sizes))
        
        fig, ax = plt.subplots(figsize=(12,10))
        # fig, ax = plt.subplots()
        if self.cytof_simple_corr_network_color_by == 'univar_corr':
            # plt.title('{}, simple corr. network,\n node_size and color intensity=univariate pearson corr. pval (blue=negative, red=positive), edge:= featurewise pearson pval bonf corrected <0.01'.format(timepoint), fontsize= 5)
            plt.title('{}, simple corr. network,\n node_size and color intensity=univariate spearman corr. pval (blue=negative, red=positive), edge:= featurewise spearman pval bonf corrected <0.01'.format(timepoint), fontsize= 5)
        else:
            plt.title('{}, simple corr. network,\n node_size=univariate pearson corr. pval , edge:= featurewise pearson pval bonf corrected <0.01, colored by = {}'.format(timepoint, self.cytof_simple_corr_network_color_by), fontsize= 5)
        # plt.title('Timepoint = {}, repeat. 10-fold cv, {},\n size:= {} cytof feat importance., edge:= featuresize spearman corr>0.9'.format(timepoint, 'METs', 'Mean'), fontsize= 10)
        # red_patch = mpatches.Patch(color='red', label='High-level Act Ft')
        # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
        # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
        # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
        # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
        # plt.legend(handles=[red_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, 0.2),
        #        ncol=3,
        #        fontsize=8)
        # nx.draw(G, pos, node_color='black', node_size=20, alpha=0.8, edgecolors='lightgray', linewidths=.4, edge_color='lightgray')
        if len(embedding) != 0:
            pos = nx.get_node_attributes(G,'pos')
        else:
            # pos = nx.spring_layout(G,scale=4)
            # pos = nx.spectral_layout(G)
            # pos = nx.random_layout(G)
            # pos = nx.spectral_layout(G)
            pos = nx.circular_layout(G)
        
        nx.draw(G, pos, ax=ax, node_color=node_colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, edge_color='darkgray', width=0.1)
        # nx.draw(G, pos, ax=ax, node_color=colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, linewidths=0.1, edgecolors='lightgray', edge_color='black', width=0.1)

        # red_patch = mpatches.Patch(color=cmap[0], label='High-level Act Ft')
        # blue_patch = mpatches.Patch(color=cmap[1], label='Low level Act Ft')
        # green_patch = mpatches.Patch(color=cmap[2], label='Sleep Ft')
        # yellow_patch = mpatches.Patch(color=cmap[3], label='Engineered Ft Primary')
        # black_patch = mpatches.Patch(color=cmap[4], label='Engineered Ft Secondary')
        # plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch, black_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, -0.2),
        #        ncol=3,
        #        fontsize=8)

        # plt.title('tSNE cytof features')
        # plt.legend(handles=[red_patch],
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(0.0, -0.2),
        #        ncol=3,
        #        fontsize=8)
        # plt.xlim((-5, 5))
        # plt.ylim((-5, 5))
        # G2 = nx.Graph()
        # labels2={}
        # for i in range(2):
        #     G2.add_node(i, pos=(100, -15*i))
        #     labels2[i] = ['rho=0.25', 'rho=0.5'][i]
        # node_sizes2 = [label*400 for label in [0.25, 0.5]]
        # nx.draw(G2, nx.get_node_attributes(G2,'pos'), node_color='white', labels=labels2, font_size=8, node_size=node_sizes2, alpha=1, edgecolors='black', linewidths=.4)
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin = 0, vmax=5))
        # sm._A = []
        # if self.cytof_simple_corr_network_color_by == 'univar_corr':
        #     plt.colorbar(scalarMap, shrink=.5)
        # else:
        #     plt.legend(handles=legend_elements,
        #        scatterpoints=1,
        #        loc='lower left',
        #        bbox_to_anchor=(-0.1, -0.1),
        #        ncol=8,
        #        fontsize=4)
        # plt.show()
        plot_save_dir = recovery_feature_predictions_save_folder+'/simple_univariate_cytof_corr_networks/'
        os.makedirs(plot_save_dir, exist_ok=True)
        # plt.show()
        plt.savefig(plot_save_dir+'Cytof_corr_network_colored_by_{0:}_tsne_no_legend_labeled.pdf'.format(self.cytof_simple_corr_network_color_by), format='pdf', dpi=600)
        # plt.savefig(plot_save_dir+'Cytof_corr_network_colored_by_{0:}_tsne_no_legend_labels_spearman.pdf'.format(self.cytof_simple_corr_network_color_by), format='pdf', dpi=600)
        plt.close()
        fig, ax = plt.subplots(figsize=(12,10))
        if self.cytof_simple_corr_network_color_by == 'univar_corr':
            plt.colorbar(scalarMap, shrink=.5)
        else:
            plt.legend(handles=legend_elements,
               scatterpoints=1,
               loc='lower left',
               bbox_to_anchor=(-0.1, -0.1),
               ncol=8,
               fontsize=4)

         
        # plt.show()
        print(plot_save_dir)
        plt.savefig(plot_save_dir+'Cytof_corr_network_colored_by_{0:}_tsne_only_legend.pdf'.format(self.cytof_simple_corr_network_color_by), format='pdf', dpi=600)
       

        

    def classification_recovery_slope(self):
        df_cytof = pd.read_csv(self.cytof_filename)
        # df_recovery_features = pd.read_csv(self.org_slope_filename)
        df_recovery_features = pd.read_csv(self.recovery_feature_predictions_save_folder+'clustered_surrogates/3/median_predictions_gt_3_clusters.csv')
        print(df_recovery_features)
        
        df_cytof.fillna(0, inplace=True)
        

        df_cytof['class'] = df_recovery_features['cluster_labels']
        df_cytof = df_cytof[(df_cytof['class'] == 0) | (df_cytof['class'] == 2)]
        df_cytof['class'] = np.interp(df_cytof['class'], (0, 2), (0, 1))

        # print(df_cytof)
        
        
        # assert df_cytof.shape[0] == df_recovery_features.shape[0], 'rows missmatch!'
    
        df_predictions = pd.DataFrame(columns=['Subject', 'class', 'predicted'])

        for sub in df_cytof['patient_id'].values:
            train_df = df_cytof[df_cytof['patient_id'] != sub]
            # print('Train df = ')
            # print(train_df)
            test_df = df_cytof[df_cytof['patient_id'] == sub]

            # X_train = , X_test, y_train, y_test = train_test_split(X, y, test_size=0.02)
            X_train = train_df.iloc[:, 1:-1].values
            y_train = train_df['class'].values

            X_test = test_df.iloc[:, 1:-1].values
            y_test = test_df['class'].values

            # selected_features = []
            # for feat in range(X_train.shape[1]):
            #     rho, pval = spearmanr(X_train[:,feat], y_train)
            #     # print('rho = ', rho)
            #     # print('pval = ', pval)
            #     if abs(pval) < 0.1:
            #         # print('sub = ', sub)
            #         # print('rho = ', rho)
            #         # print('pval = ', pval)
            #         # print('####################')
            #         selected_features.append(feat)
            # print(selected_features)
            # # print(X_train[:,10])
            # # print(X_train[:,selected_features])
            # X_train = X_train[:,selected_features]
            # X_test = X_test[:,selected_features]
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            # model = DecisionTreeClassifier()
            # param_grid=[{'max_depth':[2, 5, 10, None], 'criterion':['gini', 'entropy'], 'max_features':[None, 'sqrt', 'log2'] }]
            # # param_grid = [{'alpha': np.logspace(-2,0.2,20), 'fit_intercept': [True, False]}]
            # # param_grid = [{'n_estimators': [50,100,200], 'max_depth': [2, 5, 10]}]
            # clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=100)
            # clf.fit(X_train, y_train.flatten())
        
            clf = DecisionTreeClassifier()
            clf = RandomForestClassifier(n_estimators=700, max_depth=2)
            clf.fit(X_train, y_train)

            y_predicted = clf.predict(X_test)
            print('test predicted label size is {}'.format(len(y_predicted)))
            print('run ended')
            
            print('********************************')
            print('Sub = ', sub)
            print('Y_test = ', y_test)
            print('y_pred = ', y_predicted)
            print('********************************')
            df_predictions.loc[df_predictions.shape[0],:] = [sub, y_test[0], y_predicted[0]]

        print(df_predictions)
        print('==================================')
        y = list(df_predictions['class'].values)
        y_predicted = list(df_predictions['predicted'].values)
        # y = [0, 2, 1]
        # y_predicted = [2, 0, 1]
        # print(len(y_predicted))
        from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score, f1_score, recall_score, precision_score, confusion_matrix
        print('f_score per class, loocv = ', f1_score(y, y_predicted, average=None))
        print('f_score weighted avg, loocv = ', f1_score(y, y_predicted, average='weighted'))
        print('recall, loocv = ', recall_score(y, y_predicted, average='weighted'))
        print('precision, loocv = ', precision_score(y, y_predicted, average='weighted'))
        print('confusion_matrix, loocv = ', confusion_matrix(y, y_predicted))
        print('roc_auc_score, loocv = ', roc_auc_score(y, y_predicted))



    def univ_corr_clinical_vs_surrogate(self):

        df_clinical_report = pd.read_csv(self.clinical_report_filename)
        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        # df_recovery_features = df_recovery_features.iloc[:35,:] # VAL<45
        print(df_clinical_report)
        # df_clinical_report['stress_score'].fillna(df_clinical_report['stress_score'].mean(),inplace=True)

        # df_clinical_report = df_clinical_report.iloc[:35,:] # VAL<45
        print(df_recovery_features)

        
        # df_univ_corrs = pd.DataFrame(columns=['clinical_measure', 'recovery_slope_pearson_rho', 'recovery_slope_pearson_pval', 'recovery_slope_spearman_rho', 'recovery_slope_spearman_pval', 'baseline_pred_pearson_rho', 'baseline_pred_pearson_pval', 'baseline_pred_spearman_rho', 'baseline_pred_spearman_pval'])
        df_univ_corrs = pd.DataFrame(columns=['clinical_measure', 'baseline_pred_pearson_rho', 'baseline_pred_pearson_pval', 'baseline_pred_spearman_rho', 'baseline_pred_spearman_pval'])
        # df_univ_corrs = pd.DataFrame(columns=['clinical_measure','baseline_pred_spearman_rho', 'baseline_pred_spearman_pval'])
        for clinical_feat in df_clinical_report.columns[1:]:
            row = [clinical_feat]
            # for recovery_feature in ['recovery_slope', 'baseline_pred']:
            for recovery_feature in ['baseline_pred']:
                # print(clinical_feat, recovery_feature)
                c_pearson, p_pearsonr = pearsonr(df_clinical_report[clinical_feat], df_recovery_features[recovery_feature])
                c_spearman, p_spearman = spearmanr(df_clinical_report[clinical_feat], df_recovery_features[recovery_feature])
                row.extend([c_pearson, p_pearsonr, c_spearman, p_spearman])
                # row.extend([c_spearman, p_spearman])
            df_univ_corrs.loc[df_univ_corrs.shape[0], :] = row
        print(df_univ_corrs)
        os.makedirs(self.clinical_vs_surrogate_univ_corr_save_folder, exist_ok=True)
        df_univ_corrs.to_csv(self.clinical_vs_surrogate_univ_corr_save_folder+'df_univ_corrs.csv',header=True, index=False)
        
        


        # print(df_univ_corrs)
        # df_univ_corrs.to_csv(self.clinical_vs_surrogate_univ_corr_save_folder+'clinical_vs_surrogate_univ_corrs.csv', header=True, index=False)
    


        # plots = ['srs_auc',
        #         'srs_auc_new',
        #         'srs_sum_pos',
        #         'auc_42_days',
        #         'srs_recv_day']
        # for plot in plots:
        #     fig, ax = plt.subplots(figsize=(6,6))
        #     df_plot = pd.DataFrame(data=df_clinical_report[plot].values, columns=[plot])
        #     df_plot['baseline_pred'] = df_recovery_features['baseline_pred']
        #     print(df_plot)
        #     sns.regplot(x=plot, y='baseline_pred', data=df_plot, ax=ax, robust=True, color='black', x_ci='ci', ci=90)
            
        #     df_temp = df_univ_corrs[df_univ_corrs['clinical_measure']==plot].reset_index(drop=True)
        #     print(df_temp)
        #     # print(df_temp.loc[0,plot[1]+'_spearman_rho'])
        #     # print(df_temp.loc[0,plot[1]+'_spearman_pval'])
        #     # print(df_temp.loc[0,plot[1]+'_pearson_rho'])
        #     # print(df_temp.loc[0,plot[1]+'_pearson_pval'])
        #     ax.set_xlabel(plot+' clinical measure')
        #     ax.set_ylabel('recovery surrogate')
        #     ax.set_title('Spearman rho= {:.2f}\npval = {:.5f}'.format(
        #         df_temp.loc[0,'baseline_pred_spearman_rho'],
        #         df_temp.loc[0,'baseline_pred_spearman_pval']))
        #     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        #     # place a text box in upper left in axes coords
        #     ax.text(0.05, 0.95, 'Spearman rho= {:.2f}\npval = {:.5f}'.format(
        #         df_temp.loc[0,'baseline_pred_spearman_rho'],
        #         df_temp.loc[0,'baseline_pred_spearman_pval']), transform=ax.transAxes, fontsize=14,
        #             verticalalignment='top', bbox=props)
            
        #     # plt.show()
        #     plt.savefig(self.clinical_vs_surrogate_univ_corr_save_folder+'{}_vs_{}.pdf'.format('baseline_pred', plot), format='pdf', dpi=600)
        #     plt.close()

        # # bins = pd.IntervalIndex.from_tuples([(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)])
        # new_col= []
        # for item in df_clinical_report['d_to_disch']:
        #     if item <= 1.5:
        #         new_col.append(1)
        #     elif item <= 2.5:
        #         new_col.append(2)
        #     else:
        #         new_col.append(3)
        # df_recovery_features['d_to_disch'] = new_col
        # print(df_recovery_features)
        # fig, ax = plt.subplots(figsize=(10,5))
        # sns.boxplot(x='d_to_disch', y='baseline_pred',data=df_recovery_features, ax=ax,  palette=['white', 'white'],showfliers=False)
        # sns.swarmplot(x="d_to_disch", y="baseline_pred", data=df_recovery_features, size=5, color=".3", linewidth=0, ax=ax)
        # plt.savefig(self.clinical_vs_surrogate_univ_corr_save_folder+'boxplot_baseline_pred_vs_d_to_disch.pdf', format='pdf', dpi=600)


        # plt.show()
    
    def univ_corr_clinical_vs_preop(self, report_avg=True):

        df_clinical_report = pd.read_csv(self.clinical_report_filename)
        df_actigraph = pd.read_csv(self.actigraph_filename)
        print(df_clinical_report)
        print(df_actigraph)
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]
        df_pre_op_actigraph = pd.DataFrame(columns=df_actigraph.columns.values[:-1])
        for patient in patients:
            
            print('patient = {}'.format(patient))
                
            if patient < 10:
                patient_id = 'VAL0{}.agd'.format(patient)
            else:
                patient_id = 'VAL{}.agd'.format(patient)

            df = df_actigraph[(df_actigraph['Day'] < 0) & (df_actigraph['Day'] >=self.first_day) & (df_actigraph['Filename'] == patient_id)]
            # df = df_actigraph[(df_actigraph['Day'] < 0) & (df_actigraph['Filename'] == patient_id)]
            # print(df)
            df.drop(['Filename', 'Day'], axis=1, inplace=True)
            # print(df.median(axis=0))
            row = [patient]
            if report_avg:
                row.extend(df.mean(axis=0).values)
            else: # calculate median
                row.extend(df.median(axis=0).values)
            df_pre_op_actigraph = pd.concat([df_pre_op_actigraph, pd.DataFrame([row], columns=df_actigraph.columns.values[:-1])], axis=0)
        df_pre_op_actigraph.reset_index(drop=True, inplace=True)
        print(df_pre_op_actigraph)
       
        
        df_corr = pd.DataFrame(columns=['col_clinical', 'col_actigraph', 'rho', 'pval'])
        for col_clinical in df_clinical_report.columns[1:]:
            for col_actigraph in df_pre_op_actigraph.columns[1:]:
                c, p = spearmanr(df_clinical_report[col_clinical], df_pre_op_actigraph[col_actigraph])
                df_corr = pd.concat([df_corr, pd.DataFrame([[col_clinical, col_actigraph,c,p]], columns=['col_clinical', 'col_actigraph', 'rho', 'pval'])], axis=0)
        
    
        os.makedirs(self.clinical_vs_preop_univ_corr_save_folder, exist_ok=True)
        print(df_corr)
        df_corr.to_csv(self.clinical_vs_preop_univ_corr_save_folder+'clinical_vs_preop_univ_spearmanr.csv', header=True, index=False)
        # fig, ax = plt.subplots(figsize=(12,10))
        # for x, y in zip(df_pre_op_actigraph['Sed'], df_clinical_report['cirs_severity']):
        #     ax.scatter(x,y)
        # plt.show()
        
    def univ_corr_surrogate_vs_preop(self, report_avg=True):

        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'Opt_breaking_points/opt_breaking_points.csv')
        
        df_actigraph = pd.read_csv(self.actigraph_filename)
        print(df_recovery_features)
        print(df_actigraph)
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]
        df_pre_op_actigraph = pd.DataFrame(columns=df_actigraph.columns.values[:-1])
        for patient in patients:
            
            print('patient = {}'.format(patient))
                
            if patient < 10:
                patient_id = 'VAL0{}.agd'.format(patient)
            else:
                patient_id = 'VAL{}.agd'.format(patient)

            df = df_actigraph[(df_actigraph['Day'] < 0) & (df_actigraph['Day'] >=self.first_day) & (df_actigraph['Filename'] == patient_id)]
            # df = df_actigraph[(df_actigraph['Day'] < 0) & (df_actigraph['Filename'] == patient_id)]
            # print(df)
            df.drop(['Filename', 'Day'], axis=1, inplace=True)
            # print(df.median(axis=0))
            row = [patient]
            if report_avg:
                row.extend(df.mean(axis=0).values)
            else: # calculate median
                row.extend(df.median(axis=0).values)
            df_pre_op_actigraph = pd.concat([df_pre_op_actigraph, pd.DataFrame([row], columns=df_actigraph.columns.values[:-1])], axis=0)
        df_pre_op_actigraph.reset_index(drop=True, inplace=True)
        print(df_pre_op_actigraph)
       
        
        df_corr = pd.DataFrame(columns=['col_actigraph', 'rho', 'pval'])
        for col_actigraph in df_pre_op_actigraph.columns[1:]:
            c, p = spearmanr(df_recovery_features[self.target_recovery_feature], df_pre_op_actigraph[col_actigraph])
            df_corr = pd.concat([df_corr, pd.DataFrame([[col_actigraph,c,p]], columns=['col_actigraph', 'rho', 'pval'])], axis=0)
        
    
        os.makedirs(self.surrogate_vs_preop_univ_corr_save_folder, exist_ok=True)
        print(df_corr)
        df_corr.to_csv(self.surrogate_vs_preop_univ_corr_save_folder+'surrogate_{}_vs_preop_univ_spearmanr.csv'.format(self.target_recovery_feature), header=True, index=False)
        for feat in ['NSBS', 'NSBrE', 'NSBO', 'NSBE', 'NSBrS','Axis1MC']:
            fig, ax = plt.subplots(figsize=(5,5))
            for x, y in zip(df_pre_op_actigraph[feat], df_recovery_features[self.target_recovery_feature]):
                ax.scatter(x,y, c='black')
                plt.xlabel(feat)
                plt.ylabel(self.target_recovery_feature)
            # plt.show()
            plt.savefig(self.surrogate_vs_preop_univ_corr_save_folder+'{}_vs_{}.jpg'.format(self.target_recovery_feature,feat), format='jpg', pad_inches=1)
            plt.close()
            


    def extract_recovery_feature_per_actigraph_feature(self):

        def baseline_recovery(patient_id, actigraph_features_decline):

            
            df = pd.read_csv(self.actigraph_filename)
            df = df[df['Filename']==patient]
            actigraph_features = df.columns.values[1:-1]
            
            # print(actigraph_features)
            return_dict = {'patient_id':patient_id}
            for i, act_feat in enumerate(actigraph_features):
                print('i= {}'.format(i))
                df_feat = df[['Day',act_feat]]
                # print(df_feat)
                # print(df_feat[(df_feat['Day']<0) & (df_feat['Day']>=self.first_day)])
                if actigraph_features_decline[i]:
                    df_feature_threshold = df_feat[(df_feat['Day']<0) & (df_feat['Day']>=self.first_day)][act_feat].quantile(0.25)
                else:
                    df_feature_threshold = df_feat[(df_feat['Day']<0) & (df_feat['Day']>=self.first_day)][act_feat].quantile(0.75)
                # print(df_feature_threshold)
                df_feat_post = df_feat[(df_feat['Day'] > 0) & (df_feat['Day'] < 41)]
                # df_feat_post = df_feat[df_feat['Day']>0]
                df_feat_post.reset_index(drop=True, inplace=False)
                for _, row in df_feat_post.iterrows():
                    
                    return_dict.update({act_feat:row['Day']})
                    if actigraph_features_decline[i]:
                        if row[act_feat] >= df_feature_threshold:
                            print(row[act_feat])
                            print(row['Day'])
                            break
                    else:
                        if row[act_feat] <= df_feature_threshold:
                            print(row[act_feat])
                            print(row['Day'])
                            break

            # print(return_dict)  
            
            return return_dict
            

        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]

        df = pd.read_csv(self.actigraph_filename)
        df = df[(df['Filename'] != 'VAL63.agd') & (df['Filename'] != 'VAL66.agd')]
        actigraph_features = df.columns.values[1:-1]
        
        df_patient_average = pd.DataFrame(columns=df.columns.values[1:])

        for day in range(-5,41):
            if day==0:
                continue
            df_day = df[df['Day']==day]
            # print(df_day)
            # print(df_day.mean(axis=0))
            df_patient_average.loc[len(df_patient_average),:] = df_day.mean(axis=0).values
        print(df_patient_average)
        actigraph_features_decline = [True for _ in actigraph_features]
        for i, act_feat in enumerate(actigraph_features):
            fig, ax = plt.subplots(figsize=(5,5))
            ax.plot(df_patient_average['Day'], df_patient_average[act_feat])
            ax.set_title(act_feat)
            plt.show()
            inp = input('Decline? ')
            while(True):
                if inp == '1':
                    actigraph_features_decline[i] = True
                    break
                elif inp == '0':
                    actigraph_features_decline[i] = False
                    break
                inp = input('Decline?')
        print(actigraph_features_decline)
       
        output_df = pd.DataFrame()
        for patient in [i for i in range(1,76) if i not in excluded_file_no]:
                if patient < 10:
                    patient = 'VAL0{}.agd'.format(patient)
                else:
                    patient = 'VAL{}.agd'.format(patient)
                row = baseline_recovery(patient, actigraph_features_decline)
                
                output_df = pd.concat([output_df, pd.DataFrame(data=np.asarray(list(row.values())).reshape(1,-1), columns=list(row.keys()))])
            
        print(output_df)
        if not os.path.exists(self.extract_recovery_feature_save_folder):
            os.makedirs(self.extract_recovery_feature_save_folder)
        output_df.to_csv(self.extract_recovery_feature_save_folder+'Per_feature_baseline_recovery.csv', header=True, index=False)



if __name__ == "__main__":


    # org_slope_filename = '/home/raminf/HipVal/recovery_features/New Ramin/AGD_AdaBoost_PerUser_BaggingVal_newRun_preIncluded4prediction_rerun/CSV/Opt_breaking_points/opt_breaking_points_new.csv'

    actigraph_filename = './data/wearable/Activity_Sleep_UserStandardized.csv'

    # estimator_str = 'RF'
    
    # RF_no_of_estimators = 100
    # no_of_folds = 10
    # no_of_reps = 30

    # horizontal_recovered = True
    # first_day = -5

    main_save_folder = None # set the main save folder otherwise it will generate a new folder
    # main_save_folder = '/home/raminf/HipVal/recovery_features/Ramin_slope_prediction/output/RF_Pre_stim_adjusted_100estimators_10folds_30reps_imputed_DayAdded_OutlierRemoved_first2days_removed_UserStandardized_2020-05-12_17-28-34/' # final with penalization VAL<54


    # target_recovery_feature = 'baseline_pred' 
    # target_recovery_feature_type = '0.25_quantile' 
    
    
    cytof_filename = './data/immune/HipValidation_cytof_Pre_stim_baseline_adjusted_all_plates.csv.csv'



    # cytof_dataset = 'Pre_stim_adjusted'
   

    olink_filename = './data/proteomics/proteomics/olink_Pre.csv'

    clinical_report = '/data/clinical/Demographics/clinical.csv'


    recovery_prediction_pipeline = recovery_prediction_pipeline(main_save_folder, actigraph_filename, cytof_filename, olink_filename, clinical_report)

    # recovery_prediction_pipeline.predict_single_feature_personalized_activity_model_parallel_iterations_k_fold_per_patient()
    # recovery_prediction_pipeline.predict_single_feature_personalized_activity_model_boxplot_median_k_fold_per_patient()
    # recovery_prediction_pipeline.predict_single_feat_personalized_activity_model_rmse_patient_median()
    # recovery_prediction_pipeline.plot_predict_single_feat_vs_multi_personalized_activity_model_rmse_patient_median()

    # recovery_prediction_pipeline.predict_personalized_activity_model_parallel_iterations_k_fold_per_patient()
    # recovery_prediction_pipeline.predict_personalized_activity_model_boxplot_median_k_fold_per_patient()
    # recovery_prediction_pipeline.predict_personalized_activity_model_boxplot_k_fold_each_patient()
    # recovery_prediction_pipeline.predict_personalized_activity_model_rmse_patient_median()

    # recovery_prediction_pipeline.extract_recovery_feature()

    # recovery_prediction_pipeline.plot_personalized_activity_model_lineplot_median_k_fold_per_patient()
    # recovery_prediction_pipeline.predict_personalized_activity_model_corr_network()

    # recovery_prediction_pipeline.predict_recovery_feature_repeated_k_folds()
    # recovery_prediction_pipeline.plot_recovery_feature_scatter_plot_predictions()

    # recovery_prediction_pipeline.predict_recovery_feature_repeated_k_folds_proteomics()
    # recovery_prediction_pipeline.plot_recovery_feature_scatter_plot_predictions_proteomics()
    # recovery_prediction_pipeline.plot_cytof_simple_proteomics_network()

    # recovery_prediction_pipeline.plot_cytof_corr_network()
    # recovery_prediction_pipeline.plot_cytof_simple_corr_network(color_by='stim')

    # recovery_prediction_pipeline.predict_recovery_feature_repeated_k_folds_subset_of_features(predict_by='marker')    
    # recovery_prediction_pipeline.plot_recovery_feature_predictions_repeated_k_folds_subset_of_features(predict_by='marker') 

    # recovery_prediction_pipeline.plot_clustered_personalized_activity_model_lineplot_median_k_fold_per_patient(CI=0.9, n_c=2)
    # # recovery_prediction_pipeline.classification_recovery_slope()
    

    # recovery_prediction_pipeline.univ_corr_clinical_vs_surrogate() 
    # recovery_prediction_pipeline.univ_corr_clinical_vs_preop(report_avg=True) 
    # recovery_prediction_pipeline.univ_corr_surrogate_vs_preop(report_avg=True) 

    # recovery_prediction_pipeline.extract_recovery_feature_per_actigraph_feature()



    
   


