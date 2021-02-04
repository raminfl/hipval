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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, ParameterGrid, cross_validate
from scipy import interpolate, stats
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import cm
import networkx as nx



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
        self.recovery_feature_predictions_save_folder = self.main_save_folder+'recovery_predictions/'
        self.missing_ids = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
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


    def repeated_k_folds_per_patient_single_feature(self, output_array,single_feat,patient,X,Y):
        

        for k in range(1, self.no_of_reps+1):
            df_predictions = pd.DataFrame(columns=['single_feat', 'patient_id', 'ground_truth', 'predicted'])
            
            kf = KFold(n_splits=self.no_of_folds, shuffle=True, random_state=k)
            
            fold = 0
            X = X.reshape(-1,1)
            for train_index, test_index in kf.split(X):
            
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
            
                max_iter = 1000000
                clf = RandomForestRegressor(n_estimators=self.RF_no_of_estimators, random_state=k)
                clf.fit(X_train, y_train.flatten())
            
                y_predicted = clf.predict(X_test)
               
                df_fold_predictions = pd.DataFrame(columns=['single_feat', 'patient_id', 'ground_truth', 'predicted'])
                for y_ground, y_pred in zip(y_test.flatten(), y_predicted):
                    df_predictions.loc[df_predictions.shape[0],:] = [single_feat, patient, y_ground, y_pred]
                    df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [single_feat, patient, y_ground, y_pred]
                
                fold += 1
                
                rho, pval = spearmanr(df_fold_predictions['ground_truth'], df_fold_predictions['predicted'])
            
            df_predictions.sort_values(by='ground_truth', axis=0, ascending=True, inplace=True)
            df_predictions.reset_index(drop=True, inplace=True)
            df_predictions_filename = self.single_feat_actigraph_predictions_save_folder+'{}/patient_{}/'.format(single_feat, patient)
            os.makedirs(df_predictions_filename, exist_ok=True)
            df_predictions.to_csv(df_predictions_filename+'prediction_rep_{}.csv'.format(k), header=True, index=False)

            rho, pval = spearmanr(df_predictions['ground_truth'], df_predictions['predicted'])
            output_array.append([single_feat, patient, rho, pval])



    def train_univariate_accelerometry_models(self):
        # for comparison of univariate vs multivariate RF
        
        df_original = pd.read_csv(self.actigraph_filename)

        output_cols = ['single_feat', 'patient_id', 'rho', 'pval']
        manager = multiprocessing.Manager()
        output_array = manager.list()
        features = df_original.columns.values[1:-1]

        patients = [i for i in range(1,76) if i not in self.missing_ids]

        numberOfThreads = 10 # change for your CPU
        
        
        for single_feat in features:
            print('single feat = {}'.format(single_feat))
            jobs = []
            for patient in patients:
                    
                if patient < 10:
                    patient_id = 'VAL0{}.agd'.format(patient)
                else:
                    patient_id = 'VAL{}.agd'.format(patient)

                df = df_original[(df_original['Day'] > 0) & (df_original['patient_id'] == patient_id)]
                
                df.drop(['patient_id'], axis=1,inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.dropna(inplace=True)

                label = 'Day'

                X = df[single_feat].values
                Y = df[label].values
            
                p = multiprocessing.Process(target=self.repeated_k_folds_per_patient_single_feature, args=(output_array,single_feat,patient,X,Y,))
                jobs.append(p)
            for i in self.chunks(jobs,numberOfThreads):
                for j in i:
                    j.start()
                for j in i:
                    j.join()
        
            output_df = pd.DataFrame(np.array(output_array), columns=output_cols)
            output_df.to_csv(self.single_feat_actigraph_predictions_save_folder+single_feat+'/predictions_corrs.csv', index=False, header=True)

    def boxplot_univariate_accelerometry_models(self):

        # box plot for R and p values per feature
    
        df_actigraph = pd.read_csv(self.actigraph_filename)
        features = df_actigraph.columns.values[1:-1]
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]

        for single_feat in features:
            
            pred_filename = self.single_feat_actigraph_predictions_save_folder+single_feat+'/predictions_corrs.csv'
            df_original = pd.read_csv(pred_filename)
            df_original['rho'] = df_original['rho'].clip(lower=0)
            df_original['log10_pval'] = -1*np.log10(df_original['pval'].values)
            plot_df = pd.DataFrame(columns=['patient_id', 'rho', 'log10_pval'])
            for patient in patients:

                df = df_original[df_original['single_feat'] == single_feat]
                df = df[df['patient_id'] == patient]
                plot_df.loc[plot_df.shape[0],:] = [patient, df['rho'].median(), df['log10_pval'].median()]
                
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(5,5), constrained_layout=True) 
            sns.boxplot(y="rho", data=plot_df, showfliers = False, ax=ax[0], color="royalblue")
            sns.swarmplot(y="rho", data=plot_df, color=".25", ax=ax[0])
            sns.boxplot(y="log10_pval", data=plot_df, showfliers = False, ax=ax[1], color="limegreen")
            sns.swarmplot(y="log10_pval", data=plot_df, color=".25", ax=ax[1])
            plt.tight_layout()
            ax[0].set_ylim([0,1])
            ax[1].set_ylim([0.1,20])
            ax[1].axhline(y=-1*np.log10(0.05), linewidth=4, color='r')
            plt.savefig(self.single_feat_actigraph_predictions_save_folder+single_feat+'/boxplot_med_corrs.jpg', format='jpg', pad_inches=1)
            plot_df.to_csv(self.single_feat_actigraph_predictions_save_folder+single_feat+'/med_predictions_corrs.csv', header=True, index=False) 
            
    def RMSE_univariate_accelerometry_models(self):
        # get the rmse and mae for each patient
        df_actigraph = pd.read_csv(self.actigraph_filename)
        features = df_actigraph.columns.values[1:-1]

        for single_feat in features:
            
            df_rmse = pd.DataFrame(columns=['patient_id', 'RMSE', 'MAE'])
            miss_ids = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
            patients = [i for i in range(1,76) if i not in miss_ids]
            for patient in patients:
                
                df_patient = pd.DataFrame(columns=['ground_truth', 'predicted'])
                for rep in range(1,self.no_of_reps+1):
                
                    pred_filename = self.single_feat_actigraph_predictions_save_folder+single_feat+'/patient_{}/prediction_rep_{}.csv'.format(patient, rep)
                    df_rep =pd.read_csv(pred_filename)
                    assert df_rep.iloc[0,0] == single_feat, 'feat names dont match'
                    assert df_rep.iloc[0,1] == patient, 'patient ids dont match'
                    df_rep.drop('single_feat', axis=1, inplace=True)
                    df_rep.drop('patient_id', axis=1, inplace=True)
                    df_patient = pd.concat([df_patient, df_rep], axis=0)
                df_patient.reset_index(drop=True, inplace=True)
                rmse = mean_squared_error(df_patient['ground_truth'],df_patient['predicted'], squared=False)
                mae = mean_absolute_error(df_patient['ground_truth'],df_patient['predicted'])
                df_rmse = pd.concat([df_rmse, pd.DataFrame([[patient, rmse, mae]], columns=['patient_id', 'RMSE', 'MAE'])], axis=0)
        
            df_rmse.to_csv(self.single_feat_actigraph_predictions_save_folder+single_feat+'/predictions_rmse.csv', index=False, header=True)
            
    def boxplot_RMSE_univariate_accelerometry_models(self):
        #plot model performance comparison in terms of RMSE and MAE for our model vs single_feature models
        # run this after training both univariate and multivariate RFs
        
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(15,10)) 
        
        df_plot_RMSE = pd.DataFrame()
        df_plot_MAE = pd.DataFrame()

        df = pd.read_csv(self.actigraph_predictions_save_folder+'predictions_rmse.csv') # this files is generated by multivariate RF model
        df_plot_RMSE['Multivariate Clock'] = df['RMSE']
        df_plot_MAE['Multivariate Clock'] = df['MAE']

        df_actigraph = pd.read_csv(self.actigraph_filename)
        features = df_actigraph.columns.values[1:-1]

        for single_feat in features:
            df = pd.read_csv(self.single_feat_actigraph_predictions_save_folder+single_feat+'/predictions_rmse.csv')
            df_plot_RMSE[single_feat] = df['RMSE']
            df_plot_MAE[single_feat] = df['MAE']

            
        df_plot_RMSE.plot(kind='box', ax=ax[0], showfliers=False)
        
        df_plot_MAE.plot(kind='box', ax=ax[1], showfliers=False)
        plt.tight_layout()
        ax[0].tick_params(labelrotation=90)
        ax[1].tick_params(labelrotation=90)
        plt.setp(ax[0], ylabel='RMSE')
        plt.setp(ax[1], ylabel='MAE')
        fig.tight_layout(h_pad=3)
        plt.savefig(self.single_feat_actigraph_predictions_save_folder+'comparison_boxplot_RMSE.pdf', format='pdf')

        


    def repeated_k_folds_per_patient(self, output_array,patient,X,Y,X_pre,Y_pre,features):
        
        df_feat_importance_col = ['iteration', 'fold']
        df_feat_importance_col.extend(features)
        df_feat_importance = pd.DataFrame(columns=df_feat_importance_col)
        
        for k in range(1, self.no_of_reps+1):
            df_predictions = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted'])
            df_predictions_preIncluded = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted'])
            kf = KFold(n_splits=self.no_of_folds, shuffle=True, random_state=k)
            
            fold = 0

            for train_index, test_index in kf.split(X):
            
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                
                X_test_combined = np.concatenate((X_test, X_pre), axis=0)
                Y_test_combined = np.concatenate((y_test, Y_pre), axis=0)

                max_iter = 1000000
                   
                clf = RandomForestRegressor(n_estimators=self.RF_no_of_estimators, random_state=k)
                clf.fit(X_train, y_train.flatten())
            
                y_predicted = clf.predict(X_test)
                y_predicted_combined = clf.predict(X_test_combined)
        
                importances = clf.feature_importances_
                df_feat_importance.loc[df_feat_importance.shape[0],:] = [k, fold] + list(importances)
                
                df_fold_predictions = pd.DataFrame(columns=['patient_id', 'ground_truth', 'predicted'])
                for y_ground, y_pred in zip(y_test.flatten(), y_predicted):
                    df_predictions.loc[df_predictions.shape[0],:] = [patient, y_ground, y_pred]
                    df_fold_predictions.loc[df_fold_predictions.shape[0],:] = [patient, y_ground, y_pred]
                
                for y_ground, y_pred in zip(Y_test_combined.flatten(), y_predicted_combined):
                    df_predictions_preIncluded.loc[df_predictions_preIncluded.shape[0],:] = [patient, y_ground, y_pred]
                
                fold += 1
                
                rho, pval = spearmanr(df_fold_predictions['ground_truth'], df_fold_predictions['predicted'])         
        
            
            df_predictions.sort_values(by='ground_truth', axis=0, ascending=True, inplace=True)
            df_predictions.reset_index(drop=True, inplace=True)
            df_predictions_filename = self.actigraph_predictions_save_folder+'patient_{}/'.format(patient)
            df_predictions_preIncluded_filename = self.actigraph_predictions_save_folder+'patient_{}_preIncluded/'.format(patient)
        
            os.makedirs(df_predictions_filename, exist_ok=True)
            os.makedirs(df_predictions_preIncluded_filename, exist_ok=True)
            df_predictions.to_csv(df_predictions_filename+'prediction_rep_{}.csv'.format(k), header=True, index=False)
            df_predictions_preIncluded.to_csv(df_predictions_preIncluded_filename+'prediction_rep_{}.csv'.format(k), header=True, index=False)

            rho, pval = spearmanr(df_predictions['ground_truth'], df_predictions['predicted'])
            output_array.append([patient, rho, pval])
       
        df_feat_importance.to_csv(df_predictions_filename+'feature_importances.csv', header=True, index=False) 


        

    def train_accelerometry_model(self):
        # train and test the multivariate RF model on accelerometry featureset
        df_original = pd.read_csv(self.actigraph_filename)
        output_cols = ['patient_id', 'rho', 'pval']
        manager = multiprocessing.Manager()
        output_array = manager.list()
        
        patients = [i for i in range(1,76) if i not in self.missing_ids]

        numberOfThreads = 5 # adjust for your CPU config
        jobs = []
        for patient in patients:
            
                
            if patient < 10:
                patient_id = 'VAL0{}.agd'.format(patient)
            else:
                patient_id = 'VAL{}.agd'.format(patient)

            df = df_original[(df_original['Day'] > 0) & (df_original['patient_id'] == patient_id)]
            df_pre = df_original[(df_original['Day'] < 1) & (df_original['patient_id'] == patient_id)]
            
            df.drop(['patient_id'], axis=1,inplace=True)
            df.reset_index(inplace=True)
            df.dropna(inplace=True)
            label = 'Day'
            features = df.columns.values[1:-1] #remove index and label columns
            X = df[features].values
            Y = df[label].values
            X_pre = df_pre[features].values
            Y_pre = df_pre[label].values

            p = multiprocessing.Process(target=self.repeated_k_folds_per_patient, args=(output_array,patient,X,Y,X_pre,Y_pre,features,))
            jobs.append(p)
        for i in self.chunks(jobs,numberOfThreads):
            for j in i:
                j.start()
            for j in i:
                j.join()
    

        output_df = pd.DataFrame(np.array(output_array), columns=output_cols)
        output_df.to_csv(self.actigraph_predictions_save_folder+'predictions.csv', index=False, header=True)
        
        


    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def boxplot_accelerometry_model(self):
        pred_filename = self.actigraph_predictions_save_folder+'predictions.csv'
        df_original =pd.read_csv(pred_filename)
        df_original['rho'] = df_original['rho'].clip(lower=0)
        df_original['log10_pval'] = -1*np.log10(df_original['pval'].values)
        patients = [i for i in range(1,76) if i not in self.missing_ids]
        plot_df = pd.DataFrame(columns=['patient_id', 'rho', 'log10_pval'])
        
        for patient in patients:
            df = df_original[df_original['patient_id'] == patient]
            plot_df.loc[plot_df.shape[0],:] = [patient, df['rho'].median(), df['log10_pval'].median()]
            
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(5,5), constrained_layout=True) 
        sns.boxplot(y="rho", data=plot_df, showfliers = False, ax=ax[0], color="royalblue")
        sns.swarmplot(y="rho", data=plot_df, color=".25", ax=ax[0])
        sns.boxplot(y="log10_pval", data=plot_df, showfliers = False, ax=ax[1], color="limegreen")
        sns.swarmplot(y="log10_pval", data=plot_df, color=".25", ax=ax[1])
        plt.tight_layout()
        ax[0].set_ylim([0,1])
        ax[1].set_ylim([0.1,20])
        ax[1].axhline(y=-1*np.log10(0.05), linewidth=4, color='r')

        plt.savefig(self.actigraph_predictions_save_folder+'boxplot_medians.jpg', format='jpg', pad_inches=1)
        plot_df.to_csv(self.actigraph_predictions_save_folder+'median_predictions.csv', header=True, index=False) 

        
    def RMSE_accelerometry_model(self):
        # get the rmse for each patient
        df_rmse = pd.DataFrame(columns=['patient_id', 'RMSE', 'MAE'])
        excluded_file_no = [3, 12, 15, 16, 19, 20, 24, 27, 28, 31, 36, 40, 41, 46, 47, 50, 52, 53, 56, 58, 62,  63, 66, 65, 67, 74]
        patients = [i for i in range(1,76) if i not in excluded_file_no]
        for patient in patients:
            
            df_patient = pd.DataFrame(columns=['ground_truth', 'predicted'])
            for rep in range(1,self.no_of_reps+1):
    
                pred_filename = self.actigraph_predictions_save_folder+'patient_{}/prediction_rep_{}.csv'.format(patient, rep)
                df_rep =pd.read_csv(pred_filename)
                assert df_rep.iloc[0,0] == patient, 'patient ids dont match'
                
                df_rep.drop('patient_id', axis=1, inplace=True)
                df_patient = pd.concat([df_patient, df_rep], axis=0)
            df_patient.reset_index(drop=True, inplace=True)
            rmse = mean_squared_error(df_patient['ground_truth'],df_patient['predicted'], squared=False)
            mae = mean_absolute_error(df_patient['ground_truth'],df_patient['predicted'])
            df_rmse = pd.concat([df_rmse, pd.DataFrame([[patient, rmse, mae]], columns=['patient_id', 'RMSE', 'MAE'])], axis=0)
    
        df_rmse.to_csv(self.actigraph_predictions_save_folder+'predictions_rmse.csv', index=False, header=True)
            
            
    
    def lineplot_accelerometry_model(self):
        # plot line plot with CI

        patients = [i for i in range(1,76) if i not in self.missing_ids]
        confidence_df_col = ['patient_id']
        confidence_df_col.extend([d for d in range(-5,41) if d!=0])
        confidence_df = pd.DataFrame(columns=confidence_df_col)
        
        x1 = np.linspace(-5, 46, 500)
        for patient in patients: 
            patient_filename = self.actigraph_predictions_save_folder+'median_predictions/Predictions_patient_{}.csv'.format(patient)
            df_patient_pred = pd.read_csv(patient_filename)
            df_patient_pred = df_patient_pred[df_patient_pred['ground_truth']>=-5]
            row = [patient]
            row.extend(list(df_patient_pred['predicted_value'].values))
            confidence_df.loc[confidence_df.shape[0]] = row

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

        confidence=0.9
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
        ax.fill_between(x1, y3, y4, color='red', alpha=0.5)
        ax.set_xlim([-5,40])
        ax.set_ylim([0,40])
        plt.xticks([-5,0,15,30,40])
        plt.title('avg spline (actual vs predicted day since surgery) over all patient models, {} CI'.format(confidence), fontsize=7)
        plt.savefig(self.actigraph_predictions_save_folder+'Preds_lineplot_CI90.pdf'.format(confidence), format='pdf', dpi=600)
        

    def accelerometry_model_clusters_lineplot(self):

        df = pd.read_csv(self.actigraph_filename)
        df_avg = pd.DataFrame(columns=df.columns.values[1:])
        for day in range(-5,41):
            if day == 0:
                continue
            df_day = df[(df['Day']==day)]
            df_day.drop(['patient_id'], axis=1, inplace=True)
            row = list(df_day.mean(axis=0).values[:-1])
            row.append(day)
            df_avg.loc[df_avg.shape[0],:] = row
        df_avg.to_csv(self.actigraph_predictions_save_folder+'Features_vs_days.csv', header=True, index=False)
        df_avg_t = df_avg[df_avg.columns.values[:-1]].T
        df_avg_t.reset_index(drop=False, inplace=True)
        df_avg_t.columns = [i for i in range(df_avg_t.shape[1])]
        km = KMeans(n_clusters=6, random_state=42)
        km.fit(df_avg_t.iloc[:,1:].values)
        df_avg_t['cluster_id'] = km.labels_
        df_avg_t.to_csv(self.actigraph_predictions_save_folder+'Features_vs_days_clustered.csv', header=True, index=False)

        patients = [i for i in range(1,76) if i not in self.missing_ids]

        fig,ax = plt.subplots(int(6/2)+1,3,figsize=(10,10))
        cmap_set1 = plt.get_cmap('Set2')
        colors = cmap_set1([i for i in range(6)]) # colors are randomly assigned to a cluster_id (may not match the colores in manuscript)

        x = np.linspace(0, 45, 50)
        for c_id in range(6):
            ax[int(c_id/2),int(c_id%2)].set_ylim([np.min(df_avg_t.iloc[:,1:(45+1)].values),np.max(df_avg_t.iloc[:,1:(45+1)].values)])
            df_c_id = df_avg_t[df_avg_t['cluster_id']==c_id]
            df_c_id.reset_index(drop=True, inplace=True)
            confidence_df_col = ['patient_id']
            confidence_df_col.extend([d for d in range(-5,41) if d!=0])
            confidence_df = pd.DataFrame(columns=confidence_df_col)
            for patient in patients:
                if patient < 10:
                    patient_id = 'VAL0{}.agd'.format(patient)
                else:
                    patient_id = 'VAL{}.agd'.format(patient)
                df_patient_cid = df[(df['patient_id']==patient_id)][list(df_c_id[0].values)]
                df_patient_cid['avg'] = df_patient_cid.mean(axis=1)
                row = [patient_id]
                row.extend(list(df_patient_cid['avg'].values))
                confidence_df.loc[confidence_df.shape[0]] = row
            
            confidence=0.9
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
            tck2 = interpolate.splrep([idx for idx in range(45)],average)
            y2 = interpolate.splev(x, tck2)
            tck3 = interpolate.splrep([idx for idx in range(45)],upper)
            y3 = interpolate.splev(x, tck3)
            tck4 = interpolate.splrep([idx for idx in range(45)],lower)
            y4 = interpolate.splev(x, tck4)  
            ax[int(c_id/2),int(c_id%2)].fill_between(x, y3, y4, color=colors[c_id], alpha=0.5)
            tck1 = interpolate.splrep([idx for idx in range(45)],km.cluster_centers_[c_id])
            y1 = interpolate.splev(x, tck1)
            ax[int(c_id/2),int(c_id%2)].plot(x, y1, color=colors[c_id], linewidth=2)
            ax[int(c_id/2),int(c_id%2)].set_xlim([0,45])
        plt.xticks([-5,15,30,40])

        plt.savefig(self.actigraph_predictions_save_folder+'Actigraph_clusters_lineplot_CI90_all_patients.pdf', format='pdf', dpi=600)
        plt.close()


    def accelerometry_model_corr_network(self, net_type='clusters'):

        df = pd.read_csv(self.actigraph_filename)
       
        df_avg_t = pd.read_csv(self.actigraph_predictions_save_folder+'Features_vs_days_clustered.csv')
        df_avg = pd.read_csv(self.actigraph_predictions_save_folder+'Features_vs_days.csv')

        features = df.columns.values[1:-1]
        patients = [i for i in range(2,76) if i not in self.missing_ids]
        df_feat_importances = pd.read_csv(self.actigraph_predictions_save_folder+'patient_1/feature_importances.csv') #read the first file
        for patient in patients:
            df_feat_importances = pd.concat([df_feat_importances,pd.read_csv(self.actigraph_predictions_save_folder+'patient_{}/'.format(patient)+'feature_importances.csv')])
        df_feat_importances.reset_index(drop=True, inplace=True)
        df_feat_importances.loc[df_feat_importances.shape[0],2:] = df_feat_importances[features].median(axis=0).values
        feature_importances = df_feat_importances.loc[df_feat_importances.shape[0]-1, features]
     
        univar_pval = []
        for col in features:
            c, p = spearmanr(df_avg[col].values[5:], df_avg['Day'].values[5:])
            univar_pval.append(-1*np.log10(p))
       
        embedded = pd.read_csv(self.actigraph_predictions_save_folder+'Actigraph_Rtsne_embedding.csv', index_col=0).values #using the results from Rtsne package for embedding. If files is missing you should run 'Rtsne_embedding.R' in scripts folder
        df_embedded = pd.DataFrame(data=embedded, columns=['d1', 'd2'])
        G = nx.Graph()
        eps = 1e-11
        node_colors = []
        node_colors_feat_imp = []
        node_colors_indeces = []
        node_colors_hex = []
        gray_cm = plt.get_cmap('hot')

        cNorm  = matplotlib.colors.Normalize(vmin=-(max(feature_importances.max(),0.035)), vmax=max(feature_importances.max(),0.035))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=gray_cm)
        for feature_imp in feature_importances.values:
            if feature_imp != 0 and feature_imp < eps:
                print('eps not small enough')
                sys.exit()
            node_colors_feat_imp.append(scalarMap.to_rgba(feature_imp))

        cmap_set1 = plt.get_cmap('Set1')
        colors = cmap_set1([i for i in range(6)])    
        node_sizes = pd.Series(univar_pval, dtype=object).fillna(0).tolist()
        node_sizes_copy = node_sizes.copy()
        for idx in range(len(node_sizes)):
            node_sizes[idx] = 20*node_sizes[idx]
            if np.abs(node_sizes[idx]) < 1:
                node_sizes[idx] = 1

        labels={}
        for i in range(len(node_sizes)):
            this_color = colors[df_avg_t[df_avg_t['0']==features[i]]['cluster_id'].values[0]]
            node_colors.append(this_color)
            node_colors_hex.append(matplotlib.colors.rgb2hex(this_color[:3]))
            node_colors_indeces.append(df_avg_t[df_avg_t['0']==features[i]]['cluster_id'].values[0])
            G.add_node(i, pos=(embedded[i][0],embedded[i][1]))
            labels[i] = features[i]
 
        for i, col1 in enumerate(df_avg.columns.values[:-1]):
            for j, col2 in enumerate(df_avg.columns.values[:-1]):
                if j > i:
                    rho, pval = spearmanr(df_avg[col1], df_avg[col2])
                    if (pval*1891) < 0.01:
                        G.add_edge(i,j)

        df_embedded['node_label'] = features
        df_embedded['color'] = node_colors_hex
        df_embedded['cluster_id'] = node_colors_indeces
        df_embedded.to_csv(self.actigraph_predictions_save_folder+'Actigraph_features_Rtsne_embedded_hex_clusterId_color.csv', header=False, index=False)
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        pos = nx.get_node_attributes(G,'pos')
        if net_type == 'clusters':
            nx.draw(G, pos, ax=ax, node_color=node_colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, edge_color='darkgrey', width=0.1)
            plt.savefig(self.actigraph_predictions_save_folder+'Actigraph_corr_network_clustered.pdf', format='pdf', dpi=600)
        else: # color is grdient of feature importance index
            nx.draw(G, pos, ax=ax, node_color=node_colors_feat_imp, labels=labels, font_size=1, node_size=node_sizes, alpha=1, edge_color='darkgrey', width=0.1)
            plt.savefig(self.actigraph_predictions_save_folder+'Actigraph_corr_network_feat_imp.pdf', format='pdf', dpi=600)
        plt.close()

    def get_recovery_surrogate(self,  patient_id):
        ''' return recovery surrogate for patient id '''

        patient_filename = self.actigraph_predictions_save_folder+'median_predictions/Predictions_patient_{}.csv'.format(patient_id)
        df = pd.read_csv(patient_filename)
        df_pre_feat = df[df['ground_truth']<0]['predicted_value'].quantile(0.25)
        patient_pre_save_filename = self.actigraph_predictions_save_folder+'pre_predictions/'
        os.makedirs(patient_pre_save_filename, exist_ok=True)
        df[df['ground_truth']<0].to_csv(patient_pre_save_filename+'Pre_predictions_patient_{}.csv'.format(patient_id), header=True, index=False)

        return {'patient_id':patient_id, 'baseline_pred_surr':df_pre_feat}
        


    def compute_multivariate_surrogate(self):

        # get median predictions of reps 
        for patient in [i for i in range(1,76) if i not in self.missing_ids]:
            patient_med_preds = pd.DataFrame(columns=['ground_truth', 'predicted_value'])
            patient_df = pd.DataFrame()
            for k in range(1,self.no_of_reps+1):
                patient_rep_df = pd.read_csv(self.actigraph_predictions_save_folder+'patient_{0:}_preIncluded/prediction_rep_{1:}.csv'.format(patient, k))
                patient_df = pd.concat([patient_df, patient_rep_df])
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
        for patient in [i for i in range(1,76) if i not in self.missing_ids]:
                row = self.get_recovery_surrogate(patient)
                output_df = pd.concat([output_df, pd.DataFrame(data=np.asarray(list(row.values())).reshape(1,-1), columns=list(row.keys()))])

        output_path = self.extract_recovery_feature_save_folder + 'multiv_surrogate/'
        os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_df.to_csv(output_path+'multiv_surrogate.csv', header=True, index=False)


    def extract_recovery_feature_per_actigraph_feature(self):

        def baseline_recovery(patient_id, actigraph_features_decline):

            df = pd.read_csv(self.actigraph_filename)
            df = df[df['patient_id']==patient]
            actigraph_features = df.columns.values[1:-1]
            return_dict = {'patient_id':patient_id}
            for i, act_feat in enumerate(actigraph_features):
                df_feat = df[['Day',act_feat]]
                if actigraph_features_decline[i]:
                    df_feature_threshold = df_feat[df_feat['Day']<0][act_feat].quantile(0.25)
                else:
                    df_feature_threshold = df_feat[df_feat['Day']<0][act_feat].quantile(0.75)
                df_feat_post = df_feat[df_feat['Day'] > 0]
                df_feat_post.reset_index(drop=True, inplace=False)
                for _, row in df_feat_post.iterrows():
                    return_dict.update({act_feat:row['Day']})
                    if actigraph_features_decline[i]:
                        if row[act_feat] >= df_feature_threshold:
                            break
                    else:
                        if row[act_feat] <= df_feature_threshold:
                            break
            return return_dict

        df = pd.read_csv(self.actigraph_filename)
        actigraph_features = df.columns.values[1:-1]
        df_patient_average = pd.DataFrame(columns=df.columns.values[1:])
        for day in range(-5,41):
            if day==0:
                continue
            df_day = df[df['Day']==day]
            df_patient_average.loc[len(df_patient_average),:] = df_day.mean(axis=0).values
        actigraph_features_decline = [True for _ in actigraph_features]
        actigraph_features_decline = [True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, True, False, False, True, False,
                                        False, False, False, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                                        True, True, True, True, True, True, True, True, True, False, True, False, True, True, False, False, False, False, False, False,
                                        False, False] # recycle a previously computed list
        # for i, act_feat in enumerate(actigraph_features): # if you wish to compute the actigraph_features_decline again uncomment this block o.w. will use the previously computed
        #     fig, ax = plt.subplots(figsize=(5,5))
        #     ax.plot(df_patient_average['Day'], df_patient_average[act_feat])
        #     ax.set_title(act_feat)
        #     # plt.show()
        #     os.makedirs(self.surrogate_vs_preop_univ_corr_save_folder+'temp/', exist_ok=True)
        #     plt.savefig(self.surrogate_vs_preop_univ_corr_save_folder+'temp/{}.jpg'.format(act_feat), format='jpg', pad_inches=1) # or save in a temp file
        #     plt.close()
        #     inp = input('Feature ({}). Is there a decline after surgery? input 1 if yes otherwise input 0:\n(If plot not shown check the {} path for saved jpg):'.format(act_feat, self.surrogate_vs_preop_univ_corr_save_folder+'temp/')) # manually set, a one time process, can be made automatic if needed
        #     while(True):
        #         if inp == '1':
        #             actigraph_features_decline[i] = True
        #             break
        #         elif inp == '0':
        #             actigraph_features_decline[i] = False
        #             break
        #         inp = input('Feature ({}). Is there a decline after surgery? input 1 if yes otherwise input 0:\n(If plot not shown check the {} path for saved jpg):'.format(act_feat, self.surrogate_vs_preop_univ_corr_save_folder+'temp/'))
        output_df = pd.DataFrame()
        for patient in [i for i in range(1,76) if i not in self.missing_ids]:
            if patient < 10:
                patient = 'VAL0{}.agd'.format(patient)
            else:
                patient = 'VAL{}.agd'.format(patient)
            row = baseline_recovery(patient, actigraph_features_decline)
            
            output_df = pd.concat([output_df, pd.DataFrame(data=np.asarray(list(row.values())).reshape(1,-1), columns=list(row.keys()))])
        univ_recovery_feature_save_folder = self.extract_recovery_feature_save_folder+'univar_surrogate/'
        os.makedirs(univ_recovery_feature_save_folder, exist_ok=True)
        output_df.to_csv(univ_recovery_feature_save_folder+'univar_surrogate.csv', header=True, index=False)



    def plot_cytof_corr_network(self, color_by='celltype'):

        df_cytof = pd.read_csv(self.cytof_filename)  
        features = df_cytof.columns.values[1:]
        df_cytof = df_cytof.iloc[:35,:] # first two plates
        imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), random_state=42)
        imp_data = imp.fit_transform(df_cytof.iloc[:,1:].values)
        df_cytof.iloc[:,1:] = imp_data
        df_cytof.drop(['patient_id'], axis=1, inplace=True)
        corr = df_cytof.corr(method ='spearman').values
        where_are_NaNs = np.isnan(corr)
        corr[where_are_NaNs] = 0
        corr = np.clip(corr, -1, 1)
        distance_matrix = 1 - np.abs(corr)
        X = df_cytof.values.transpose()
        scalerX = StandardScaler()
        scalerX.fit(X)
        X_scaled = scalerX.transform(X)
        embedded = TSNE(n_components=2, metric='precomputed', random_state=42).fit_transform(distance_matrix)
        df_embedded = pd.DataFrame(data=embedded, columns=['d1', 'd2'])
        df_embedded['node_label'] = df_cytof.columns.values

        self.draw_network_from_scratch_simple_corr(embedded, distance_matrix, features, df_cytof, df_embedded, color_by)

    def draw_network_from_scratch_simple_corr(self, embedding, distance_matrix, col_labels, df_cytof, df_embedded, color_by):

        G = nx.Graph()
        eps = 1e-11
        node_colors = []
        node_colors_idx = []
            
        if color_by == 'celltype':
            position_in_col_label=2
        elif color_by == 'stim':
            position_in_col_label=1
        elif color_by == 'marker':
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
        cm_subsection = np.linspace(0.01, 0.75, number_of_subsets) 
        colors = [cm.terrain(x) for x in cm_subsection]
    
        for feat in df_cytof.columns.values:
            subset = feat.split('.')[position_in_col_label]
            if feat.split('.')[0] != 'freq' or color_by != 'marker':
                node_colors.append(colors[subsets.index(subset)])
            else:
                node_colors.append('black')
        legend_elements = []
        for ct,c in zip(subsets,colors):
            legend_elements.append(mpatches.Patch(facecolor=c,label=ct))
    
        df_embedded['color'] = [matplotlib.colors.rgb2hex(i) for i in node_colors]
        # os.makedirs(self.recovery_feature_predictions_save_folder+'corr_networks/', exist_ok=True)
        # df_embedded.to_csv(self.recovery_feature_predictions_save_folder+'corr_networks/Cytof_tsne_embedded.csv', header=True, index=False)
        # use cca weights as size
        node_sizes = list(pd.read_csv(self.recovery_feature_predictions_save_folder+'cca-cytof-acti_cytofweights.csv')['V1'].values) # if not found you should execute 
        
        for idx in range(len(node_sizes)):
            if np.abs(node_sizes[idx]) < 0.003:
                node_sizes[idx] = 0.003*1000
            else:
                node_sizes[idx] = np.abs(node_sizes[idx])*1000

        labels={}
        for i in range(distance_matrix.shape[0]):
            if len(embedding) != 0:
                G.add_node(i, pos=(embedding[i][0],embedding[i][1]))
            else:
                G.add_node(i)
            
            labels[i] = col_labels[i]

       
        all_comparisons = df_cytof.shape[1]*(df_cytof.shape[1]-1)/2
        for i, col1 in enumerate(df_cytof.columns.values):
            for j, col2 in enumerate(df_cytof.columns.values):
                if j > i:
                    rho, pval = spearmanr(df_cytof[col1], df_cytof[col2])
                    if (pval*all_comparisons) < 0.05:
                        G.add_edge(i,j)
        
        fig, ax = plt.subplots(figsize=(12,10))
        pos = nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, ax=ax, node_color=node_colors, labels=labels, font_size=1, node_size=node_sizes, alpha=1, edge_color='darkgray', width=0.1)
        plt.savefig(self.recovery_feature_predictions_save_folder+'Cytof_corr_network_colored_by_{0:}_tsne_labeled.pdf'.format(color_by), format='pdf', dpi=600)
        plt.close()
        # fig, ax = plt.subplots(figsize=(12,10))
        # plt.legend(handles=legend_elements,
        #     scatterpoints=1,
        #     loc='lower left',
        #     bbox_to_anchor=(-0.1, -0.1),
        #     ncol=8,
        #     fontsize=4)
        # plt.savefig(self.recovery_feature_predictions_save_folder+'Cytof_corr_network_colored_by_{0:}_tsne_only_legend.pdf'.format(color_by), format='pdf', dpi=600)



    def plot_proteomics_corr_network(self):

        df_olink = pd.read_csv(self.olink_filename)        
        features = df_olink.columns.values[1:]
        df_olink.drop(['patient_id'], axis=1, inplace=True)
        corr = df_olink.corr(method ='spearman').values
        where_are_NaNs = np.isnan(corr)
        corr[where_are_NaNs] = 0
        corr = np.clip(corr, -1, 1)
        distance_matrix = 1 - np.abs(corr)
        embedded = TSNE(n_components=2, metric='precomputed', random_state=42).fit_transform(distance_matrix)
        df_embedded = pd.DataFrame(data=embedded, columns=['d1', 'd2'])
        df_embedded['node_label'] = df_olink.columns.values
        col_labels= features.copy()
        embedding = embedded.copy()
        G = nx.Graph()
        eps = 1e-11
        node_colors = []
        node_colors_idx = []
        df_prot_labels = pd.read_csv('./data/proteomics/protlabels.csv')
        subsets = list(set(df_prot_labels['protlabels'].values)) 
        number_of_subsets= len(subsets)
        cm_subsection = np.linspace(0.01, 0.75, number_of_subsets) 
        colors = [cm.terrain(x) for x in cm_subsection]
    
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
            node_colors.append(colors[subsets.index(prot_label)])
        legend_elements = []
        for ct,c in zip(subsets,colors):
            legend_elements.append(mpatches.Patch(facecolor=c,label=ct))
        
        df_embedded['color'] = [matplotlib.colors.rgb2hex(i) for i in node_colors]
        os.makedirs(self.recovery_feature_predictions_save_folder, exist_ok=True)
        # df_embedded.to_csv(self.recovery_feature_predictions_save_folder+'Olink_features_df_tsne_embedded_spearman.csv', header=True, index=False)

        labels={}
        for i in range(distance_matrix.shape[0]):
            if len(embedding) != 0:
                G.add_node(i, pos=(embedding[i][0],embedding[i][1]))
            else:
                G.add_node(i)
            labels[i] = ''

        all_comparisons = df_olink.shape[1]*(df_olink.shape[1]-1)/2
        for i, col1 in enumerate(df_olink.columns.values):
            for j, col2 in enumerate(df_olink.columns.values):
                if j > i:
                    rho, pval = spearmanr(df_olink[col1], df_olink[col2])
                    if (pval*all_comparisons) < 0.05:
                        G.add_edge(i,j)

        fig, ax = plt.subplots(figsize=(12,10))
        pos = nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, ax=ax, node_color=node_colors, labels=labels, font_size=1, node_size=30, alpha=1, edge_color='darkgray', width=0.1)
   
        plt.savefig(self.recovery_feature_predictions_save_folder+'Olink_corr_network_colored_tsne.pdf', format='pdf', dpi=600)
        plt.close()
        # fig, ax = plt.subplots(figsize=(12,10))
        # plt.legend(handles=legend_elements,
        #     scatterpoints=1,
        #     loc='lower left',
        #     bbox_to_anchor=(-0.1, -0.1),
        #     ncol=8,
        #     fontsize=4)
        # plt.savefig(self.recovery_feature_predictions_save_folder+'Olink_corr_network_colored_tsne_only_legend.pdf', format='pdf', dpi=600)



    def univ_corr_clinical_vs_surrogate(self):

        df_clinical_report = pd.read_csv(self.clinical_report_filename)
        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'multiv_surrogate/multiv_surrogate.csv')
        df_univ_corrs = pd.DataFrame(columns=['clinical_measure', 'baseline_pred_spearman_rho', 'baseline_pred_spearman_pval'])
        for clinical_feat in df_clinical_report.columns[1:]:
            row = [clinical_feat]
            c_spearman, p_spearman = spearmanr(df_clinical_report[clinical_feat], df_recovery_features['baseline_pred_surr'])
            row.extend([c_spearman, p_spearman])
            df_univ_corrs.loc[df_univ_corrs.shape[0], :] = row
        os.makedirs(self.clinical_vs_surrogate_univ_corr_save_folder, exist_ok=True)
        df_univ_corrs.to_csv(self.clinical_vs_surrogate_univ_corr_save_folder+'univ_corrs.csv',header=True, index=False)
        
    def plot_univ_corr_clinical_vs_surrogate(self):
        df= pd.read_csv(self.clinical_vs_surrogate_univ_corr_save_folder+'univ_corrs.csv')
        df['logpval'] = -1*np.log10(df['baseline_pred_spearman_pval'])
        df.sort_values(by='logpval', ascending=False, inplace=True)

        fig,ax = plt.subplots(1,1,figsize=(5,2))
        sns.barplot(y='clinical_measure', x='logpval', data=df, color='darkslategrey', ax=ax)
        ax.axvline(x=-1*np.log10(0.05), linewidth=2, color='r')
        plt.tight_layout()
        plt.savefig(self.clinical_vs_surrogate_univ_corr_save_folder+'clinical_vs_surrogate_univ_corr.pdf', format='pdf', dpi=600)
    
    def univ_corr_clinical_vs_preop(self):

        df_clinical_report = pd.read_csv(self.clinical_report_filename)
        df_actigraph = pd.read_csv(self.actigraph_filename)
        patients = [i for i in range(1,76) if i not in self.missing_ids]
        df_pre_op_actigraph = pd.DataFrame(columns=df_actigraph.columns.values[:-1])
        for patient in patients:
            if patient < 10:
                patient_id = 'VAL0{}.agd'.format(patient)
            else:
                patient_id = 'VAL{}.agd'.format(patient)
            df = df_actigraph[(df_actigraph['Day'] < 0) & (df_actigraph['patient_id'] == patient_id)]
            df.drop(['patient_id', 'Day'], axis=1, inplace=True)
            row = [patient]
            row.extend(df.mean(axis=0).values)
            df_pre_op_actigraph = pd.concat([df_pre_op_actigraph, pd.DataFrame([row], columns=df_actigraph.columns.values[:-1])], axis=0)
        df_pre_op_actigraph.reset_index(drop=True, inplace=True)
        
        df_corr = pd.DataFrame(columns=['col_clinical', 'col_actigraph', 'rho', 'pval'])
        for col_clinical in df_clinical_report.columns[1:]:
            for col_actigraph in df_pre_op_actigraph.columns[1:]:
                c, p = spearmanr(df_clinical_report[col_clinical], df_pre_op_actigraph[col_actigraph])
                df_corr = pd.concat([df_corr, pd.DataFrame([[col_clinical, col_actigraph,c,p]], columns=['col_clinical', 'col_actigraph', 'rho', 'pval'])], axis=0)
        
        os.makedirs(self.clinical_vs_preop_univ_corr_save_folder, exist_ok=True)
        df_corr.to_csv(self.clinical_vs_preop_univ_corr_save_folder+'clinical_vs_preop_univ_spearmanr.csv', header=True, index=False) # before correction
        
    def univ_corr_surrogate_vs_preop(self):

        df_recovery_features = pd.read_csv(self.extract_recovery_feature_save_folder + 'multiv_surrogate/multiv_surrogate.csv')
        df_actigraph = pd.read_csv(self.actigraph_filename)
        patients = [i for i in range(1,76) if i not in self.missing_ids]
        df_pre_op_actigraph = pd.DataFrame(columns=df_actigraph.columns.values[:-1])
        for patient in patients:
         
            if patient < 10:
                patient_id = 'VAL0{}.agd'.format(patient)
            else:
                patient_id = 'VAL{}.agd'.format(patient)

            df = df_actigraph[(df_actigraph['Day'] < 0) & (df_actigraph['patient_id'] == patient_id)]
            df.drop(['patient_id', 'Day'], axis=1, inplace=True)
            row = [patient]
            row.extend(df.mean(axis=0).values)
            df_pre_op_actigraph = pd.concat([df_pre_op_actigraph, pd.DataFrame([row], columns=df_actigraph.columns.values[:-1])], axis=0)
        df_pre_op_actigraph.reset_index(drop=True, inplace=True)
       
        df_corr = pd.DataFrame(columns=['col_actigraph', 'rho', 'pval'])
        for col_actigraph in df_pre_op_actigraph.columns[1:]:
            c, p = spearmanr(df_recovery_features['baseline_pred_surr'], df_pre_op_actigraph[col_actigraph])
            df_corr = pd.concat([df_corr, pd.DataFrame([[col_actigraph,c,p]], columns=['col_actigraph', 'rho', 'pval'])], axis=0)
        
        os.makedirs(self.surrogate_vs_preop_univ_corr_save_folder, exist_ok=True)
        df_corr.to_csv(self.surrogate_vs_preop_univ_corr_save_folder+'surrogate_vs_preop_univ_spearmanr.csv', header=True, index=False)




if __name__ == "__main__":


    actigraph_filename = './data/wearable/Activity_Sleep_UserStandardized.csv'
    cytof_filename = './data/immune/HipValidation_cytof_Pre_stim_baseline_adjusted_all_plates.csv'
    olink_filename = './data/proteomics/olink_Pre.csv'
    clinical_report = './data/clinical/clinical.csv'

    main_save_folder = None # if the working main save folder is not defined, generate a new folder
    # main_save_folder = './output/RF_100estimators_10folds_30reps_YOURWORKINGDIRECTORY/' # continue working with this output folder

    recovery_prediction_pipeline = recovery_prediction_pipeline(main_save_folder, actigraph_filename, cytof_filename, olink_filename, clinical_report) # define class obj
    # uncomment each function that you wish to execute
    # recovery_prediction_pipeline.train_univariate_accelerometry_models() #Train and evaluate univariate accelerometery RF models 
    # recovery_prediction_pipeline.boxplot_univariate_accelerometry_models() # plot corrs
    # recovery_prediction_pipeline.RMSE_univariate_accelerometry_models()
    # recovery_prediction_pipeline.boxplot_RMSE_univariate_accelerometry_models()
    # recovery_prediction_pipeline.train_accelerometry_model()
    # recovery_prediction_pipeline.boxplot_accelerometry_model()
    # recovery_prediction_pipeline.RMSE_accelerometry_model()
    # recovery_prediction_pipeline.accelerometry_model_clusters_lineplot() # Fig2C.
    # recovery_prediction_pipeline.accelerometry_model_corr_network(net_type='clusters') # Fig2B 
    # recovery_prediction_pipeline.accelerometry_model_corr_network(net_type='feat_imp') # Fig3B
    # recovery_prediction_pipeline.compute_multivariate_surrogate() 
    # recovery_prediction_pipeline.lineplot_accelerometry_model() # Fig3A
    # recovery_prediction_pipeline.univ_corr_clinical_vs_surrogate() 
    # recovery_prediction_pipeline.plot_univ_corr_clinical_vs_surrogate() # Fig4
    # recovery_prediction_pipeline.extract_recovery_feature_per_actigraph_feature()

    # recovery_prediction_pipeline.plot_cytof_corr_network(color_by='stim')
    # recovery_prediction_pipeline.plot_cytof_corr_network(color_by='marker')
    # recovery_prediction_pipeline.plot_cytof_corr_network(color_by='celltype')
    # recovery_prediction_pipeline.plot_proteomics_corr_network()

    ## recovery_prediction_pipeline.univ_corr_clinical_vs_preop() 
    ## recovery_prediction_pipeline.univ_corr_surrogate_vs_preop()








    
   


