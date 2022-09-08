import os
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import date
from sklearn import preprocessing, metrics
from sklearn.linear_model import RidgeClassifierCV
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from .Noise import Noise
from .Explanation import load_explanation, create_random_explanation
from .Experiment import Evaluate, train_referee
import utils.visualization as vis
from utils.data import LocalDataLoader



class CompareExplanation():
    def __init__(self, datapath, dataset,explanation_list, explanation_names, 
        referee_list, noise_list):
        self.datapath = datapath
        self.dataset = dataset
        data = LocalDataLoader(datapath,self.dataset)
        self.X_train,self.y_train,self.X_test,self.y_test = data.get_X_y()
        self.referee_list = referee_list
        self.noise_list = noise_list
        self.explanation_names = explanation_names
        self.explanation_list = explanation_list
        

        col_names=['dataset','noise_type','XAI_method', 'Referee', 'threshold','metrics: acc']
        self.acc_df = pd.DataFrame(columns=col_names)

        col_names = ['dataset','noise_type','XAI_method','Referee','metrics: explanation_auc']
        auc_df = pd.DataFrame(columns=col_names)
        for ref in self.referee_list:
            model,transformer = train_referee(self.X_train,self.y_train,ref,self.dataset)
            for noise_type in noise_list:
                for exp,name in zip(self.explanation_list, self.explanation_names):
                    evaluate = Evaluate(datapath=self.datapath,dataset=self.dataset,explanation=exp,
                        referee=ref, noise_type=noise_type,model=model,transformer=transformer)
                    # print('Explanation AUC for ' + name + ':' + str(evaluate.explanation_auc))
                    self.acc_df = evaluate.record_result(explanation_name=name, existing_df=True, df=self.acc_df)
                    auc_df = auc_df.append({'dataset': self.dataset,
                    'noise_type':noise_type,
                    'XAI_method': name,
                    'Referee': ref,
                    'metrics: explanation_auc': evaluate.explanation_auc}, ignore_index=True)
                    

        self.auc_df = auc_df

    def visualize(self,noise_type='global_mean'):
        df1 = self.acc_df.loc[self.acc_df['noise_type'] == noise_type]
        path = './plot/acc_curve_%s_%s' %(self.dataset,noise_type)
        vis.visualize_experiment_result(df1,savefig=True, savepath=path)

    def statistics(self,):
        pass
        


def run_experiment(datapath, dataset,referee_list = ['rocket','resnet','knn','MrSEQLClassifier',],
    perturbation_types=['global_mean','local_mean']):
    today = str(date.today()).replace('-', '')

    # get explanations:
    if dataset in ['CBF', 'CMJ', 'Coffee', 'ECG200', 'GunPoint']:
        xais = ['LIME', 'MrSEQL', 'Resnet','GradientShap','IG','ROCKET'] 
        explanation_names =[ 'lime_mrseql', 'mrseql', 'cam','GS','IG','lime_rocket']
    else:
        xais = ['LIME', 'MrSEQL','GradientShap','IG','ROCKET'] 
        explanation_names =[ 'lime_mrseql', 'mrseql','GS','IG','lime_rocket']
    
    # xais = ['ROCKET']
    # explanation_names = ['lime_rocket']    
    
    assert len(xais) == len(explanation_names)
    random_seeds = [2020]
    lime_xais = ['lime','LIME','Lime','ROCKET']
    
    explanation_list = []
    for xai in xais:
        is_reshape=True if xai in lime_xais else False
        weight = load_explanation(datapath=datapath,dataset=dataset,explanation_type=xai,reshape_lime=is_reshape)
        explanation_list.append(weight)

    for seed in random_seeds:
        random_weight, random_weight_name = create_random_explanation(datapath=datapath,dataset=dataset)
        explanation_list.append(random_weight)
        explanation_names.append(random_weight_name)

    print('Explanation Shape: ', explanation_list[0].shape)
    print('Total numbers of XAI Methods: ', len(explanation_list))
    print('Names of XAI Methods: ', explanation_names)


    # Compare explanations
    compare = CompareExplanation(datapath, dataset,explanation_list, explanation_names, 
        referee_list, perturbation_types)
    auc_df=compare.auc_df
    acc_df=compare.acc_df
    auc_path = './output/%s_%s.csv' %(dataset,today)
    acc_path = './output/acc_%s_%s.csv' %(dataset,today)
    auc_df.to_csv(auc_path, index=False)
    acc_df.to_csv(acc_path, index=False)
    # compare.visualize(noise_type='local_mean')
    # compare.visualize(noise_type='global_mean')


    return


def get_final_ranking_new(auc_df,digit=4,beautify_display=True,ranking_by_perturbation_method=False):
    # auc_df is the resulted dataframe from CompareExplanation class
    dataset = list(set(auc_df['dataset'].tolist()))
    assert len(dataset) == 1
    dataset=dataset[0]
    xais = set(auc_df['XAI_method'].tolist())
    referees = set(auc_df['Referee'].tolist())
    pers = set(auc_df['noise_type'].tolist())

    
    col_names=['dataset','noise_type','XAI_method','Referee','metrics: explanation_auc','average_scaled_auc']
    val_df = pd.DataFrame(columns=col_names)
    for ref in referees:
        for noise_type in pers:
            df = auc_df[(auc_df['noise_type']==noise_type) & 
                        (auc_df['Referee']==ref)]
            min_,max_ = df['metrics: explanation_auc'].min(), df['metrics: explanation_auc'].max()
            if min_ != max_:
                df['average_scaled_auc'] = (df['metrics: explanation_auc']-min_)/(max_-min_)
            else:
                df['average_scaled_auc'] = (df['metrics: explanation_auc']-min_)/1.0
            val_df = pd.concat([val_df, df], ignore_index=True, axis=0)
            
    val_df = pd.pivot_table(val_df, values='average_scaled_auc', 
        index=['dataset','noise_type','XAI_method'],
        aggfunc=np.average)
 
    if ranking_by_perturbation_method:
        val_df = pd.pivot_table(val_df, values='average_scaled_auc', 
        index=['dataset','noise_type','XAI_method'],
        aggfunc=np.average)
    
    else:
        val_df = pd.pivot_table(val_df, values='average_scaled_auc', 
        index=['dataset','XAI_method'],
        aggfunc=np.average)
    
    val_df = val_df.reset_index()
    col_names=val_df.columns.tolist()
    col_names.append('scaled_ranking')
    ans = pd.DataFrame(columns=col_names)

    
    if ranking_by_perturbation_method:
        for noise_type in pers:
            df = val_df[val_df['noise_type'] == noise_type]
            min_,max_ = df['average_scaled_auc'].min(),df['average_scaled_auc'].max()
            df['scaled_ranking'] = (df['average_scaled_auc']-min_)/(max_-min_)
            ans = pd.concat([ans, df], ignore_index=True, axis=0)

    else:
        df = val_df
        # cal_scaled_ranking(df, ans1)
        min_,max_ = df['average_scaled_auc'].min(),df['average_scaled_auc'].max()
        df['scaled_ranking'] = (df['average_scaled_auc']-min_)/(max_-min_)
        ans = pd.concat([ans, df], ignore_index=True, axis=0)
   
    if beautify_display:
        cm = sns.light_palette("green", as_cmap=True,reverse=True)
        display(ans.style.background_gradient(cmap = cm,axis=0))

    return ans




def get_final_ranking(auc_df,digit=4,beautify_display=True,ranking_by_perturbation_method=False):
    # auc_df is the resulted dataframe from CompareExplanation class
    dataset = list(set(auc_df['dataset'].tolist()))
    assert len(dataset) == 1
    dataset=dataset[0]
    xais = set(auc_df['XAI_method'].tolist())
    referees = set(auc_df['Referee'].tolist())
    pers = set(auc_df['noise_type'].tolist())


    # Step1: get scaled auc for each noise_type and referee
    vals = defaultdict(list)
    for ref in referees:
        for noise_type in pers:
            df = auc_df[(auc_df['noise_type']==noise_type) & 
                        (auc_df['Referee']==ref)]
            # print(df)
            min_,max_ = df['metrics: explanation_auc'].min(), df['metrics: explanation_auc'].max()
            if min_ != max_:
                df['pct'] = (df['metrics: explanation_auc']-min_)/(max_-min_)
                for xai in xais:
                    val = df[df['XAI_method']==xai]['pct']
                    vals[xai].append(val)
            else:
                for xai in xais:
                    vals[xai].append(0.0) 
    
    # Step2.1 : get average auc for the dictionary by xai method
    final = defaultdict(float)
    for key,val in vals.items():
        x = np.average(vals[key])
        final[key] = x
    
        # Step2.2: get min/max scaled auc 
    min_,max_ = min(final.values()), max(final.values())
    col_names=['dataset','XAI_method','average_scaled_auc','scaled_ranking']
    ans = pd.DataFrame(columns=col_names)
    for key,val in final.items():


        ranking = (val-min_)/(max_-min_)
        ans = ans.append({'dataset': dataset,
                          'XAI_method':key,
                    'average_scaled_auc':round(val,digit),
                          'scaled_ranking': round(ranking,digit)
                 }, ignore_index=True)
    if beautify_display:
        cm = sns.light_palette("green", as_cmap=True,reverse=True)
        display(ans.style.background_gradient(cmap = cm,axis=0))

    return ans

def summarize_auc(auc_df, beautify_display=True):
    table = pd.pivot_table(auc_df, values='metrics: explanation_auc', index=['dataset','noise_type', 'Referee'],
                        columns=['XAI_method'], aggfunc=np.sum)
    if beautify_display:
        cm = sns.light_palette("green", as_cmap=True,reverse=True)
        display(table.style.background_gradient(cmap = cm,axis=1))

    return table