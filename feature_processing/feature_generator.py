import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pickle
import datetime
import os
import sys
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

 
    
class Generator():
    def __init__(self,cohort,input_features,feat_cond,feat_lab,feat_proc,feat_med,impute,saved_data_path,include_time=14,chunk_size=1,agg_interval=24):
        self.impute=impute
        self.feat_cond,self.feat_proc,self.feat_med,self.feat_lab = feat_cond,feat_proc,feat_med,feat_lab
        self.cohort=cohort
        self.input_features=input_features
        self.include_time=include_time
        self.saved_data_path=saved_data_path
        self.agg_interval=agg_interval

        if not os.path.exists(f"{self.saved_data_path}/processed_admission_features_dict/{self.cohort}/agg_interval_{self.agg_interval}h"):
            os.makedirs(f"{self.saved_data_path}/processed_admission_features_dict/{self.cohort}/agg_interval_{self.agg_interval}h")
        if not os.path.exists(f"{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h"):
            os.makedirs(f"{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h")  
        
        print("[ READ COHORT ]")
        self.admissions = self.generate_adm()
        
        self.admissions[['subject_id','hadm_id']].to_csv(f'{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h/patients.csv',index=False)  
        self.hids=self.admissions['hadm_id'].unique()
        self.pts =self.admissions['subject_id'].unique()
        if(self.feat_cond):
            self.cond=self.cond[self.cond['hadm_id'].isin(self.admissions['hadm_id'])]
            
        
        print("[ READ ALL FEATURES ]")
        self.generate_feat()


        #print("[ DESCRETIZE DATA INTO CHUNKS ]")
        #self.descretize_data_daily(chunk_size)
        
        
    def generate_feat(self):
        
        if(self.feat_cond):
            print("[ ======READING DIAGNOSIS ]")
            self.generate_cond()
        if(self.feat_proc):
            print("[ ======READING PROCEDURES ]")
            self.generate_proc()
        if(self.feat_med):
            print("[ ======READING MEDICATIONS ]")
            self.generate_meds()
        if(self.feat_lab):
            print("[ ======READING LABS ]")
            #self.generate_labs_daily()
            self.generate_labs_temporal()
    
    
    def generate_adm(self):
        data=pd.read_csv(f"{self.saved_data_path}/cohorts/{self.cohort}.csv.gz", compression='gzip', header=0, index_col=None)
        print('Admissions generated, #Admissions', data['hadm_id'].nunique())
        return data
    
    
    # def generate_labs_daily(self):
    #     loading_chunksize = 1000000
    #     final=pd.DataFrame()
    #     for labs in tqdm(pd.read_csv(f"{self.saved_data_path}/features/{self.input_features}.csv.gz", compression='gzip', header=0, index_col=None,chunksize=loading_chunksize)):
    #         labs=labs[labs['hadm_id'].isin(self.admissions['hadm_id'])]
    #         if "uker" in self.cohort.lower():
    #             labs['days_from_discharge'] = labs['dischtime'] - labs['date']

    #         if "mimic" in self.cohort.lower():
    #             labs['date'] = pd.to_datetime(labs['date'], format='mixed', errors='coerce').dt.date
    #             labs['admittime'] = pd.to_datetime(labs['admittime'], format='mixed', errors='coerce').dt.date
    #             labs['dischtime'] = pd.to_datetime(labs['dischtime'], format='mixed', errors='coerce').dt.date
    #             labs['days_from_discharge'] = (pd.to_datetime(labs['dischtime']) - pd.to_datetime(labs['date'])).dt.days

                
    #         labs['int'] = self.include_time - labs['days_from_discharge']
    #         if final.empty:
    #             final=labs
    #         else:
    #             final = pd.concat([final, labs], ignore_index=True)
    #     self.labs=final
    #     print('#final lab items',self.labs['itemid'].nunique())
    #     print('#lab admissions',self.labs['hadm_id'].nunique())
    #     chunk_size =1 #every one day
    #     self.descretize_data(chunk_size)
      
        
        # def descretize_data(self,chunk_size):

        #     final_labs=pd.DataFrame()
        #     if(self.feat_lab):
        #         self.labs=self.labs.sort_values(by=['int'])
        #     t=0
        #     for i in tqdm(range(0,self.include_time +1,chunk_size)):

        #         ###LABS
        #         if(self.feat_lab):
        #             sub_labs=self.labs[(self.labs['int']>=i) & (self.labs['int']<i+chunk_size)].groupby(['hadm_id','itemid']).agg({'subject_id':'max','value':'mean'}).reset_index()
        #             sub_labs['int']=t
        #             if final_labs.empty:
        #                 final_labs=sub_labs
        #             else:    
        #                 final_labs = pd.concat([final_labs, sub_labs], ignore_index=True)
                
        #         t=t+1

        #     ###CREATE DICT
        #     num_chunks=int(self.include_time+1/chunk_size) 
        #     final_labs = final_labs[['subject_id', 'hadm_id', 'itemid','value', 'int']]
        #     self.create_Dict(final_labs,num_chunks)
        #     #final_labs.to_csv(f"{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/labs_daily.csv",index=False)
            
        # return final_labs  

    def generate_labs_temporal(self):
        
        loading_chunksize = 1000000
        final=pd.DataFrame()
        print(f"Discretization: {self.agg_interval} h")
        print((f"Days before Discharge: {self.include_time} d"))
        for labs in tqdm(pd.read_csv(f"{self.saved_data_path}/features/{self.input_features}.csv.gz", compression='gzip', header=0, index_col=None,chunksize=loading_chunksize)):
            labs=labs[labs['hadm_id'].isin(self.admissions['hadm_id'])]
            
            if "mimic" in self.cohort.lower():
                labs["dischtime"] = pd.to_datetime(labs["dischtime"], errors='coerce')   
                labs["date"] = pd.to_datetime(labs["date"], errors='coerce')   
                labs["start_time"] = (labs["dischtime"] - pd.DateOffset(days=self.include_time)).apply(lambda x: x.replace(hour=0, minute=0, second=0))
                labs["minute"] = (labs["date"] - labs["start_time"]).dt.total_seconds() / 60
                labs['int'] = (labs['minute'] // (60 * self.agg_interval)).astype(int)
                
                
            if "uker" in self.cohort.lower():
                labs["start_time"] = labs["dischtime"] - self.include_time
                labs["int"] = ((labs["date"] - labs["start_time"])).astype(int)

            labs = labs[labs['int'] >= 0] #remove the labs out of include time
            if final.empty:
                final=labs
            else:
                final = pd.concat([final, labs], ignore_index=True)
        self.labs=final

        self.labs = self.labs.groupby(['hadm_id','itemid','int']).agg({'subject_id':'max','value':'mean'}).reset_index()
        self.labs = self.labs[['subject_id', 'hadm_id', 'itemid','value', 'int']]
        
        print('#final lab items',self.labs['itemid'].nunique())
        print('#lab admissions',self.labs['hadm_id'].nunique())
        
        #self.labs.to_csv(f"{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/labs_temporal.csv",index=False)
        num_chunks=self.labs['int'].nunique()
        self.create_Dict(self.labs,num_chunks)
        
    def create_Dict(self,labs,num_chunks):
        
        
        print("[ CREATING DATA DICTIONARIES ]")
        dataDic={}
        labels_csv=pd.DataFrame(columns=['hadm_id','label'])
        labels_csv['hadm_id']=pd.Series(self.hids)
        labels_csv['label']=0
        
        for hid in self.hids:
            grp=self.admissions[self.admissions['hadm_id']==hid]
            dataDic[hid]={'Lab':{}}
            labels_csv.loc[labels_csv['hadm_id']==hid,'label']=int(grp['label'])
            
        for hid in tqdm(self.hids):
            #23928292
            #for hid in [20004811]:
            grp=self.admissions[self.admissions['hadm_id']==hid]
            demo_csv=grp[['age','gender']]
            if not os.path.exists(f"{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h/"+str(hid)):
                os.makedirs(f"{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h/"+str(hid))
            demo_csv.to_csv(f"{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h/"+str(hid)+"/demo.csv",index=False)
            dyn_csv=pd.DataFrame()
            
            ###LABS
            if(self.feat_lab):
                features=labs['itemid'].unique() # 422 unique
                hadm_data=labs[labs['hadm_id']==hid]
                
                if hadm_data.shape[0]==0:
                    continue
                    '''feature_values=pd.DataFrame(np.zeros([num_chunks,len(features)]),columns=features)
                    feature_values=feature_values.fillna(0)
                    feature_values.columns=pd.MultiIndex.from_product([["LAB"], feature_values.columns])'''

                else:
                    # save raw lab data for later preprocessing
                    hadm_data.to_csv(f'{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h/'+str(hid)+'/raw_labs.csv',index=False)  
                    
                    feature_values=hadm_data.pivot_table(index='int',columns='itemid',values='value')
                    hadm_data = hadm_data.copy()


                    hadm_data['feature_values'] = 1
                    feature_values_mask=hadm_data.pivot_table(index='int',columns='itemid',values='feature_values')
                    add_indices = pd.Index(range(num_chunks)).difference(feature_values_mask.index)              # find missing timestamps
                    add_df = pd.DataFrame(index=add_indices, columns=feature_values_mask.columns).fillna(np.nan) # add missing timestamps as an empty row (e.g. no measurements recorded in one day) 
                    feature_values_mask=pd.concat([feature_values_mask, add_df])
                    feature_values_mask=feature_values_mask.sort_index()
                    feature_values_mask=feature_values_mask.fillna(0)
        
                    feature_values=pd.concat([feature_values, add_df])
                    feature_values=feature_values.sort_index()
                    # impute missing data
                    if self.impute=='Mean':
                        feature_values=feature_values.ffill()
                        feature_values=feature_values.bfill()
                        feature_values=feature_values.fillna(feature_values.mean())
                    elif self.impute=='Median':
                        feature_values=feature_values.ffill()
                        feature_values=feature_values.bfill()
                        feature_values=feature_values.fillna(feature_values.median())
                        
                    feature_values=feature_values.fillna(0)
                    
                    feature_values_mask[feature_values_mask>0]=1
                    feature_values_mask[feature_values_mask<0]=0
                    #print('final feature values',feature_values.shape)
                    
                    #dataDic containg only the available features and their values for each admissions - therefore the shape of features might be different for each admission
                    #dataDic[hid]['Lab']['signal']=feature_values_mask.iloc[:,0:].to_dict(orient="list")
                    #dataDic[hid]['Lab']['feature_values']=feature_values.iloc[:,0:].to_dict(orient="list")
                    
                    # Add missing featrures (e.g. in mimic there are 422 features but one admission might have only 50 items here we add all the missing itemids with value of 0)
                    # Ensure that al admissions have all features (422 in mimic)
                    features_df=pd.DataFrame(columns=list(set(features)-set(feature_values.columns)))  
                    feature_values=pd.concat([feature_values,features_df],axis=1)
                    feature_values=feature_values[features]
                    feature_values=feature_values.fillna(0)
                    feature_values.columns=pd.MultiIndex.from_product([["LAB"], feature_values.columns])
                    dataDic[hid]['feature_values']=feature_values.iloc[:,0:].to_dict(orient="list")
        
                    

                
                if(dyn_csv.empty):
                    dyn_csv=feature_values
                else:
                    dyn_csv=pd.concat([dyn_csv,feature_values],axis=1)
            #Save temporal data to csv
            dyn_csv.to_csv(f'{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h/'+str(hid)+'/dynamic.csv',index=False)
            grp.to_csv(f'{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h/'+str(hid)+'/static.csv',index=False)   
            labels_csv.to_csv(f'{self.saved_data_path}/processed_admission_features_csv/{self.cohort}/agg_interval_{self.agg_interval}h/labels.csv',index=False)
  
                
                
        ######SAVE DICTIONARIES##############
        metaDic={'Cond':{},'Proc':{},'Med':{},'Lab':{},'chunk_size':{}}
        metaDic['chunk_size']= num_chunks
        with open(f'{self.saved_data_path}/processed_admission_features_dict/{self.cohort}/agg_interval_{self.agg_interval}h/dataDic', 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open(f'{self.saved_data_path}/processed_admission_features_dict/{self.cohort}/agg_interval_{self.agg_interval}h/hadmDic', 'wb') as fp:
            pickle.dump(self.hids, fp)
        
        '''with open("./saved_data/processed_admission_features_dict/ethVocab", 'wb') as fp:
            pickle.dump(list(self.data['ethnicity'].unique()), fp)
            self.eth_vocab = self.data['ethnicity'].nunique()'''
            
        with open(f'{self.saved_data_path}/processed_admission_features_dict/{self.cohort}/agg_interval_{self.agg_interval}h/ageVocab', 'wb') as fp:
            pickle.dump(list(self.admissions['age'].unique()), fp)
            self.age_vocab = self.admissions['age'].nunique()
            
        '''with open("./saved_data/processed_admission_features_dict/insVocab", 'wb') as fp:
            pickle.dump(list(self.data['insurance'].unique()), fp)
            self.ins_vocab = self.data['insurance'].nunique()'''
            
            
        if(self.feat_lab):    
            with open(f'{self.saved_data_path}/processed_admission_features_dict/{self.cohort}/agg_interval_{self.agg_interval}h/labsVocab', 'wb') as fp:
                pickle.dump(list(labs['itemid'].unique()), fp)
            self.lab_vocab = labs['itemid'].unique()
            #metaDic['Lab']=self.labs_per_adm
            
        with open(f'{self.saved_data_path}/processed_admission_features_dict/{self.cohort}/agg_interval_{self.agg_interval}h/metaDic', 'wb') as fp:
            pickle.dump(metaDic, fp)
            
        print("[ SUCCESSFULLY SAVED DATA DICTIONARIES ]")

            
        
    
    
    
if __name__ == "__main__":
    
    cohort = 'mimic_cohort' #'cohort_only_readmissions_30_days' 
    input_features ='mimic_labs'#'lab_NF_cohort_14_days' 
    feat_cond, feat_lab, feat_proc, feat_med = False, True, False, False
    impute = 'Mean'
    include_time = 14
    chunk_size = 1
    saved_data_path = '/home/hpc/iwbn/iwbn102h/FLabNet/MIMIC_IV/saved_data'
    generator_instance = Generator(cohort,input_features, feat_cond, feat_lab, feat_proc, feat_med, impute, include_time, chunk_size,saved_data_path)
    
    admissions_data = generator_instance.data
    #print(admissions_data)