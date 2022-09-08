import numpy as np
import math
import os
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import utils.visualization as vis 

_NUM_FEATURES = 50

class LocalDataLoader():
    def __init__(self, datapath="./data", dataset='CMJ',num_features=_NUM_FEATURES):
        """ Load dataset from a local folder
        """
        self.ds_dir = datapath
        self.dataset = dataset
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.num_features = num_features        
    def get_X_y(self,onehot_label=False,synth=False):
        if synth == False:
            org_ds = ['CBF', 'CMJ', 'Coffee', 'ECG200', 'GunPoint']
            
            s = '' if self.dataset in  org_ds else '.txt'
            sep =',' if self.dataset in org_ds  else None
            train_file =  './%s/%s/%s_TRAIN%s' %(self.ds_dir,self.dataset,self.dataset,s) 
            test_file = './%s/%s/%s_TEST%s' %(self.ds_dir,self.dataset,self.dataset,s)
            # os.path.join(self.ds_dir, self.dataset, str(self.dataset)+'_TRAIN'+s)
            # test_file  = os.path.join(self.ds_dir, self.dataset, str(self.dataset)+'_TEST'+s)

            train_data = np.genfromtxt(train_file,delimiter=sep)
            test_data = np.genfromtxt(test_file, delimiter=sep)

            self.X_train = np.expand_dims(train_data[:,1:], 1)
            self.y_train = train_data[:,0]


            self.X_test = np.expand_dims(test_data[:,1:], 1)
            self.y_test = test_data[:,0]

        else:
            self.ds_dir=self.ds_dir+'/synth/'
            train_file = self.ds_dir+self.dataset+'_TRAIN.npy'
            test_file  = self.ds_dir+self.dataset+'_TEST.npy'
            train_label = self.ds_dir+self.dataset+'_TRAIN_meta.npy'
            test_label = self.ds_dir+self.dataset+'_TEST_meta.npy'
            
            train_data = np.load(train_file)
            test_data = np.load(test_file)
            train_label = np.load(train_label)
            test_label = np.load(test_label)
            
            selected_feature = self.num_features//2
            self.X_train = np.expand_dims(train_data[:,:,selected_feature],1)
            self.X_test = np.expand_dims(test_data[:,:,selected_feature],1)
            
            self.y_train = train_label[:,0]
            self.y_test = test_label[:,0]
            


        # Standardize labels 
        encoder = OneHotEncoder(categories='auto', sparse=False)
        self.y_train = encoder.fit_transform(np.expand_dims(self.y_train, axis=-1))
        self.y_test = encoder.transform(np.expand_dims(self.y_test, axis=-1))
        
        if onehot_label == False:
            self.y_train = np.argmax(self.y_train,axis=1)
            self.y_test = np.argmax(self.y_test, axis=1)


        return self.X_train, self.y_train, self.X_test, self.y_test

    
        





    def createTensorDataset(self,batch_size=64):
        self.X_train,self.y_train,self.X_test,self.y_test=self.get_X_y(onehot_label=True)

        train_dataset = TensorDataset(torch.from_numpy(self.X_train).float(),torch.from_numpy(self.y_train))
        test_dataset = TensorDataset(torch.from_numpy(self.X_test).float(),torch.from_numpy(self.y_test))
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    
        return train_loader,test_loader

    def get_loaders(self,mode='train',batch_size=64,val_size=0.2):
        self.X_train,self.y_train,self.X_test,self.y_test=self.get_X_y(onehot_label=True)
        
        self.X_train = torch.from_numpy(self.X_train).float()
        self.y_train = torch.from_numpy(self.y_train)
        self.X_test = torch.from_numpy(self.X_test).float()
        self.y_test = torch.from_numpy(self.y_test)

        if mode == 'train':
            assert val_size is not None
            X_train,y_train,X_val,y_val = train_test_split_tensor(
                self.X_train,self.y_train,
                split_size=val_size,
                )

            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=batch_size,
                shuffle=True,
                )
            val_loader = DataLoader(
                TensorDataset(X_val,y_val),
                batch_size=batch_size,
                shuffle=False,
                )
            return train_loader,val_loader
        
        else:
            test_loader = DataLoader(
                TensorDataset(self.X_test,self.y_test),
                batch_size=batch_size,
                shuffle=False
                )
            return test_loader,None

def train_test_split_tensor(X_tensor,y_tensor,split_size):
    X_train,X_test,y_train,y_test=train_test_split(
        X_tensor.numpy(),y_tensor.numpy(),
        test_size=split_size,
        )
    X_train_tensor = torch.from_numpy(X_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_train_tensor = torch.from_numpy(y_train)
    y_test_tensor = torch.from_numpy(y_test)
    return X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor

def data_summary(datapath,dataset):
    print('Dataset: %s' %(dataset))
    data = LocalDataLoader(datapath,dataset)
    X_train,y_train,X_test,y_test = data.get_X_y()

    print('X_train.shape: ', X_train.shape)
    print('X_test.shape: ', X_test.shape)
    names = [str(int(x)) for x in np.unique(y_test)]
    print('Unique target class: ', names)

    print('Dataset: %s, Training Data-Global mean value: %2.5f' % (dataset, np.mean(X_test)))
    
    vis.visualize_class_ts(X_train,y_train)


    

# def get_random_explanation(X_test, seed=None):
#     """ Get a random string of explanation with same shape as provided string
#     """
#     if seed is not None:
#         np.random.seed(seed)
#     explanation = np.random.uniform(0,100,size=X_test.shape)
#     return explanation

# def get_saved_explanation(dataset='CMJ',explanation_method ='MrSEQL-SM'):
#     """ Load a saved explanation from a local folder
#     """

#     method = explanation_method
#     LIME_explanation = ['MrSEQL-LIME', 'Rocket-LIME']
#     if method == 'MrSEQL-SM':
#         test_weight_file = 'output/explanation_weight/weights_MrSEQL_%s.txt' % ds
#     elif method == 'MrSEQL-LIME':    
#         test_weight_file = 'output/explanation_weight/weights_LIME_%s.txt' % ds
#     elif c == 'ResNetCAM':      
#         test_weight_file = 'output/resnet_weights/ResNet_%s_BestModel.hdf5_model_weights.txt' % ds
#     else: 
#         print('ERROR')
#         return

#     explanation = np.genfromtxt(test_weight_file, delimiter = ',')

#     # Convert from LIME explanation (time-slice level) to general explanation (time-step level)
#     if method in LIME_explanation:
#         explanation = np.repeat(LIME_explanation, X_test.shape[-1]//10).reshape(X_test.shape[0],-1)
#         if explanation.shape[-1] != X_test.shape[-1]: #recalibrate LIME explanation
#             last_step_explanation = np.transpose(LIME_explanation)[-1].reshape(-1,1)
#             n_pad = X_test.shape[-1] - explanation.shape[-1]
#             padded_array = np.repeat(last_step_explanation, n_pad)
#             explanation = np.append(explanation, padded_array, axis=-1)

#     return explanation

