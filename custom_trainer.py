import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image,ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm

class Trainer(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model=None
        self.train_loader=None
        self.valid_loader=None
        self.optimizer=None
        self.scheduler=None 
        self.criterion = None
        self.step_scheduler_after=None
        self.step_scheduler_metric=None
        self.train_metrics=None 
        self.valid_metrics=None
        self.plot=None
        self.batch_loss=0 #to custmize the schedulers as well. 
        self.epoch_train_loss=0
        self.epoch_val_loss=0
        self.current_epoch=0
        self.scheduler_metric=None #if you want to return metric to scheduler?
        self.learning_rate=None
        self.weight_decay=None
        
    def forward(self,inputs):
        if self.model:
            outputs=self.model(inputs)
            return outputs
        else:
            return 
    
    def train_one_step(self,inputs,targets):
        inputs,targets=inputs.to(self.device),targets.to(self.device)
        self.optimizer.zero_grad()
        outputs= self.model.forward(inputs)
        self.batch_loss=self.criterion(outputs.view(-1,1).float(),targets.view(-1,1).float()) 
        self.batch_loss.backward() 
        self.optimizer.step()
        if self.scheduler:
            if self.step_scheduler_after == "batch":
                if self.scheduler_metric:
                    self.scheduler.step(self.batch_loss)
                else:
                    self.scheduler.step()
                    
        return outputs.reshape(-1),self.batch_loss 
    
    def train_one_epoch(self,data_loader): 
        model.train()
        epoch_loss=0
        pred=[]
        true=[]
        tk0 = tqdm(data_loader, total=len(data_loader), position = 0, leave = True)
        for b_idx,batch in enumerate(tk0):
            torch.cuda.empty_cache()
            inputs=batch['inputs']
            targets=batch['targets']
            outputs,loss=self.train_one_step(inputs,targets)
            epoch_loss+=loss
            pred.append(outputs.cpu().detach().numpy())
            true.append(targets.cpu().detach().numpy())
        tk0.close()
        return np.array(pred).reshape(-1,1),np.array(true).reshape(-1,1),epoch_loss/len(data_loader) 
    
    def validate_one_step(self,inputs,targets=None):
        inputs = inputs.to(self.device, non_blocking=True)
        if targets is not None:
            targets = targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs.flatten().float(), targets.float())
            return outputs, loss 
        else:
            outputs = self.model(inputs)
            return outputs.reshape(-1), None 
    
    def validate_one_epoch(self,testloader): 
        model.eval()
        epoch_loss=0
        pred=[]
        true=[]
        tk0 = tqdm(testloader, total=len(testloader), position = 0, leave = True)
        for b_idx,batch in enumerate(tk0):
            torch.cuda.empty_cache()
            inputs=batch['inputs']
            targets=batch['targets'] 
            outputs,loss=self.validate_one_step(inputs,targets) 
            epoch_loss+=loss
            pred.append(outputs.cpu().detach().numpy())
            true.append(targets.cpu().detach().numpy())
        tk0.close() 
        return np.array(pred).reshape(-1,1),np.array(true).reshape(-1,1),epoch_loss/len(testloader) 
        
        
    def fetch_optimizer(self,*args,**kwargs):
        return
    
    def fetch_scheduler(self,*args,**kwargs):
        return 
    
    def fetch_criterion(self,*args,**kwargs):
        return
    
    def initialize(self,model,device): 
        self.model=model
        self.device=device 
        if next(self.model.parameters()).device != self.device: 
            self.model=self.model.to(self.device)
        if self.optimizer==None:
            self.optimizer=self.fetch_optimizer()
        if self.scheduler==None:
            self.scheduler=self.fetch_scheduler()
        if self.criterion==None:
            self.criterion=self.fetch_criterion()
        
            
    def update_metrics(self,train_pred,train_targets,valid_pred,valid_targets,train_loss,valid_loss):
        return # should return a dictionay with key: [values] pairs 
    
    def validate(self,dataset=None,batch_size=32): #validate on new or existing dataset 
        if dataset==None:
            if self.valid_loader!=None:
                predictions,test_loss=self.validate_one_epoch(self.valid_loader)
                return predictions,test_loss
            else:
                return None,None 
        else:
            dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)
            predictions,test_loss=self.validate_one_epoch(dataloader)
            return predictions,test_loss 
        
        
    def fit(self,train_dataset,valid_dataset,batch_size,epochs,plot_results=False): 
        self.plot=plot_results
        self.train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        self.valid_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False) 
        for epoch in range(epochs): 
            self.train_metrics=None
            self.valid_metrics=None
            self.current_epoch=epoch+1 
            train_predictions,train_targets,epoch_loss=self.train_one_epoch(self.train_loader)
            valid_predictions,valid_targets,valid_loss=self.validate_one_epoch(self.valid_loader) 
            self.epoch_val_loss=valid_loss
            self.epoch_train_loss=epoch_loss
            if self.plot:
                self.update_metrics(train_predictions,train_targets,valid_predictions,valid_targets,epoch_loss,valid_loss) 
            print(f"epoch:{self.current_epoch} : training loss: {epoch_loss} and validation loss: {valid_loss}")
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.scheduler_metric:
                        self.scheduler.step(self.epoch_val_loss) 
                    else:
                        self.scheduler.step() 
            
            torch.cuda.empty_cache()  
       
    def tune_paramters(self): 
        pass  
    
    def plot_results(self):  
        epochs = range(1, len(next(iter(self.train_metrics.values()))) + 1)  

        for key in self.train_metrics.keys(): 
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.train_metrics[key], label=f'Train {key}')
            plt.plot(epochs, self.valid_metrics[key], label=f'Validation {key}')
            plt.xlabel('Epochs')
            plt.ylabel(key.capitalize())  
            plt.title(f'{key.capitalize()} over Epochs')  
            plt.legend()

        plt.show() 

    def save_model(self): 
        pass  
