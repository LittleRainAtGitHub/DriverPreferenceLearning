import torch
import torch.nn as nn

import numpy as np
import os
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset, DataLoader
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product
from collections import OrderedDict
from collections import namedtuple

import matplotlib.pyplot as plt

class MemDataset(Dataset):
    '''
    Create a Memory Dataset that storages data from the current task.

    The total memory size and sampling size from every task should be designated 
    when initialization.

    MemDataset.update(cur_task, Dataset_t) samples data in task_size from the current task
    dataset and map the task_th data into corresponding reserved space.

    '''
    def __init__(self, mem_size, task_size, transform=None, target_transform=None):

        # initialize an empty dataset
        self.mem_size = mem_size
        self.sample_size = int(mem_size/task_size)
        self.current_task = 0

        self.states = torch.zeros(self.mem_size,7) # 7 is the dimension of features in states
        self.labels = torch.zeros(self.mem_size)

        self.transform = transform
        self.target_transform = target_transform
        
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        state_instance = self.states[idx]
        label_instance = int(self.labels[idx])

        if self.transform:
            state_instance = self.transform(state_instance)
        if self.target_transform:
            label_instance = self.target_transform(label_instance)
        return state_instance, label_instance 

    def update(self, cur_task, Dataset_t):
        
        sample_loader = DataLoader(Dataset_t
                                    , batch_size=self.sample_size
                                    , shuffle=True)

        states_task, labels_task = next(iter(sample_loader))
        
        self.states[(cur_task-1)*self.sample_size:(cur_task)*self.sample_size,:] = states_task 
        self.labels[(cur_task-1)*self.sample_size:(cur_task)*self.sample_size] = labels_task
        self.current_task = cur_task

def CurMemBatch(Memset, cur_task, batch_size):
    ref_indices = list(range((cur_task-1)*Memset.sample_size)) # index to last updated Memdata
    np.random.shuffle(ref_indices)
    ref_sampler = SubsetRandomSampler(ref_indices)
    ref_loader = DataLoader(Memset, batch_size=batch_size, sampler=ref_sampler)
    ref_batch = next(iter(ref_loader))

    return ref_batch

class DriverDataset(Dataset):
    def __init__(self, driverpath, driverid, lcid, label_type='3'):
        
        #  get path
        self.driverpath = driverpath
        self.lcid = lcid
        self.label_type = label_type

        # read lane change [id] data and concatenate
        self.start = 0
        for id in lcid:
            
            self.statep = f'driver{driverid}_{id}_stateRecord.csv'
            self.label3p = f'driver{driverid}_{id}_labelRecord3.csv'
            self.actionp = f'driver{driverid}_{id}_actionRecord.csv'
            self.label2p = f'driver{driverid}_{id}_labelRecord.csv'

            # read label3, label, state, action
            self.label3_id = pd.read_csv(os.path.join(driverpath, self.label3p)).values
            self.label2_id = pd.read_csv(os.path.join(driverpath, self.label2p)).values
            self.state_id = pd.read_csv(os.path.join(driverpath, self.statep)).values
            self.action_id = pd.read_csv(os.path.join(driverpath, self.actionp)).values

            if self.start==0:
                self.state = self.state_id
                self.label3 = self.label3_id
                self.action = self.action_id
                self.label2 = self.label2_id  
            else:
                self.state = np.concatenate((self.state,self.state_id), axis=0)
                self.label3 = np.concatenate((self.label3,self.label3_id), axis=0)
                self.action = np.concatenate((self.action,self.action_id), axis=0)
                self.label2 = np.concatenate((self.label2,self.label2_id), axis=0)
            self.start += 1

        self.classes_label3 = ['bf/aft lane change', 'during lane change', 'prepare lane change']
        self.classes_label2 = ['bf/aft lane chagne', 'during lane change']


    def __len__(self):
        return len(self.label3)

    def __getitem__(self, idx):
        
        state_instance = torch.tensor(self.state[idx], dtype=torch.float32)
        if self.label_type == '2':
            label_instance = int(self.label2[idx])
        else:
            label_instance = int(self.label3[idx])

        return state_instance, label_instance

def TVData(dataset, split, shuffle, batch_size):

    random_seed = 50
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split * dataset_size))

    if shuffle :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset
                                , batch_size=batch_size
                                , sampler=train_sampler
                                )

    validation_loader = torch.utils.data.DataLoader(dataset
                                , batch_size=batch_size
                                , sampler=valid_sampler
                                )

    return train_loader, validation_loader

# TODO: reset network parameters compatibly
class NetworkFactory():
    @staticmethod
    def get_network(name):
        if name:
            torch.manual_seed(50)
            return nn.Sequential(
                     nn.Linear(in_features=7, out_features=32)
                    ,nn.ReLU()
                    ,nn.Linear(in_features=32, out_features=32)
                    ,nn.ReLU()
                    #    ,('Hidden2', nn.Linear(in_features=32, out_features=32))
                    #    ,('relu3', nn.ReLU())
                    ,nn.Linear(in_features=32, out_features=3)
                    )

        else:
            return None

class RunBuilder():
    @staticmethod
    def get_runs(params):
        
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None


    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        # self.tb = SummaryWriter(comment=f'-{run}')

        # states, labels = next(iter(self.loader))
        # grid = torchvision.utils.make_grid(states)

        # self.tb.add_image('states', grid)
        # self.tb.add_graph(self.network, states.to(getattr(run, 'device', 'cpu')))

    def end_run(self):
        # self.tb.close()
        self.epoch_count = 0

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait=True)
        display(df)

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0


    def end_epoch(self, network, validation_loader, device):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / len(self.loader.sampler.indices)
        accuracy = self.epoch_num_correct / len(self.loader.sampler.indices)
        
        with torch.no_grad():
            val_preds, val_labels = get_all_preds_labels(network, validation_loader,device)
            val_correct = get_num_correct(val_preds, val_labels)
            val_accuracy = val_correct / len(validation_loader.sampler.indices)
        

        # self.tb.add_scalar('Loss', loss, self.epoch_count)
        # self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        # for name, param in self.network.named_parameters():
        #     self.tb.add_histogram(name, param, self.epoch_count)
        #     self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results["validation accuracy"] = val_accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        
        self.run_data.append(results)

        # df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        # clear_output(wait=True)
        # display(df)

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        num_correct = preds.argmax(dim=1).eq(labels).sum().item()
        self.epoch_num_correct += num_correct
 

    # @torch.no_grad()
    # def get_all_preds_labels(self, model, loader, device):
    #     all_preds = torch.tensor([]).to(device)
    #     all_labels = torch.tensor([]).to(device)
    #     for batch in loader:
    #         states = batch[0].to(device)
    #         labels = batch[1].to(device)
    #         preds = model(states) # predict using trained model

    #         all_preds = torch.cat(
    #             (all_preds, preds),
    #             dim=0
    #         )
    #         all_labels = torch.cat(
    #             (all_labels, labels)
    #         )
            
    #     return all_preds, all_labels

    # def get_num_correct(self, preds, labels):
    #     return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):

        pd.DataFrame.from_dict(
            self.run_data
            , orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
    
def get_grad(network):
    grad = []
    for param in network.parameters():
        a = param.grad.view(-1).clone()
        grad.append(a)
    grad = torch.cat(grad, dim=0)
    
    return grad

def unpack_grads(network, grad_p):
    len_param = []
    shape_param = []
    split_param = []
    for param in network.parameters():
        len_param.append(len(param.view(-1)))
        split_param.append(sum(len_param))
        shape_param.append(list(param.size()))

    split_grad = torch.tensor_split(grad_p, split_param[:-1])

    i = 0
    new_grad = []
    for param in network.parameters():
        new_grad.append(torch.reshape(split_grad[i], shape_param[i]))
        param.grad = new_grad[i]
        i +=1

@torch.no_grad()
def check_param_max(network,check):
    i = 0
    maxx = []
    for param in network.parameters():
        if check=='grad':
            maxx.append(param.grad.max())
        if check=='param':
            maxx.append(param.max())
        i +=1
    return maxx


class TaskRunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.cur_task = 0
        self.task_size = 0

        self.run_data = []
        self.task_data = []
        self.task_start_time = None

        self.loader = None
        self.result_path = './results'
        

    def begin_task(self, cur_task, task_size, run, loader):

        self.task_start_time = time.time()

        self.run_params = run
        self.cur_task = cur_task
        self.task_size = task_size

        self.loader = loader

    def end_task(self, val_all):

        self.epoch_count = 0

        # display training results of epochs in this task
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        clear_output(wait=True)
        display(df)

        # collect validation results on all task sets in this task
        i = 0
        results = OrderedDict()
        for task in range(self.task_size):
            results[f'v_lc{task+1}'] = val_all[i]
            i += 1
    
        self.task_data.append(results)
        # return self.task_data


    def begin_epoch(self):
        self.epoch_start_time = time.time()
        
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0


    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.task_start_time
        loss = self.epoch_loss / len(self.loader.sampler.indices)
        accuracy = self.epoch_num_correct / len(self.loader.sampler.indices)
        
        # with torch.no_grad():
        #     val_preds, val_labels = get_all_preds_labels(network, validation_loader,device)
        #     val_correct = get_num_correct(val_preds, val_labels)
        #     val_accuracy = val_correct / len(validation_loader.sampler.indices)
        
        results = OrderedDict()
        results["current task"] = self.cur_task
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        # results["validation accuracy"] = val_accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        
        self.run_data.append(results)

        # df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        # clear_output(wait=True)
        # display(df)

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        num_correct = preds.argmax(dim=1).eq(labels).sum().item()
        self.epoch_num_correct += num_correct
 

    def save_run(self, fileName):

        os.makedirs(self.result_path, exist_ok=True)

        pd.DataFrame.from_dict(
            self.run_data
            , orient='columns'
        ).to_csv(os.path.join(self.result_path, f'{fileName}.csv'))

        with open(os.path.join(self.result_path, f'{fileName}.json'), 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
    
    def save_task(self, fileName, all = False):

        os.makedirs(self.result_path, exist_ok=True)

        if all:
            pd.DataFrame(
            self.task_data
            # , orient='columns'
            , index = ['AllinOne']
            ).to_csv(os.path.join(self.result_path, f'{fileName}.csv'))
        else:
            pd.DataFrame(
                self.task_data
                # , orient='columns'
                , index = [f't_lc{x}' for x in range(1,self.task_size+1)]
            ).to_csv(os.path.join(self.result_path, f'{fileName}.csv'))

        with open(os.path.join(self.result_path, f'{fileName}.json'), 'w', encoding='utf-8') as f:
            json.dump(self.task_data, f, ensure_ascii=False, indent=4)

    def save_model(self, network, model_name):
        model_path = './model/'
        os.makedirs(model_path, exist_ok=True)
        torch.save(network.state_dict(), os.path.join(model_path, f'{model_name}.pt'))



def val_on_all(task_size, datasets, network, run):

    val_all = []
    for task in range(task_size):

        _, validation_loader = TVData(dataset=datasets[f'lc{task+1}']
                                            , split=.2
                                            , shuffle = run.shuffle
                                            , batch_size = run.batch_size)

        with torch.no_grad():
            val_preds, val_labels = get_all_preds_labels(network, validation_loader,run.device)
            val_correct = get_num_correct(val_preds, val_labels)
            val_accuracy = val_correct / len(validation_loader.sampler.indices)

        val_all.append(val_accuracy)
    
    return val_all

@torch.no_grad()
def get_all_preds_labels(model, loader, device):
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    for batch in loader:
        states = batch[0].to(device)
        labels = batch[1].to(device)
        preds = model(states) # predict using trained model

        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels)
        )
        
    return all_preds, all_labels

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class DriverDataset_all(Dataset):
    def __init__(self, driverpath, driverid, label_type='3'):
        
        #  get path
        self.driverpath = driverpath
        self.label_type = label_type

        self.statep = f'driver{driverid}_stateRecord.csv'
        self.label3p = f'driver{driverid}_labelRecord3.csv'
        self.actionp = f'driver{driverid}_actionRecord.csv'
        self.label2p = f'driver{driverid}_labelRecord.csv'

        # read label3, label, state, action
        self.label3 = pd.read_csv(os.path.join(driverpath, self.label3p)).values
        self.label2 = pd.read_csv(os.path.join(driverpath, self.label2p)).values
        self.state = pd.read_csv(os.path.join(driverpath, self.statep)).values
        self.action = pd.read_csv(os.path.join(driverpath, self.actionp)).values


    def __len__(self):
        return len(self.label3)

    def __getitem__(self, idx):
        
        state_instance = torch.tensor(self.state[idx], dtype=torch.float32)
        if self.label_type == '2':
            label_instance = int(self.label2[idx])
        else:
            label_instance = int(self.label3[idx])

        return state_instance, label_instance

