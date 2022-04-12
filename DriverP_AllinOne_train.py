import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict


from mytools import (
    DriverDataset
    , TVData

    , NetworkFactory
    , RunBuilder
    , TaskRunManager
    , val_on_all

)

torch.set_printoptions(linewidth=120)


def train():
    
    # Network setting

    FL3 = OrderedDict([
    ('Input', nn.Linear(in_features=7, out_features=32))
    ,('relu1', nn.ReLU())
    ,('Hidden', nn.Linear(in_features=32, out_features=32))
    ,('relu2', nn.ReLU())
    #    ,('Hidden2', nn.Linear(in_features=32, out_features=32))
    #    ,('relu3', nn.ReLU())
    ,('Output', nn.Linear(in_features=32, out_features=3))
        ])

    # Network hyper-parameters setting

    params =OrderedDict(
        lr = [.01]
        , batch_size = [100]
        , shuffle = [True]
        , device = ['cuda']
    )
    runs = RunBuilder.get_runs(params) # return product of hyps
    run = runs[0] # stick hyps



    # Task datasets

    driverid = 3
    task_size = 4
    driverpath = '.\driverdata\data_split'
    label_type = '3'
    driversets = {}
    for lcid in range(1,task_size+1):
        driversets[f'lc{lcid}'] = DriverDataset(
            driverpath = driverpath
            , driverid = driverid
            , lcid = [lcid]
            , label_type = label_type
            )

    # AllinOne dataset

    driverset = DriverDataset(
        driverpath = '.\driverdata\data_split'
        , driverid = driverid
        , lcid = range(1,task_size+1)
        , label_type = label_type
        )


    m = TaskRunManager()

    device = torch.device(run.device)
    network = NetworkFactory.get_network(FL3).to(device)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

        
    train_loader, _ = TVData(dataset=driverset
                                    , split=.2
                                    , shuffle = run.shuffle
                                    , batch_size = run.batch_size)


    m.begin_task(lcid, task_size, run, train_loader)

    for epoch in range(10):

        m.begin_epoch() # initialize time,loss,correct in this epoch

        for batch in train_loader:

            ## TRAIN FORWARD

            states = batch[0].to(device)
            labels = batch[1].to(device)

            preds = network(states)
            loss_train = F.cross_entropy(preds, labels)

            network.zero_grad()
            loss_train.backward()
            optimizer.step() 

            m.track_loss(loss_train, batch)
            m.track_num_correct(preds, labels)

        # end batch loop
        
        m.end_epoch() # return time,loss,correct in this epoch
        
    # display performance in this task and 
    # collect cross_val results
    val_all = val_on_all(task_size, driversets, network, run)
    m.end_task(val_all) 

    run_name = f'AllinOne_run_results_driver{driverid}_label{label_type}'
    eval_name = f'AllinOne_eval_results_driver{driverid}_label{label_type}'
    
    m.save_run(run_name)
    m.save_task(eval_name, all = True)

    model_name = f'AllinOne_driver{driverid}_label{label_type}'
    m.save_model(network, model_name)



if __name__ == "__main__":
    train()