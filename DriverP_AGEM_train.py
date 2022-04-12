import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict


from DriverP_Eval import Eval_CMplot
from mytools import (
    DriverDataset
    , TVData
    , MemDataset
    , CurMemBatch

    , NetworkFactory
    , RunBuilder
    , TaskRunManager

    , get_grad
    , unpack_grads
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

    # Task datasets
    driverid = 3
    label_type = '3' 
    driverpath = '.\driverdata\data_split'
    driversets = {}
    task_size = 4 # total number of lane-change task

    for lcid in range(1,task_size+1):
        driversets[f'lc{lcid}'] = DriverDataset(
            driverpath = driverpath
            , driverid = driverid
            , lcid = [lcid]
            , label_type = label_type
            )

    # Using AGEM or not
    AGem = True


    # Network hyper-parameters setting
    params =OrderedDict(
        lr = [.01]
        , batch_size = [100]
        , shuffle = [True]
        , device = ['cuda']
    )
    runs = RunBuilder.get_runs(params) # return product of hyps
    run = runs[0] # stick hyps (pre-selected)

    # Memory dataset settings
    mem_size = 4500 # total memory
    Memset = MemDataset(mem_size, task_size)


    m = TaskRunManager()

    device = torch.device(run.device)
    network = NetworkFactory.get_network(FL3).to(device)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    for lcid in range(1,task_size+1):

        
        train_loader, _ = TVData(dataset=driversets[f'lc{lcid}']
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
                loss_train = F.cross_entropy(preds,labels)

                network.zero_grad()
                loss_train.backward(retain_graph=True)
                grad_train = get_grad(network) # get grad_train

                
                ## REF FORWARD

                ## Index to Newly-updated Memset and sample ref_batch for A-GEM
                if lcid>1:
                    ref_batch = CurMemBatch(Memset=Memset
                                            , cur_task=lcid
                                            , batch_size=run.batch_size)
                    states_ref = ref_batch[0].to(device)
                    labels_ref = ref_batch[1].to(device)
                else:    # when lcid=1, let batch_ref = train_ref 
                    states_ref = states
                    labels_ref = labels

                preds_ref = network(states_ref)
                loss_ref = F.cross_entropy(preds_ref,labels_ref)

                network.zero_grad()
                loss_ref.backward()
                grad_ref = get_grad(network) #get grad_ref

            

                ## UPDATE GRAD_P
                with torch.no_grad():
                    k1 = torch.matmul(grad_train, grad_ref)
                    k2 = torch.matmul(grad_ref, grad_ref)
                    if k1 >= 0:
                        grad_p = grad_train
                    else:
                        if AGem:
                            grad_p = grad_train - k1/k2*grad_ref
                        else:
                            grad_p = grad_train  # test general learning process

                ## UNPACK GRAD_P BACK TO network.parameter.grad
                network.zero_grad()
                unpack_grads(network, grad_p)
                
                optimizer.step()

                m.track_loss(loss_train,batch)
                m.track_num_correct(preds,labels)

            # end batch loop
            
            m.end_epoch() # return time,loss,correct in this epoch

        # end epoch loop
            
        # display network performance in this task and 
        # collect validation results on all tasks
        val_all = val_on_all(task_size, driversets, network, run)
        m.end_task(val_all) 
        
        # Update Memset by sampling data from current training set
        Memset.update(lcid, driversets[f'lc{lcid}']) 

        # end task loop


    run_name = f'AGEM_{AGem}_run_results_driver{driverid}_label{label_type}'
    eval_name = f'AGEM_{AGem}_eval_results_driver{driverid}_label{label_type}'
    
    m.save_run(run_name)
    m.save_task(eval_name)

    model_name = f'AGEM_{AGem}_driver{driverid}_label{label_type}'
    m.save_model(network, model_name)

    Eval_CMplot(model_name)

if __name__ == "__main__":
    train()
    