
import torch
import torch.nn as nn
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from pretty_confusion_matrix import pp_matrix
from mytools import (
    DriverDataset
    , TVData
    , get_all_preds_labels
    , NetworkFactory
    , RunBuilder

)


def Eval_CMplot(model_name):
    '''
    Evaluate the model on all task
    under modifying
    '''


    # Reload the trained network

    FL3 = OrderedDict([
    ('Input', nn.Linear(in_features=7, out_features=32))
    ,('relu1', nn.ReLU())
    ,('Hidden', nn.Linear(in_features=32, out_features=32))
    ,('relu2', nn.ReLU())
    #    ,('Hidden2', nn.Linear(in_features=32, out_features=32))
    #    ,('relu3', nn.ReLU())
    ,('Output', nn.Linear(in_features=32, out_features=3))
        ])

    params =OrderedDict(
        lr = [.01]
        , batch_size = [100]
        , shuffle = [True]
        , device = ['cuda']
    )
    runs = RunBuilder.get_runs(params) # return product of hyps
    run = runs[0] # stick hyps (pre-selected)

    model_path = './model/'
    m_path = os.path.join(model_path, f'{model_name}.pt')

    network = NetworkFactory.get_network(FL3).to(run.device)
    network.load_state_dict(torch.load(m_path), strict=False)

    # Evaluate on validation datasets of all tasks

    network.eval()

    driverid = 3
    task_size = 4
    driversets = {}
    for lcid in range(1,task_size+1):
        driversets[f'lc{lcid}'] = DriverDataset(
            driverpath = './driverdata/data_split'
            , driverid = driverid
            , lcid = [lcid]
            )

    val_preds = []
    val_labels = []
    for task in range(task_size):

        _, validation_loader = TVData(dataset=driversets[f'lc{task+1}']
                                            , split=.2
                                            , shuffle = run.shuffle
                                            , batch_size = run.batch_size)

        with torch.no_grad():
            val = get_all_preds_labels(network, validation_loader,run.device)
            val_preds.append(val[0].argmax(dim=1).to(dtype=torch.int32))
            val_labels.append(val[1].to(dtype=torch.int32))
            # val_correct = get_num_correct(val_preds, val_labels)
            # val_accuracy = val_correct / len(validation_loader.sampler.indices)

        # val_all.append(val_accuracy)

    # Plot confusion matrix

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

    cm = confusion_matrix(val_labels.to('cpu'), val_preds.to('cpu'))

    classes=['bf/aft lc','dur lc','prep lc']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    cmap = 'Blues'
    cmtf,_ = pp_matrix(df_cm, cmap=cmap, title=model_name)

    cmtf.savefig(f'./results/{model_name}_ConfusionMatrix.svg',format='svg')


if __name__ == "__main__":
    model_name = 'AllinOne_driver3_label3'
    Eval_CMplot(model_name)