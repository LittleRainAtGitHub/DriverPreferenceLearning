import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from collections import OrderedDict


from mytools import (
    DriverDataset
    , NetworkFactory
    , RunBuilder
    , RunManager
    , TVData
)

torch.set_printoptions(linewidth=120)

driversets = {}
for lcid in range(1,10):
    driversets[f'lc{lcid}'] = DriverDataset(
        driverpath = '.\driverdata\data_split'
        , driverid = 1
        , lcid = [lcid]
        )



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
    lr = [.01, .001]
    , batch_size = [100]
    , shuffle = [True]
    , device = ['cuda']
    # , trainset = ['lc1','lc3']
    # , valset= ['lc1','lc3']
)




m = RunManager()


for run in RunBuilder.get_runs(params):

    device = torch.device(run.device)
    network = NetworkFactory.get_network(FL3).to(device)
    
    
    train_loader, _ = TVData(dataset=driversets[run.trainset]
                                    , split=.2
                                    , shuffle = run.shuffle
                                    , batch_size = run.batch_size)
    _, validation_loader = TVData(dataset=driversets[run.valset]
                                    , split=.2
                                    , shuffle = run.shuffle
                                    , batch_size = run.batch_size)

    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, train_loader)
    for epoch in range(10):

        m.begin_epoch()

        for batch in train_loader:

            # batch = next(iter(train_loader))   # get batch
            states = batch[0].to(device)
            labels = batch[1].to(device)

            preds = network(states)         # pass batch
            loss = F.cross_entropy(preds, labels)   # calculate loss

            optimizer.zero_grad()
            loss.backward()     # update network.weight.grad
            optimizer.step()    # using grad to update parameters(weights)

            m.track_loss(loss,batch)
            m.track_num_correct(preds,labels)

        
        m.end_epoch(network, validation_loader, device)
        # print( "epoch:", epoch, " total_correct:", total_correct, " loss:", total_loss)

    m.end_run()
    
m.save('hyps_results')