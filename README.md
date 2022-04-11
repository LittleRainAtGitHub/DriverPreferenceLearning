# DriverPreferenceLearning
Learning Lane-change preference of drivers

## Introduction

This project is to learn the lane-change preference of different drivers in several lane-change tasks. For every task, the dataset contains the state features of drivers and the host vehicle in the lane-change process, with corresponding lane-change status of the driver in each scene, namely Lane Keeping (LK), Preparing Lane-Change (PLC), and Lane Changing (LC). The aim of this project is to train a classification network. The network can infer the lane-change status of drivers in terms of current state features.

In the learning process, the learner is presented with a compounded dataset that accumulates all history data from previous tasks. Training on all task data guarantees the prediction accuracy of the network when testing on every history task. However, in real-world application, the learner needs to acquire new skills to adapt to the changes in the environment. In this case, the learner has to continuously update the network w.r.t. newly-collected data from new tasks. A solution is to continously increment dataset and go over all data to retrain the network, but this process can be very computational and memory consuming, since all previous data has to be stored. 

To address this problem, multi-task learning process can be utilized to enable the network learn from a stream of tasks. In this setting, the learner only stores the trained network when learning from every task, and update the network only with training data in new task. However, the network may suffer catastrophic forgetting if new data relates little to the previous experience, leading to performance drop when testing on previous tasks. Moreover, since the network cannot reuse the experience from previous tasks, it may even requires copious amounts of labeled data from new tasks in order to obtain high prediction performance.

To learn from new tasks efficiently and maintain the experience accumulated in the past, LLL with A-GEM is adopted in this project. Similar to multi-task learning process, the learner of LLL is presented with a stream of lane-change tasks, and it continuously updates the network with the training data in new task. The difference is, in this setting, the learner can remember and reuse the knowledge acquired in the previous lane-change tasks. This is achieved by A-GEM, which leverages a small episodic memory and a small change to the loss function to maintain the past experience. By incrementally building a data-driven prior which may be leveraged to speed up learning of a new task, the LLL learner enables the network to quickly learn from new tasks with limited training samples, time and memory. Since the past experience is constantly retained and updated, the network can also maintain high prediction performance in previous tasks. 

In our experiments, we trained a three-layer classification network to learn the lane-change preference with the above introduced learning settings:
- **All-in-One Learning** (train with compounded data from all tasks)
- **Multi-task learning** (train with streams of data from multiple tasks without using A-GEM)
- **LifeLong learning with A-GEM** (train with streams from multiple tasks using LLL with A-GEM)

## Dataset

Our simulated-based dataset contains the state features of 3 different drivers in lane-change scenario. Specifically, every state contains 7 features, namely the longitudinal and the lateral position of the host vehicle at time step t, the longtudinal and lateral distance of the front vehicle at time step t, and the heading angle, velocity and steering wheel angle of host vehicle at time step t. The state at time step t can therefore be represented as below.

![image](https://user-images.githubusercontent.com/45302863/162686957-64b98615-848d-49ad-acc7-7d8d2940bff4.png)

In each scenario, the lane-change status of drivers is discriminated into three categories, namely Lane Keeping (LK), Preparing Lane-Change (PLC), and Lane Changing (LC), shown as below. For each time step, the lane-change status of the driver are manually discriminated and labeled.

![image](https://user-images.githubusercontent.com/45302863/162692338-629dbca5-3b20-4038-aa3e-9045f4e9f55e.png)

Each driver was instructed to perform the lane-change task for multiple times. Ultimately, our dataset contains data from 9, 10, and 4 lane-change tasks for driver #1, #2 and #3 respectively.

The details of data collecting can be found in our previous paper [Transferable Driver Behavior Learning via Distribution Adaption in the Lane Change Scenario](https://ieeexplore.ieee.org/abstract/document/8813781).

\* In this repositry, only **data in lane-change task #1 and #2 of driver #3** are included for demonstration.

## Code

- **DriverP_AllinOne_train**: Training code with data from all tasks wrapped in a compounded dataset. To switch the data source from different drivers and tasks, simply change `driverid`, `lcid` (abbr for lane-change id) and `task_size`.

- **DriverP_AGEM_train**: Training code with data from a stream of tasks, the use of A-GEM is optional. When setting `AGem = True`, the code practices LLL with A-GEM; when setting `AGem = False`, the code practices Multi-task learning without using A-GEM.

- **DriverP_HypsSelection**: Training code for hyperparameter selection before LLL with A-GEM. The training data should be in a relatively small scale (e.g., data from only 1 or 2 tasks). Then in LLL with A-GEM, all hyperparameters are fixed.

  After a run, the code saves the trained network in .\model, and the run_result & evaluation_result in .\results. The run_result stores the loss, accuracy and duration in the training process of each epoch and task, as well as the selected hyper-parameters. The evaluation_result stores the classification accuracy on validation sets of all tasks after the training of each task.

- **DriverP_Eval**: Load the saved network and test its performance on validation sets of all tasks. This is to generate confusion matrix of the test results. The plotting of confusion matrix uses code from [pretty-print-confusion-matrix](https://github.com/wcipriano/pretty-print-confusion-matrix) by Wagner Cipriano.

- **mytools**: Utilities mainly for:
  - Dataset initializing
  - Data Loading
  - Train/Validation dataset spliting
  - Network initializing
  - Run Builder and Run Manager that stores the settings and results in each run (reference to codes from [DeepLizard](https://deeplizard.com/learn/video/NSKghk0pcco))
  - Memory_dataset settings and gradient operations for A-GEM

## Results

Here shows the confusion matrix of the accuracy of network trained over 4 tasks of driver #3.

- **AllinOne Learning**: 

![AllinOne_driver3_label3_ConfusionMatrix](https://user-images.githubusercontent.com/45302863/162710349-4a72a5c8-8ef4-49fa-a520-aad0a7f4ab7a.svg)

- **Multi-task Learning**:

![AGEM_False_driver3_label3_ConfusionMatrix](https://user-images.githubusercontent.com/45302863/162710376-89209105-e5da-4886-b74b-580b1ba3e65b.svg)

- **LLL with A-GEM**:

![AGEM_True_driver3_label3_ConfusionMatrix](https://user-images.githubusercontent.com/45302863/162710390-bba8034e-6ee6-4933-a3a4-ab0195685c80.svg)

The performance of LLL-AGEM is obviously better than that of Multi-task Learning, and almost the same to AllinOne Learning. This indicates LLL-AGEM can efficiently
 learn from new tasks while avoiding forgetting previous knowledge. While AllinOne Learning stored all task data and went over all data during training, LLL-AGM used only about 1/3 the size of memory used in AllinOne Learning, and consumed only about 1/3 the time consumed in AllinOne training process. In real-time application, this can effectively save the memory and computational cost when driving data accumulate fast and the computational/memory source is limited.
 
 
More experiments are still undergoing, and the overall results will be released and analyzed in a coming paper.


## Reference

- [Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420)

- [Transferable Driver Behavior Learning via Distribution Adaption in the Lane Change Scenario](https://ieeexplore.ieee.org/abstract/document/8813781)

- [Confusion Matrix in Python: plot a pretty confusion matrix (like Matlab) in python using seaborn and matplotlib](https://github.com/wcipriano/pretty-print-confusion-matrix)

- [PyTorch - Python Deep Learning Neural Network API](https://deeplizard.com/learn/video/NSKghk0pcco)


## Dependencies
- Python 3.9
- [Pytorch 1.11](https://pytorch.org/)
- Seaborn
- Numpy
- Matplotlib
- Pandas
