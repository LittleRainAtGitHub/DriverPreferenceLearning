# DriverPreferenceLearning
Learning Lane-change preference of drivers

## Introduction

This project is to learn the lane-change preference of different drivers in multiple lane-change tasks. For every task, the dataset contains the state features of drivers and the host vehicle in the lane-change process, with corresponding lane-change status of the driver in each scene, namely Lane Keeping (LK), Preparing Lane-Change (PLC), and Lane Changing (LC). The aim of this project is to train a classification network. The network can infer the lane-change status of drivers in terms of current state features.

In our experiments, we trained a three-layer classification network to learn the lane-change preference with three different learning settings:

- **All-in-One Learning**
  
  In the learning process, the learner is presented with a compounded dataset that accumulates all history data from previous tasks.
  
- **Multi-task Learning**
  
  Multi-task learning process is also constructed to make the network learn from a stream of tasks. In this setting, the learner stores the trained network when learning from every task, and update the network only with training data in new task.
  
- **LifeLong Learning with A-GEM**
  
  Similar to multi-task learning process, the learner of LLL is presented with a stream of lane-change tasks, and it continuously updates the network with the training data in new task. In this setting, A-GEM creates a small episodic memory and makes change to the loss function to maintain the past experience. Detalis of LLL-AGEM can be found at [Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420)

## Dataset

Our simulated-based dataset contains the state features of 3 different drivers in lane-change scenario. Specifically, every state contains 7 features, namely the longitudinal and the lateral position of the host vehicle at time step t, the longtudinal and lateral distance of the front vehicle at time step t, and the heading angle, velocity and steering wheel angle of host vehicle at time step t. The state at time step t can therefore be represented as below.

**s**<sub>t</sub> = \[x<sub>h,t</sub>    y<sub>h,t</sub>    &theta;<sub>h,t</sub>    x<sub>f,t</sub>    y<sub>f,t</sub>    v<sub>h,t</sub>    &alpha;<sub>h,t</sub>\]

In each scenario, the lane-change status of drivers is discriminated into three categories, namely Lane Keeping (LK), Preparing Lane-Change (PLC), and Lane Changing (LC), shown as below. For each time step, the lane-change status of the driver are manually discriminated and labeled.

<img src="https://user-images.githubusercontent.com/45302863/162692338-629dbca5-3b20-4038-aa3e-9045f4e9f55e.png" width="600"/>
<figcaption align = "center"><b>Fig.1 - Lane-change scenario and three status of driver</b></figcaption>

<p><br></p>
Each driver was instructed to perform the lane-change task for multiple times. Ultimately, our dataset contains data from 9, 10, and 4 lane-change tasks for driver #1, #2 and #3 respectively.

<p><br></p>

\* The details of data collecting can be found in our previous paper [Transferable Driver Behavior Learning via Distribution Adaption in the Lane Change Scenario](https://ieeexplore.ieee.org/abstract/document/8813781).

\* In this repositry, only **data in lane-change task #1 and #2 of driver #3** are included for demonstration.

## Code

- **DriverP_AllinOne_train** 
  
  Training code with data from all tasks wrapped in a compounded dataset. To switch the data source from different drivers and tasks, simply change `driverid`, `lcid` (abbr for lane-change id) and `task_size`.

- **DriverP_AGEM_train**
  
  Training code with data from a stream of tasks, the use of A-GEM is optional. When setting `AGem = True`, the code practices LLL with A-GEM; when setting `AGem = False`, the code practices Multi-task learning without using A-GEM.
  _*This code will be released with the coming paper._

- **DriverP_HypsSelection**
  
  Pre-training code for hyperparameter selection. The training data should be in a relatively small scale (e.g., data from only 1 or 2 tasks). In LLL with A-GEM, all hyperparameters are fixed.

  After a run, the above codes save the trained network in .\model, and the run_result & evaluation_result in .\results. The run_result stores the loss, accuracy and duration in the training process of each epoch and task, as well as the selected hyper-parameters. The evaluation_result stores the classification accuracy on validation sets of all tasks after the training of each task.

- **DriverP_Eval**: 

  Load the saved network and test its performance on validation sets of all tasks. This is to generate confusion matrix of the test results. The plotting of confusion matrix uses code from [pretty-print-confusion-matrix](https://github.com/wcipriano/pretty-print-confusion-matrix) by Wagner Cipriano.

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

<img src="https://user-images.githubusercontent.com/45302863/162710349-4a72a5c8-8ef4-49fa-a520-aad0a7f4ab7a.svg" width="500"/>

- **Multi-task Learning**:

<img src="https://user-images.githubusercontent.com/45302863/162710376-89209105-e5da-4886-b74b-580b1ba3e65b.svg" width="500"/>

- **LLL with A-GEM**:

<img src="https://user-images.githubusercontent.com/45302863/162710390-bba8034e-6ee6-4933-a3a4-ab0195685c80.svg" width="500"/>


The performance of LLL-AGEM is obviously better than that of Multi-task Learning, and almost the same to AllinOne Learning.
While AllinOne Learning stored all task data and went over all data during training, LLL-AGM used only about 1/3 the size of memory used in AllinOne Learning, and consumed only about 1/3 the time consumed in AllinOne training process. In real-time application, this can effectively save the memory and computational cost when driving data accumulate fast and the computational/memory source is limited.
 
 
More experiments are still undergoing, and the overall results will be released and analyzed in a coming paper.


## Reference

- [Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420)
- [Transferable Driver Behavior Learning via Distribution Adaption in the Lane Change Scenario](https://ieeexplore.ieee.org/abstract/document/8813781)
- [Confusion Matrix in Python: plot a pretty confusion matrix (like Matlab) in python using seaborn and matplotlib](https://github.com/wcipriano/pretty-print-confusion-matrix)
- [PyTorch - Python Deep Learning Neural Network API - DeepLizard](https://deeplizard.com/learn/video/NSKghk0pcco)


## Dependencies
- Python 3.9
- [Pytorch 1.11](https://pytorch.org/)
- Seaborn
- Numpy
- Matplotlib
- Pandas
