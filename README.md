# DL_course_MS_HPC_IA
This is the online course for MS HPC IA program. 

## Installation

Anaconda: if you don't have it to create virtual environment, use the following code if your system is WSL
```
wget https://repo.continuum.io/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

* clone this repo, and we will call the directoy that you cloned as ${DL_course}
* Install dependencies, We use python 3.7 and pytorch >=1.7.0
```
conda create -n DL_course_env python=3.7 jupyter
conda activate DL_course_env
conda install pytorch==1.7.0 torchvision cudatoolkit=10.2 -c pytorch
cd ${DL_course}
pip install -r requirements.txt
```
## Session 1: Course on the neural network 
 * [Neural Network course slides](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session1_nn.pdf)
 * [practical session 1 "Introduction to NN"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/lab2023/intro_NN_public.ipynb)
 
## Session 2: Course on the deep-Learning world of Convolutional Neural Networks
 * [DeepLearning-convNets course slides](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session2_convnet.pdf)
 * [practical session 2 "Introduction to CNN"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/lab2023/Lab2_CNN_oublic.ipynb)
 * [Optional practical session 2b "Introduction to convNet"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/L1_Introduction_CNN_MNIST.ipynb)
 
## Session 3: Course on Semantic Segmentation, object detection, instance segmentation
 * [Semantic Segmentation, object detection, instance segmentation](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session3_detection.pdf)
 * [practical session 3 "Object detection and tracking"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/Lab3 mask_r_cnn and YOLO.ipynb)
 
## Session 4: Course on Recurrent Neural Networks (RNN)
 * [Recurrent Neural Networks](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session4_RNN.pdf)
 * [practical session 4 "Language translation"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/Lab4_RNN_LSTM_language.ipynb)
 
## Session 5:  Course on Deep reinforcement learning 
 * [deep reinforecement learning](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session5_RL.pdf)
 * [practical seesion 5 "Gaming"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/Lab5_reinforcement_learning_Q.ipynb)

## Session 6: Course on advanced DL + choice and beginning of mini-project
 * [Unsupervised Generative Deep-Learning](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session6_deep_generative_model.pdf) 
 * [MINI-PROJECTS instructions and proposed topics](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/Mini_project_MS_HPC_IA.ipynb)
