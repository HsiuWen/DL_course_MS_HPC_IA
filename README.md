# DL_course_MS_HPC_IA
This is the online course for MS HPC IA program. 

## Installation

* Virtual environment: if you don't have it to create virtual environment, check this site for installation
[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

* clone this repo, and we will call the directoy that you cloned as ${DL_course}
* Install dependencies, We use python 3.13 and pytorch >=2.9.0
```
conda create -n DL_course_env python=3.13
conda activate DL_course_env
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
cd ${DL_course}
pip install -r requirements.txt
```

* Add your environment to jupyter notebook
```
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=DL_course_env
```

## Session 1: Course on the neural network 
 * [Neural Network course slides](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session1_nn.pdf)
 * [practical session 1 "Introduction to NN"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/lab2025/lab1/two_layer_net.ipynb)
 
## Session 2: Course on the deep-Learning world of Convolutional Neural Networks
 * [DeepLearning-convNets course slides](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session2_convnet.pdf)
 * [practical session 2 "Introduction to CNN"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/lab2025/Lab2_CNN_public.ipynb)
 
## Session 3: Course on Recurrent Neural Networks (RNN)
 * [Recurrent Neural Networks](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session4_RNN.pdf)
 * [practical session 3 "Language translation"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/lab2025/RNN_LSTM_language%20public.ipynb)

 ## Session 4: Course on Semantic Segmentation, object detection, instance segmentation
 * [Semantic Segmentation, object detection, instance segmentation](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session3_detection.pdf)
If you want to run this lab on your local environment, you need to create a new virtual enviroment and install these packages
 ```
conda create -n captioning_env
conda activate captioning_env
cd lab4
pip install -r requirements.txt
```
Possible error solution: 
(1)ModuleNotFoundError: No module named 'setuptools.extern.six'
update numpy version to: pip install numpy==2.0.0
 * [practical session 4 "RNN_Captioning"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/lab2025/lab4/Lab4_RNN_LSTM_language%20public.ipynb)

## Session 5:  Course on Deep reinforcement learning 
 * [deep reinforecement learning](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session5_RL.pdf)
 * [practical session 5 "Gaming"](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/lab2025/Lab5_reinforcement_learning_Q.ipynb)

## Session 6: Course on advanced DL + choice and beginning of mini-project
 * [Unsupervised Generative Deep-Learning](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/slides/session6_deep_generative_model.pdf) 
 * [MINI-PROJECTS instructions and proposed topics](https://github.com/HsiuWen/DL_course_MS_HPC_IA/blob/main/Mini_project_MS_HPC_IA.ipynb)
