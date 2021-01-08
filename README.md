# CondConvContinual

This is an official Pytorch implementation of [Extending Conditional Convolution Structures for Enhancing Multitasking Continual Learning](http://www.apsipa.org/proceedings/2020/pdfs/0001605.pdf)

Created by Cheng-Hao Tu, Cheng-En Wu and Chu-Song Chen

The code is released for academic research use only. For commercial use, please contact [Prof. Chu-Song Chen](chusong@csie.ntu.edu.tw).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extending-conditional-convolution-structures/continual-learning-on-imagenet-50-5-tasks)](https://paperswithcode.com/sota/continual-learning-on-imagenet-50-5-tasks?p=extending-conditional-convolution-structures)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extending-conditional-convolution-structures/continual-learning-on-cifar100-20-tasks)](https://paperswithcode.com/sota/continual-learning-on-cifar100-20-tasks?p=extending-conditional-convolution-structures)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extending-conditional-convolution-structures/continual-learning-on-flowers-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-flowers-fine-grained-6?p=extending-conditional-convolution-structures)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extending-conditional-convolution-structures/continual-learning-on-imagenet-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-imagenet-fine-grained-6?p=extending-conditional-convolution-structures)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extending-conditional-convolution-structures/continual-learning-on-sketch-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-sketch-fine-grained-6?p=extending-conditional-convolution-structures)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extending-conditional-convolution-structures/continual-learning-on-wikiart-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-wikiart-fine-grained-6?p=extending-conditional-convolution-structures)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extending-conditional-convolution-structures/continual-learning-on-cubs-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-cubs-fine-grained-6?p=extending-conditional-convolution-structures)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/extending-conditional-convolution-structures/continual-learning-on-stanford-cars-fine)](https://paperswithcode.com/sota/continual-learning-on-stanford-cars-fine?p=extending-conditional-convolution-structures)

## Introduction 
Conditional operations have received much attention in recent deep learning studies to facilitate the prediction accuracy of a model. A recent advance toward this direction is the conditional parametric convolutions (CondConv), whichis proposed to exploit additional capacities provided by the deep model weights to enhance the performance, whereas the computational complexity of the model is much less influenced. CondConv employs input-dependent fusion parameters that can combine multiple columns of convolution kernels adaptively for performance improvement. At runtime, the columns of kernels are on-line combined into a single one, and thus the time complexity is much less than that of employing multiple columns in a convolution layer under the same capacity. Although CondConv is effective for the performance enhancement of a deep model, it is currently applied to individual tasks only. As it has the nice property of adding model weights with computational efficiency, we extend it for multitask learning, where the tasks are presented incrementally. In this work, we introduce a sequential multitask (or continual) learning approach based on the CondConv structures, referred to as CondConv-Continual. Experimental results show that the proposed approach is effective for unforgetting continual learning. Compared to current approaches, CondConv is advantageous to offer a regular and easy-to-implement way to enlarge the neural networks for acquiring additional capacity and provides a cross-referencing mechanism for different task models to achieve comparative results.


## Prerequisites  
* python==3.8 
* torch==1.2.0 
* torchvision==0.4.0 
* tqdm==4.51.0
* [torchnet](https://github.com/pytorch/tnt)


## Usage 

### Prepare the data 

For the CIFAR-100 Twenty Tasks experiment, you can download the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and split the images into 20 tasks based on their super classes using the [cifar2png](https://github.com/knjcode/cifar2png) tool. Or you can just download the converted version of our CIFAR-100 [here](https://drive.google.com/file/d/1eo2RhMmhxzUNOZa0Z7jy7y4lOn3lqddU/view?usp=sharing). Unzip the compressed file, rename the extracted folder as `CIFAR100_20Tasks/` and place it under `data/`. You can see our training, validation and testing splits for each task in `splits/CIFAR100_20Tasks/` and the json files inside reveal the directory structure that our program uses to load the data. 

For the ImageNet-50 Five Tasks experiment, download the the ImageNet dataset from its [official website](http://image-net.org/download) and place the downloaded images under `data/`. Similarly, please see the json files in `splits/ImageNet50_5Tasks/` to adjust the directory structure so that our program can load the data correctly. 


### Training models from scratch 

Please use the following command to train the model for each task from scratch. Note that our program for continual learning will directly use the scratch model as the 1st task. Therefore, at least for ***the 1st task***, this step is required before we can run our continual learning process.

```
bash scripts/{EXP_NAME}/trainval_baseline.sh {GPU_ID} {START_IDX} {END_IDX} {BACKBONE}
```

The {EXP_NAME} should be CIFAR100_20Tasks or ImageNet50_5Tasks, and the {BACKBONE} should be Conv4 or ResNet18. If you want to use multi-gpu training, simply assign multiple gpu ids in {GPU_ID}. This command will train scratch models for task indices range from {START_IDX} to {END_IDX} (inclusive). Examples of using this command can be seen as follows. 

```
bash scripts/CIFAR100_20Tasks/trainval_baseline.sh 0,1,2,3 1 20 Conv4 
```

or 

```
bash scripts/ImageNet50_5Tasks/trainval_baseline.sh 0,1,2,3 1 5 ResNet18 
```


### Finetuning models 

Please use the following command to, for each task, finetune models from previous tasks. For example, for the 4th task, the program will finetune the models from the 1st, the 2nd and the 3rd tasks to the 4th task, and store the resulting 3 models seprately. Currently, we only provide this command for the CIFAR-100 Twenty Tasks experiment. 

```
bash scripts/CIFAR100_20Tasks/trainval_finetune.sh {GPU_ID} {START_IDX} {END_IDX} {BACKBONE}
```

The arguments are similar to those in the script for training scratch models as described above. 


### Conditional convolution continual learning 

Please use the following command to run our continual learning process. 

```
bash scripts/{EXP_NAME}/trainval_condconti.sh {GPU_ID} {START_TASK_IDX} {BACKBONE}
```

Similarly, please use CIFAR100_20Tasks or ImageNet50_5Tasks for {EXP_NAME}, and use Conv4 or ResNet18 for {BACKBONE}. The {START_TASK_IDX} indicates the task index we want to resume from. At beginning, this is set to 1. Please see the following commands as examples. 

```
bash scripts/CIFAR100_20Tasks/trainval_condconti.sh 0,1,2,3 1 Conv4 
```

or 

```
bash scripts/ImageNet50_5Tasks/trainval_condconti.sh 0,1,2,3 1 ResNet18 
```


### Evaluation 

For the CIFAR-100 Twenty Tasks experiment, the evaluation is under the setting with task boundary. For the ImageNet-50 Five Tasks experiment, we compute the results under the setting *without* task boundary. Their evaluations can be completed using the following command. 

```
bash scripts/{EXP_NAME}/evaluate_condconti.sh {GPU_ID} {BACKBONE} {FINAL_TASK_IDX}
```

The {EXP_NAME}, {GPU_ID} and {BACKBONE} arguments are exactly the same as the description above. The {FINAL_TASK_IDX} indicates the last task of the continual learning process whose model will be used for inference. We give examples of using this command. 

```
bash scripts/CIFAR100_20Tasks/evaluate_condconti.sh 0 Conv4 20 
```

or 

```
bash scripts/ImageNet50_5Tasks/evaluate_condconti.sh 0 ResNet18 5 
```

Note that the above two commands run `evaluate_continual_with_boundary.py` and `evaluate_continual_without_boundary.py`, respectively, for different evaluation settings. The results will be stored in a json file under `CHECKPOINTS/Continual/{EXP_NAME}/{BACKBONE}/`. 


## Citation 

    @inproceedings{tu2020extending,
    title={Extending Conditional Convolution Structures For Enhancing Multitasking Continual Learning},
    author={Tu, Cheng-Hao and Wu, Cheng-En and Chen, Chu-Song},
    booktitle={2020 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
    pages={1605--1610},
    year={2020},
    organization={IEEE}
    }

## Contact 
Please feel free to leave suggestions or comments to [Cheng-Hao Tu](andytu28@iis.sinica.edu.tw), [Cheng-En Wu](chengen@iis.sinica.edu.tw), [Chu-Song Chen](chusong@csie.ntu.edu.tw)
