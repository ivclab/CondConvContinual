# CondConvContinual

This is an official Pytorch implementation of Extending Conditional Convolution Structures for Enhancing Multitasking Continual Learning

Created by Cheng-Hao Tu, Cheng-En Wu and Chu-Song Chen 


## Introduction 
Conditional operations have received much attention in recent deep learning studies to facilitate the prediction accuracy of a model. A recent advance toward this direction is the conditional parametric convolutions (CondConv), whichis proposed to exploit additional capacities provided by the deep model weights to enhance the performance, whereas the computational complexity of the model is much less influenced. CondConv employs input-dependent fusion parameters that can combine multiple columns of convolution kernels adaptively for performance improvement. At runtime, the columns of kernels are on-line combined into a single one, and thus the time complexity is much less than that of employing multiple columns in a convolution layer under the same capacity. Although CondConv is effective for the performance enhancement of a deep model, it is currently applied to individual tasks only. As it has the nice property of adding model weights with computational efficiency, we extend it for multitask learning, where the tasks are presented incrementally. In this work, we introduce a sequential multitask (or continual) learning approach based on the CondConv structures, referred to as CondConv-Continual. Experimental results show that the proposed approach is effective for unforgetting continual learning. Compared to current approaches, CondConv is advantageous to offer a regular and easy-to-implement way to enlarge the neural networks for acquiring additional capacity and provides a cross-referencing mechanism for different task models to achieve comparative results.


## To be continued ... 
