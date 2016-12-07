# Caffe with BANG (Batch Adjusted Network Gradients)

This Caffe repository contains the code of our BANG approach which can significantly improve the robustness of deep neural networks without using additional training samples or any sort of data augmentation technique. For further details about BANG, please refer to our paper: [Towards Robust Deep Neural Networks with BANG](https://arxiv.org/abs/1612.00138).

BANG is implemented for InnerProduct and Convolutional layers (CPU, GPU, and cuDNN), and it can be used to train learning models only on a single GPU.

The repository contains pre-trained LeNet models (R0 and B1 from [Table 1.](https://arxiv.org/abs/1612.00138) under models/mnist folder) and Cifar-10 models (R0 and B0 from [Table 2.](https://arxiv.org/abs/1612.00138) in models/cifar10 folder), as well as model definition files used to train those models. In order to train models regularly, you can set beta to zero for all layers or, alternatively, remove all BANG parameters (beta, epsilon, and ratio) from those files.

Please cite BANG in your publications if it helps your research:

    @article{rozsa2016towards,
      title={Towards Robust Deep Neural Networks with BANG},
      author={Rozsa, Andras and G\"unther, Manuel and Boult, Terrance E.},
      journal={arXiv preprint arXiv:1612.00138},
      year={2016}
    }


Please read below the original README of Caffe.

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
