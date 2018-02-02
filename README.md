# Non-Local-Nets
  - This is an implement of Non-local Nets for tensorflow version. [Here](https://arxiv.org/pdf/1711.07971.pdf), you can see the paper provided by Xiaolong Wang et.al.

# Requirements
  - tensorflow > 1.0.0
  - numpy
  - tqdm

# Usages
## Download Repo
    $ git clone https://github.com/nnuyi/Non-Local-Nets
    $ cd Non-Local-Nets

## Datasets
  In this repo, I mainly focus on *MNIST* datasets.
  
  TODO:
  
    - In the following time, I will test it in *cifar10, cifar100 etc* datasets to see wheather this networks can work well
  
  You are not required to download *MNIST* datasets since I use tensorflow mnist tool to obtain this datasets, so you just run this repo like the following steps.
  
## Training
  If this is first time you run the repo, it will download *MNIST* automatically it will cost about 5 to 10 seconds, please wait for a moment. After that, you need not to download *MNIST* again since it have been downloaded at first time. Just see the following instructions for training phase:
    
      $ python main.py --is_training=True --is_testing=False
      
      # If GPU options is avaiable, you can use it as the instruction shows below:
      $ CUDA_VISIBLE_DEVICES=[no] python main.py --is_training=True --is_testing=False
      
      # notes: [no] is the device number of GPU, you can set it according to you machine
      $ CUDA_VISIBLE_DEVICES=0 python main.py --is_training=True --is_testing=False
      
## Testing
  In this repo you can will see the testing phase during training phase since I ran the test_model codes to test its performance per 5 epochs.
  If you have finished training phase and want to test it, just see the following instructions:
  
      $ python main.py --is_training=False --is_testing=True
  
# Results
  After about 30 epochs or less, you can see that the testing accuracy rate can reach to more than **$99.22%$**. And training accuracy rate can reach to **$99.91%$**. I run this repo in **Geforce GTX 1070 GPU**, it cost 8 seconds per epoch.
  
  <p align='center'><img src='./figure/figure.png'/></p>
  
# References
  - [Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)
  
# Contacts
  Email:computerscienceyyz@163.com
