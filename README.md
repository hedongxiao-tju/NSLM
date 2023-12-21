# NSLM

Code & data accompanying the paper ["Unveiling Implicit Deceptive Patterns in Multi-modal Fake News via Neuro-Symbolic Reasoning"].


### Prerequisites
- python > 3.7
- CUDA > 11
- Prepare requirements: `pip3 install -r requirements.txt`.
- Set environment variable `$PJ_HOME`: `export PJ_HOME=/YOUR_PATH/NSLM/`.



#### Quick Start

- Go to folder `src/`.
- Run `train_*.sh`. 

#### Data
- You can refer to the papers released the Weibo and Fakeddit dataset for offical data.
- Here we provide Pre-processed text data of Weibo in the `data/weibo` folder.


* Notes: 
    - You can find the output data in the `out` folder specified in the config file.
    - Since the probability of the three deceptive patterns appearing at the same time in actual situations is very small, in the experiment we set the y corresponding to the situation where all three deceptive modes exist to 2 (that is, a category that does not exist in the dataset), so the decoder finally becomes a three-category prediction
  
#### Acknowledgment

Our implementation is mainly based on follows. Thanks for their authors. 
https://github.com/jiangjiechen/LOREN
  
    

