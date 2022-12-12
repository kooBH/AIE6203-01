# AIE6203-01
2022-2 강화학습 - gymnasium KungFuMaster  

![v3_14400](https://user-images.githubusercontent.com/39723411/207061052-9ed61df0-4706-4475-80ad-1d53f1e4faa0.gif)

## Usage

see ```train_d0.sh``` and ```src/train.py``` for training.  

see ```demo.ipynb``` and ```eval.sh``` and ```src/eval.py``` for evaluation.   

see [Experiments](https://docs.google.com/spreadsheets/d/1hte_S2b6fM9T9JFwS2mAJ9D0NY0Bejd6tPiN8hBqMhs/edit?usp=sharing) for configurations and results. 

+ Note : There is hard-coded log directory in ```config/default.yaml```.log.root 

## Note
The team that has been maintaining Gym since 2021 has moved all future development to Gymnasium.   

https://gymnasium.farama.org/  
https://gymnasium.farama.org/environments/atari/kung_fu_master/


## REFERENCE   
  
Codes are based on work from https://github.com/dsinghnegi/atari_RL_agent with [Apache-2.0 license](https://github.com/dsinghnegi/atari_RL_agent/blob/master/LICENSE). Major changes are done for old gym->gymnasium compatible works. 
