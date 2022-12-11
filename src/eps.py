import numpy as np

def linear_decay(hp,cur_step):

    if cur_step >= hp.train.total_steps:
        return hp.eps.linear.end
    return (hp.eps.linear.start* (hp.train.total_steps - cur_step) +
            hp.eps.linear.end * cur_step) / hp.train.total_steps
def fixed(hp):
    return hp.eps.fixed.eps