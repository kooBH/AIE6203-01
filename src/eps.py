import numpy as np

def linear_decay(hp,cur_step):

    if cur_step >= hp.eps.linear.decay:
        return hp.eps.linear.end
    return (hp.eps.linear.start* (hp.eps.linear.decay - cur_step) +
            hp.eps.linear.end * cur_step) / hp.eps.linear.decay

def linear_annealing(hp,step) : 

    if step >= hp.eps.linear_annealing.decay:
        return hp.eps.linear_annealing.end
    else : 
        eps =  (hp.eps.linear_annealing.start* (hp.eps.linear_annealing.decay - step) +
            hp.eps.linear_annealing.end * step) / hp.eps.linear_annealing.decay + hp.eps.linear_annealing.annealing * ((hp.eps.linear_annealing.decay-step)/hp.eps.linear_annealing.decay) *np.cos(step/4) 
        if eps > 1 :
            eps = 1.0
        return eps


def fixed(hp):
    return hp.eps.fixed.eps