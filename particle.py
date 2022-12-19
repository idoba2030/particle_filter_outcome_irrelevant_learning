import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def _transition_function(particles, action,ch_key ,reward, N):
    # TODO
    alpha = particles[2,:]
    alpha = 1 / (1 + np.exp(-alpha)) # 0<alpha<1
    # q dynamics
    particles[action,:] = particles[action,:] + alpha*(reward - particles[action,:])
    particles[ch_key+4,:] = particles[ch_key+4,:] + alpha*(reward - particles[ch_key+4,:])
    # alpha beta dynamics
    particles[6] += np.random.normal(0, 0.05, N)
    particles[7] += np.random.normal(0, 0.05, N)
    particles[8] += np.random.normal(0, 0.05, N)
    return particles
def _observation_likelihood(particles, action,card_left,card_right):
    eps = 1e-7
    q_offered_left = particles[card_left-1,:]
    q_offered_right = particles[card_right-1,:]
    q_key_left = particles[4,:]
    q_key_right = particles[5,:]
    q_net_left = q_offered_left + particles[8,:]*q_key_left
    q_net_right = q_offered_right + particles[8,:]*q_key_right
    beta = (np.exp(particles[3,:])).clip(0,10)
    p_0 = ( np.exp( beta*q_net_left ) ) / ( np.exp(beta*q_net_left) + np.exp(beta*q_net_right) )
    likelihood = p_0 if action==0 else 1-p_0
    return likelihood , -np.log( likelihood.mean()+eps )
def _resampling_systematic(w, N):
    #num_particles = len(w)
    u = np.random.random()/N
    edges = np.concatenate((0, np.cumsum(w)), axis=None)
    samples = np.arange(u, 1,step=(1/N))
    idx = np.digitize(samples,edges)-1
    return idx
def _predict(particles, weights, observations, N):
    particles = _transition_function(particles, observations[0],observations[1], observations[4], N)
    state = particles@weights
    return state, particles
def _correct(particles, observation, weights, N):
    likelihood, run_bce = _observation_likelihood(particles, observation[0],observation[1],observation[2])
    weights = weights
    weights = weights*likelihood
    weights = weights/sum(weights)
    N_eff = 1/np.sum(weights**2)
    resample_percent = 0.50
    Nt = resample_percent*N
    idx = np.arange(N, dtype=int)
    if N_eff < Nt:
        idx = _resampling_systematic(weights,N)
        weights = np.ones(N)/N
        particles = particles[:, idx]
    return idx, particles, weights, likelihood, run_bce
def _pf(obs):
    N = 5000
    num_params = 9
    num_observations = len(obs)
    observations = obs
    particles = np.zeros(shape=(4,N))
    particles[0] = np.random.normal(0,1,N) # q_0
    particles[1] = np.random.normal(0,1,N) # q_1
    particles[2] = np.random.normal(0,1,N) # q_2
    particles[3] = np.random.normal(0,1,N) # q_3
    particles[4] = np.random.normal(0,1,N) # q_key0
    particles[5] = np.random.normal(0,1,N) # q_key1
    particles[6] = np.random.normal(0,3,N) # alpha
    particles[7] = np.random.normal(1,2,N) # beta
    particles[8] = np.random.normal(0,1,N) # omega
    all_particles = np.zeros(shape=(num_params,N,num_observations))
    state_arr = np.zeros(shape=(num_observations, num_params))
    weights_arr = np.zeros(shape=(num_observations, N))
    likelihood_arr = np.zeros(shape=(num_observations, N))
    weights = np.ones(N)/N
    total_bce = 0
    for t in range(num_observations):
        idx, particles, weights, likelihood, cur_bce = _correct(particles, observations[t], weights, N)
        state, particles = _predict(particles, weights, observations[t], N)
        state_arr[t,:] = state
        likelihood_arr[t,:] = likelihood
        weights_arr[t,:] = weights
        all_particles[:,:,t] = particles
        total_bce += cur_bce
    return state_arr, likelihood_arr, weights_arr, all_particles, (total_bce/num_observations)
df=pd.read_csv('df.csv')
df.rename(columns = {'selected_offer':'action'}, inplace = True)
df[['action']]-=1
obs = df[['action','ch_key','card_left','card_right','reward']].values
params, likelihood, weights, particles, mean_nll = _pf(obs)