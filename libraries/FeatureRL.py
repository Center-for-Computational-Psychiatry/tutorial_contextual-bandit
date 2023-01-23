# Feature reinforcement learning class. 

# Instantiates a feature reinforcement learning agent that is learing
# in a multidimensional environment with discrete features.

# Available methods: 
# Likelihood computation for: 
#       o Choice data
#       o Gaze data
#       o Jointly fitting choice and gaze data 

import numpy as np
from scipy.special import logsumexp
from scipy.stats import dirichlet
import warnings
import pandas as pd
import seaborn as sns

# Custom dependencies
import sys
import os
sys.path.append(os.getcwd()) 
from World import World
from Data import extract_vars

# Start Agent class.
class Agent(object):
    """ Container for agent properties and methods.

    Parameters.
    ----------
    world: 
        instance of World.
    eta: float
        Learning rate.
    eta_k: float
        Decay rate.
    beta_value_choice: float
        Softmax temperature for choice linking function.
    beta_value_gaze: float
        Softmax temperature for attention linking function.
    beta_center_dim: float
        Attention bias to center dimension.
    beta_center_feat: 
        Attention bias to center feature.
    w_init: float
        Initial feature values.
    decay_target: float
        Value to decay feature weights towards. 
    precision: float
        Precision for Dirichlet attention linking function. 
    ----------
    """

    ###############################
    ## Initialize agent properties.
    ###############################
    def __init__(self, world, params):
        """ Sets agent parameters.
        """

        self.eta = params['learning_rate']
        self.eta_k = params['decay_rate']

        self.beta_value_choice = params['beta_value_choice']
        self.beta_value_gaze = params['beta_value_gaze']
        self.beta_center_dim = params['beta_center_dim']
        self.beta_center_feat = params['beta_center_feat'] 

        self.w_init = params['w_init']
        self.dt = params['decay_target']
        self.precision = params['precision']

    def softmax(self, v, mode):
        """ Softmax action selection for an arbitrary number of actions with values v.
            Ref. on logsumexp: https://blog.feedly.com/tricks-of-the-trade-logsumexp/

            Parameters
            ----------
            v: array, float
                Array of action values.

            mode: string
                'choice' or 'gaze'.

            Returns
            -------
            p_c: array, float bounded between 0 and 1
                Probability distribution over actions
            a: int 
                Chosen action.
        """

        ## Convert values to choice probabilities.
        if mode == 'choice': 
            v_b = self.beta_value_choice * v;
        elif mode == 'gaze': 
            v_b = self.beta_value_gaze * v;

        p_c = np.exp(v_b - logsumexp(v_b));

        ## Uniformly sample from cumulative distribution over p_c.
        a = np.nonzero(np.random.random((1,)) <= np.cumsum(p_c))[0][0] + 1

        return p_c, a

    ######################
    ## Simulation methods.
    ######################

    def run_without_choice(self, world, stimuli, outcomes, center_dim, center_feat):

        """ Runs one simulation of the RL model given a particular sequence of observations.

            Parameters
            ----------
            world: instance of World.

            stimuli: int, shape(n_trials, n_feats)
                Sequence of single stimuli, expanded coding as defined in World.make_stimuli

            outcomes: int, shape(n_trials, 1)
                Sequence of outcomes.

            center_dim: int
                Center dimension. 

            center_feat: int, shape(n_trials, 1)
                Sequence of center features.

            Returns
            -------

            W: float, array(n_trials, n_feats)
                Feature values on each trial.

            A: float, array(n_trials, n_feats)
                Simulated attention vector on each trial.

        """

        ## Get number of trials.
        n_trials = len(outcomes)

        ## Initialize output.
        # Feature values.
        W = np.empty((0, world.n_feats))
        # Simulated attention. 
        A = np.empty((0, world.n_feats))

        ## Initialize feature weights. 
        w = self.w_init * np.ones(world.n_feats)

        ## Loop through trials.
        for t in np.arange(n_trials):

            ## Generate an attention vector.
            # Center bias. 
            center_bias = np.zeros(9)
            if center_dim == 1: center_bias[0:3] = self.beta_center_dim
            if center_dim == 2: center_bias[3:6] = self.beta_center_dim
            if center_dim == 3: center_bias[6:9] = self.beta_center_dim
            center_bias[center_feat[t]-1] = self.beta_center_feat
            # Map to Dirichlet alphas.
            a = self.beta_value_gaze * w + center_bias;
            trial_alphas = np.exp(a - logsumexp(a));
            trial_alphas =  trial_alphas*self.precision
            trial_alphas = trial_alphas + 0.001
            print('FRL alphas', trial_alphas)
            # Draw sample. 
            a = np.random.dirichlet(trial_alphas)

            ## Store feature values and attention.
            W = np.vstack((W, w.T))
            A = np.vstack((A, a.T))

            ## Observe stimulus.
            stimulus = stimuli[t].astype(int)

            ## Observe outcome.
            outcome = outcomes[t].astype(int)

            ## Update chosen weights.
            pe = outcome-np.sum(w[stimulus-1])
            w[stimulus-1] = w[stimulus-1] + self.eta * pe

            ## Decay unchosen weights.
            all_feats = np.arange(world.n_feats)+1
            unchosen_feats = all_feats[~np.isin(all_feats, stimulus)]
            w[unchosen_feats-1] = (1-self.eta_k) * w[unchosen_feats-1]
            
        return W, A

    ########################
    ## Likelihood functions.
    ########################

    def choice_likelihood(self, world, extracted_data):
   
        """ Returns the log likelihood of a sequence of choices. 
            
            Parameters
            ----------
            world: instance of World.

            extracted_data: dictionary of extracted variables. 
            
            Contains: 

            stimuli_1, stimuli_2, stimuli_3: int, shape(n_trials, n_dims)
                Each available stimulus, expanded coding as defined in World.make_stimuli. 

            choices: int, shape(n_trials, n_dims)
                Sequence of chosen stimuli, expanded feature coding as defined in World.make_stimuli 

            actions: int, shape(n_trials, 1)
                Sequence of chosen actions.

            outcomes: int, shape(n_trials, 1)
                Sequence of outcomes. 

            center: int, shape(n_trials,2)
                Center dimension and center feature.

            et_data: float, shape(n_trials, n_feats) 
                Sequence of attention measurements. These are proportional vectors
                that sum to 1 and have the dimensionality of the number of features.

            Returns
            -------
            w_all: float, array(n_trials, n_feats)
                Learned feature weights.

            log_lik: float
                Log-likelihood of choices.
        """

        ## Remap dictionary to necessary local variables. 
        outcomes = extracted_data["outcomes"]
        stimuli_1 = extracted_data["stimuli_1"]
        stimuli_2 = extracted_data["stimuli_2"]
        stimuli_3 = extracted_data["stimuli_3"]
        choices = extracted_data["choices"]
        actions = extracted_data["actions"]
       
        ## Get number of trials.
        n_trials = len(outcomes)

        ## Preallocate value array.
        w_all = np.ones((n_trials, world.n_feats)) * np.nan
        
        ## Initialize feature weights.
        W = self.w_init * np.ones(world.n_feats)

        ## Initialize likelihood.
        log_lik = 0

        ## Loop through trials. 
        for t in np.arange(n_trials):

            ## Store current W.
            w_all[t,:] = W

            ## Grab stimuli. 
            stimulus_1 = stimuli_1[t,:].astype(int)
            stimulus_2 = stimuli_2[t,:].astype(int)
            stimulus_3 = stimuli_3[t,:].astype(int)

            ## Compute current value. 
            V = np.full(world.n_dims, np.nan)
            V[0] = np.sum(W[stimulus_1-1])
            V[1] = np.sum(W[stimulus_2-1])
            V[2] = np.sum(W[stimulus_3-1])
      
            ## Compute action likelihood.
            p_c, a = self.softmax(V, mode='choice')
            log_p_c = np.log(p_c)
            trial_lik = log_p_c[actions[t].astype(int)-1]
            log_lik = log_lik + trial_lik

            ## Observe outcome. 
            outcome = outcomes[t].astype(int)

            ## Grab current choice. 
            choice = choices[t].astype(int)

            ## Update chosen weights.
            pe = outcome-np.sum(W[choice-1])
            W[choice-1] = W[choice-1] + self.eta * pe

            ## Decay unchosen weights.
            all_feats = np.arange(world.n_feats)+1
            unchosen_feats = all_feats[~np.isin(all_feats, choice)]

            W[unchosen_feats-1] = (1-self.eta_k) * W[unchosen_feats-1]

        return w_all, log_lik

    def attention_likelihood(self, world, extracted_data, feature_level=True):

        """ Returns the log likelihood of a sequence of attention measurements. 
            
            Parameters
            ----------
            world: instance of World.

            extracted_data: dictionary of extracted variables. 
            
            Contains: 

            stimuli_1, stimuli_2, stimuli_3: int, shape(n_trials, n_dims)
                Each available stimulus, expanded coding as defined in World.make_stimuli. 

            choices: int, shape(n_trials, n_dims)
                Sequence of chosen stimuli, expanded feature coding as defined in World.make_stimuli 

            actions: int, shape(n_trials, 1)
                Sequence of chosen actions.

            outcomes: int, shape(n_trials, 1)
                Sequence of outcomes. 

            center: int, shape(n_trials,2)
                Center dimension and center feature.

            et_data: float, shape(n_trials, n_feats) 
                Sequence of attention measurements. These are proportional vectors
                that sum to 1 and have the dimensionality of the number of features.

            Returns
            -------

            w_all: float, array(n_trials, n_feats)
                Learned feature weights.

            log_lik: float
                Log-likelihood of attention measurements.
        """

        ## Remap dictionary to necessary local variables. 
        outcomes = extracted_data["outcomes"]
        choices = extracted_data["choices"]
        center = extracted_data["center"]
        et_data = extracted_data["et_data"]

        ## Get number of trials
        n_trials = len(outcomes)

        ## Preallocate value array.
        w_all = np.ones((n_trials, world.n_feats)) * np.nan
        
        ## Initialize feature weights.
        W = self.w_init * np.ones(world.n_feats)

        ## Initialize likelihood.
        log_lik = 0

        ## Loop through trials. 
        for t in np.arange(n_trials):

            ## Store current W.
            w_all[t,:] = W
            
            ## Grab observed attention.
            if feature_level == True:
                trial_data = et_data[t,:]
            else:
                # This needs to be generalized.
                trial_data = np.zeros(world.n_dims)
                trial_data[0] = np.sum(et_data[t,0:3])
                trial_data[1] = np.sum(et_data[t,3:6])
                trial_data[2] = np.sum(et_data[t,6:9])

            # Add small constant and re-normalize (avoids 0s)
            trial_data = trial_data + 0.001
            trial_data = trial_data / sum(trial_data)
            # Raise exception if data input is invalid.
            if np.min(trial_data) <= 0:
                print(trial_data)
                raise Exception('Dirichlet data should be greater than 0.')
                
            ## Map to Dirichlet alphas. 
            ## FEATURE LEVEL. 
            if feature_level == True:

                ## Set center bias component of alphas.
                # This needs to be generalized...
                center_bias = np.zeros(9)
                if center[t,0] == 1: center_bias[0:3] = self.beta_center_dim 
                if center[t,0] == 2: center_bias[3:6] = self.beta_center_dim
                if center[t,0] == 3: center_bias[6:9] = self.beta_center_dim
                center_bias[center[t,1]-1] = self.beta_center_feat

                a = self.beta_value_gaze * W + center_bias;
                
            ## DIMENSION LEVEL. 
            else:
                ## Set center bias component of alphas.
                center_bias = np.zeros(world.n_dims)
                center_bias[center[t,0]-1] = self.beta_center_dim

                # This needs to be generalized...
                a = np.zeros(world.n_feats_per_dim)
                a[0] = np.max(W[0:3])
                a[1] = np.max(W[3:6])
                a[2] = np.max(W[6:9])

                a = self.beta_value_gaze * a + center_bias;

            trial_alphas = np.exp(a - logsumexp(a));
            trial_alphas =  trial_alphas*self.precision
            trial_alphas = trial_alphas + 0.001

            # Raise exception if Dirichlet parameters are invalid.
            if np.min(trial_alphas) <= 0:
                print(self.eta)
                print(self.eta_k)
                print(self.beta_value_gaze)
                print(self.beta_center_dim)
                print(self.beta_center_feat)
                print(self.precision)
                print(trial_alphas)
                raise Exception('Dirichlet parameters should be greater than 0.')

            ## Compute likelihood.
            trial_lik = dirichlet.pdf(trial_data, trial_alphas)

            ## Add likelihood (log Dirichlet probability units).
            log_lik = log_lik + np.log(trial_lik)

            ## Grab choice. 
            choice = choices[t].astype(int)

            ## Observe outcome. 
            outcome = outcomes[t].astype(int)

            ## Update chosen weights.
            pe = outcome-np.sum(W[choice-1])
            W[choice-1] = W[choice-1] + self.eta * pe

            ## Decay unchosen weights.
            all_feats = np.arange(world.n_feats)+1
            unchosen_feats = all_feats[~np.isin(all_feats, choice)]
            W[unchosen_feats-1] = (1-self.eta_k) * W[unchosen_feats-1]

        return w_all, log_lik

    def joint_likelihood(self, world, extracted_data):

        """ Returns the joint log likelihood of a sequence of choice and gaze measurements. 
            
            Parameters
            ----------
            world: instance of World.

            extracted_data: dictionary of extracted variables. 

            Returns
            -------
            log_lik: float
                Joint likelihood of choice and gaze data.

        """

        _, choice_log_lik = self.choice_likelihood(world, extracted_data)
        _, gaze_log_lik = self.attention_likelihood(world, extracted_data)
        log_lik = choice_log_lik + gaze_log_lik

        return log_lik

######################
## Training functions.
######################

def train_frl_choice(training_params, behav_training_data, et_training_data, verbose=False):
    
    """ Trains model on choice data. 
    """

    ## Set world properties. 
    world = World(3, 3, 0, 0.75, 0.25, 1)

    ## Initialize likelihood.
    Lik = 0
    
    ## Get indices of training games.
    training_games_idxs = behav_training_data.Game.unique()

    ## Get number of training games.
    n_training_games = len(behav_training_data.Game.unique())

    ## Set parameters.
    # Default values set to 0.
    params = {'learning_rate': training_params[0],
              'decay_rate': training_params[1],
              'beta_value_choice': training_params[2],
              'beta_value_gaze': 0,
              'beta_center_dim': 0,
              'beta_center_feat': 0,
              'w_init': 0,
              'decay_target': 0,
              'precision': 0}

    ## Instantiate agent.
    frl = Agent(world, params)
    
    ## Loop over training games.
    for g in np.arange(n_training_games-1):
        
        ## Subselect game trials and format data.
        trials = behav_training_data.loc[behav_training_data['Game'] == training_games_idxs[g]]['Trial'].values   
        extracted_data = extract_vars(behav_training_data, et_training_data, trials)

        ## Run model to obtain likelihood.
        W, lik = frl.choice_likelihood(world, extracted_data)
        
        Lik = Lik + lik
    
    if verbose: 
        print("total training set log likelihood:", Lik)
    
    return -Lik

# End Agent class.

def train_frl_attention_no_center_bias(training_params, behav_training_data, et_training_data):
    
    """ Trains model on gaze data with no center bias parameter. 
    """

    ## Set world properties. 
    world = World(3, 3, 0, 0.75, 0.25, 1)

    ## Initialize likelihood.
    Lik = 0
    
    ## Get indices of training games.
    training_games_idxs = behav_training_data.Game.unique()

    ## Get number of training games.
    n_training_games = len(behav_training_data.Game.unique())

    ## Set parameters.
    # Default values set to 0.
    params = {'learning_rate': training_params[0],
              'decay_rate': training_params[1],
              'beta_value_choice': 0,
              'beta_value_gaze': training_params[2],
              'beta_center_dim': 0,
              'beta_center_feat': 0,
              'w_init': 0,
              'decay_target': 0,
              'precision': training_params[3]}

    ## Instantiate agent.
    frl = Agent(world, params)
    
    ## Loop over training games.
    for g in np.arange(n_training_games-1):

        ## Subselect game trials.
        trials = behav_training_data.loc[behav_training_data['Game'] == training_games_idxs[g]]['Trial'].values   
        extracted_data = extract_vars(behav_training_data, et_training_data, trials)
        
        ## Run model to obtain likelihood.
        W, lik = frl.attention_likelihood(world, extracted_data, feature_level=True)
        
        Lik = Lik + lik
    
    print("total training set log likelihood:", Lik)
    
    return -Lik

def train_frl_attention_center_bias(training_params, behav_training_data, et_training_data):
    
    """ Trains model on gaze data with center bias parameter. 
    """

    ## Set world properties. 
    world = World(3, 3, 0, 0.75, 0.25, 1)

    ## Initialize likelihood.
    Lik = 0
    
    ## Get indices of training games.
    training_games_idxs = behav_training_data.Game.unique()

    ## Get number of training games.
    n_training_games = len(behav_training_data.Game.unique())

    ## Set parameters.
    # Default values set to 0.
    params = {'learning_rate': training_params[0],
              'decay_rate': training_params[1],
              'beta_value_choice': 0,
              'beta_value_gaze': training_params[2],
              'beta_center_dim': training_params[3],
              'beta_center_feat': training_params[4],
              'w_init': 0,
              'decay_target': 0,
              'precision': training_params[5]}

    ## Instantiate agent.
    frl = Agent(world, params)
    
    ## Loop over training games.
    for g in np.arange(n_training_games-1):

        ## Subselect game trials and format data.
        trials = behav_training_data.loc[behav_training_data['Game'] == training_games_idxs[g]]['Trial'].values   
        extracted_data = extract_vars(behav_training_data, et_training_data, trials)

        ## Run model to obtain likelihood.
        W, lik = frl.attention_likelihood(world, extracted_data)
        
        Lik = Lik + lik
    
    print("total training set log likelihood:", Lik)
    
    return -Lik

def train_frl_joint(training_params, behav_training_data, et_training_data):
    
    """ Trains model on choice and gaze data. Currently no center bias. 
    """

    ## Set world properties. 
    world = World(3, 3, 0, 0.75, 0.25, 1)

    ## Initialize likelihood.
    Lik = 0
    
    ## Get indices of training games.
    training_games_idxs = behav_training_data.Game.unique()

    ## Get number of training games.
    n_training_games = len(behav_training_data.Game.unique())

    ## Set parameters.
    # Default values set to 0.
    params = {'learning_rate': training_params[0],
              'decay_rate': training_params[1],
              'beta_value_choice': training_params[2],
              'beta_value_gaze': training_params[3],
              'beta_center_dim': 0,
              'beta_center_feat': 0,
              'w_init': 0,
              'decay_target': 0,
              'precision': training_params[4]}

    ## Instantiate agent.
    frl = Agent(world, params)
    
    ## Loop over training games.
    for g in np.arange(n_training_games-1):
        
        ## Subselect game trials and format data.
        trials = behav_training_data.loc[behav_training_data['Game'] == training_games_idxs[g]]['Trial'].values   
        extracted_data = extract_vars(behav_training_data, et_training_data, trials)
        
        ## Run model to obtain likelihood.
        lik = frl.joint_likelihood(world, extracted_data)
        
        Lik = Lik + lik
    
    print("total training set log likelihood:", Lik)
    
    return -Lik

def train_frl_joint_center_bias(training_params, behav_training_data, et_training_data):
    
    """ Trains model on choice and gaze data. Currently no center bias. 
    """

    ## Set world properties. 
    world = World(3, 3, 0, 0.75, 0.25, 1)

    ## Initialize likelihood.
    Lik = 0
    
    ## Get indices of training games.
    training_games_idxs = behav_training_data.Game.unique()

    ## Get number of training games.
    n_training_games = len(behav_training_data.Game.unique())

    ## Set parameters.
    # Default values set to 0.
    params = {'learning_rate': training_params[0],
              'decay_rate': training_params[1],
              'beta_value_choice': training_params[2],
              'beta_value_gaze': training_params[3],
              'beta_center_dim': training_params[4],
              'beta_center_feat': training_params[5],
              'w_init': 0,
              'decay_target': 0,
              'precision': training_params[6]}

    ## Instantiate agent.
    frl = Agent(world, params)
    
    ## Loop over training games.
    for g in np.arange(n_training_games-1):
        
        ## Subselect game trials and format data.
        trials = behav_training_data.loc[behav_training_data['Game'] == training_games_idxs[g]]['Trial'].values   
        extracted_data = extract_vars(behav_training_data, et_training_data, trials)
        
        ## Run model to obtain likelihood.
        lik = frl.joint_likelihood(world, extracted_data)
        
        Lik = Lik + lik
    
    print("total training set log likelihood:", Lik)
    
    return -Lik
