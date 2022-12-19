# Library of functions for handling data from participant behavior when learning
# in a multidimensional environment with discrete features. 

import numpy as np

class Data(object):
    """ Container for data and methods.

    Parameters
    ----------
    behav_data: Pandas dataframe 
        Behavioral data for one participant.

    et_data: Pandas dataframe
        Feature-level eyetracking data for one participant. 
        Summarized as relative looking times by feature over time.
    ----------
    """

    def __init__(self, behav_data, et_data):
    
        ## Define data.
        self.behav_data = behav_data
        self.et_data = et_data
       
        ## Get other variables 
        self.n_trials = max(behav_data['Trial'])
        self.n_games = max(behav_data['Game'])
        self.game_length = len(behav_data.loc[(behav_data['Game'] == 1)])

        ## Add trial-within-game variable.
        self.behav_data['Trial_2'] = self.behav_data['Trial'] - (self.behav_data['Game']-1)*self.game_length

    def split_data(self, test_game):
        """ Splits behavioral data into training data (n-1 games) and test data (1 game).
        """

        ## Behavioral data.
        behav_training_data = self.behav_data.loc[self.behav_data['Game'] != test_game]
        behav_test_data = self.behav_data.loc[self.behav_data['Game'] == test_game]
        
        ## Eye-tracking data.
        training_trials = behav_training_data['Trial'].values
        test_trials = behav_test_data['Trial'].values
        et_training_data = self.et_data.loc[self.et_data['Trial'].isin(training_trials)]
        et_test_data = self.et_data.loc[self.et_data['Trial'].isin(test_trials)]

        return behav_training_data, behav_test_data, et_training_data, et_test_data 

def extract_vars(behav_data, et_data, trials):
    """ Helper function that extracts variables from one game given trial indices. 
    """

    ## Get observations for this game (available stimuli, choices, outcomes, center dimension and feature).
    stimuli_1 = behav_data.loc[behav_data['Trial'].isin(trials)][['Stim11','Stim12','Stim13']].values
    stimuli_2 = behav_data.loc[behav_data['Trial'].isin(trials)][['Stim21','Stim22','Stim23']].values
    stimuli_3 = behav_data.loc[behav_data['Trial'].isin(trials)][['Stim31','Stim32','Stim33']].values  
    choices = behav_data.loc[behav_data['Trial'].isin(trials)][['Chosen1','Chosen2','Chosen3']].values
    outcomes = behav_data.loc[behav_data['Trial'].isin(trials)]['Outcome'].values
    center_dim = behav_data.loc[behav_data['Trial'].isin(trials)]['CenterDim'].values
    center_feat = behav_data.loc[behav_data['Trial'].isin(trials)]['CenterFeat'].values
    center = np.vstack((center_dim,center_feat)).T
    missed_trials = np.isnan(outcomes)

    ## Mark target.
    target = behav_data['Feat'].iloc[0]

    ## Mark whether game was learned. 
    point_of_learning = behav_data.loc[behav_data['Trial'].isin(trials)]['PoL'].values[0]
    if point_of_learning < 16: learned = 1
    else: learned = 0 

    ## Subselect eyetracking timecourses.
    et_game_data = et_data.loc[et_data['Trial'].isin(trials)]
    et_game_data.reset_index(inplace = True, drop = True)
    # Remove trial column.
    del et_game_data['Trial']
    
    ## Mark chosen action. 
    chose_1 = np.prod(choices == stimuli_1, axis=1)
    chose_2 = np.prod(choices == stimuli_2, axis=1)
    chose_3 = np.prod(choices == stimuli_3, axis=1)
    # actions = np.vstack((chose_1, chose_2, chose_3)).T.nonzero()[1]+1
    actions = np.ones(len(trials))*np.nan
    actions[np.vstack((chose_1, chose_2, chose_3)).T.nonzero()[0]] = np.vstack((chose_1, chose_2, chose_3)).T.nonzero()[1] + 1

    ## Remove missed trials, or trials in which we did not have gaze data.
    if np.sum(np.isnan(et_game_data[1].values)): 
        nan_idx_gaze = np.squeeze(np.argwhere(et_game_data.isnull()[1].values),axis=1)
    else: 
        nan_idx_gaze = []

    if np.sum(np.isnan(outcomes)):
        nan_idx_choices = np.argwhere(np.isnan(outcomes)).flatten()
    else:
        nan_idx_choices = []
    nan_idx = intersection_without_duplicates(list(nan_idx_choices), list(nan_idx_gaze))

    stimuli_1 = np.delete(stimuli_1, nan_idx, axis=0)
    stimuli_2 = np.delete(stimuli_2, nan_idx, axis=0)
    stimuli_3 = np.delete(stimuli_3, nan_idx, axis=0)
    choices = np.delete(choices, nan_idx, axis=0)
    outcomes = np.delete(outcomes, nan_idx, axis=0)   
    center = np.delete(center, nan_idx, axis=0)
    actions = np.delete(actions, nan_idx, axis=0)
    et_game_data = et_game_data.drop(nan_idx,axis=0) 
    
    ## Create dictionary.
    extracted_data = {
        "stimuli_1": stimuli_1,
        "stimuli_2": stimuli_2,   
        "stimuli_3": stimuli_3,
        "choices": choices,
        "actions": actions,
        "outcomes": outcomes,
        "center": center,
        "et_data": et_game_data.values,
        "learned_game": learned,
        "target": target
    }

    return extracted_data

def intersection_without_duplicates(first_list, second_list): 
    return first_list + list(set(second_list) - set(first_list))

