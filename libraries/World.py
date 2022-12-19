# Library of functions for defining a multidimensional environment with 
# discrete features. Probability of reward is highest for a target feature 
# that changes with probability h.

import numpy as np

class World(object):
	"""Container for world that the agent is learning in.

	Parameters
	----------
	n_dims : int
	    Number of dimensions along which world varies.
	n_feats_per_dim : int
	    Number of discrete features per dimension.
	n_feats : int
	    Number of total features
	obs_space: array, n_dims by n_feats per dim, int
	    Array corresponding to full observation space.
	target: int
		Target feature.
	outcome: float
		Outcome value.
	h: float
		Rate of change for target.
	p_high: float
		Rate of reward given target.
	p_low: float
		Rate of reward not given target.
	"""

	def __init__(self, n_dims, n_feats_per_dim, h, p_high, p_low, outcome, target=None):

        ## Define observation space.
		self.n_dims = n_dims
		self.n_feats_per_dim = n_feats_per_dim
		self.n_feats = n_dims * n_feats_per_dim
		# Make array corresponding to full observation space.
		os = np.zeros([n_dims, n_feats_per_dim])
		for d in np.arange(0, n_dims)+1:
			if d == 1:
				os[d-1,] = np.arange(0, n_feats_per_dim)+1
			else:
				os[d-1,] = np.arange(1, n_feats_per_dim+1) + (d-1)*n_feats_per_dim
		self.obs_space = os

        ## Define state model.
        # If not provided, randomly select relevant feature 
		if target is None:
			self.target = np.random.randint(self.n_feats)+1
		else:
			self.target = target
        # Probability with which target feature changes
		self.h = h

        ## Define reward function.
        # Probability of observing outcome given target feature is present
		self.p_high = p_high
        # Probability of observing outcome given target feature is not present
		self.p_low = p_low
        # Magnitude of binary outcome
		self.outcome = outcome

	def plot_properties(self):

		"""Plot reward and stay probability distributions in feature space.
		These will depend on the properties of the world (i.e. the reward
		function and change model). 

	    Returns
	    -------
	    fig, ax : plt.figure
	        Figure and axis of plot.
	      
	    Notes
	    -----
	    Requires matplotlib.
	    """

		import matplotlib.pyplot as plt
		import matplotlib.cm as cm
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		ratio = float(self.n_dims) / float(self.n_feats_per_dim)

		## Prepare probability distributions for visualization
		# Reward probability
		R = (self.obs_space == self.target).astype(float) 
		R[R == 1] = self.p_high
		R[R == 0] = self.p_low

		# Stay probability
		S = (self.obs_space == self.target).astype(float) 
		S[S == 1] = 1 - self.h
		S[S == 0] = self.h / (self.n_feats-1)

		## Plot probability distributions.
		# Prepare all axes 
		fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize=(8, 4));
		div1 = make_axes_locatable(ax[0])
		div2 = make_axes_locatable(ax[1])
		cax1 = div1.append_axes("right", size="5%", pad=0.05)
		cax2 = div2.append_axes("right", size="5%", pad=0.05)
		# Probability of reward in feature space.
		im1 = ax[0].imshow(R.T, cmap=cm.gray, aspect=ratio, clim = (0, 1))
		ax[0].set_title('Reward probability',fontsize = 20);
		ax[0].set_xticks([]);
		ax[0].set_yticks([]);
		ax[0].set_xlabel('Dimension',fontsize = 15)
		ax[0].set_ylabel('Feature',fontsize = 15)
		plt.colorbar(im1, cax=cax1)
		# Probability of switching in feature space.
		im2 = ax[1].imshow(S.T, cmap=cm.gray, aspect=ratio, clim = (0, 1))
		ax[1].set_title('Switch probability',fontsize = 20);
		ax[1].set_xticks([]);
		ax[1].set_yticks([]);
		ax[1].set_xlabel('Dimension',fontsize = 15)
		ax[1].set_ylabel('Feature',fontsize = 15)
		plt.colorbar(im2, cax=cax2)

		return fig, ax

	def make_stimuli(self):

		"""Returns one instance of all possible stimuli given the feature space, coded in 
		two ways.

		Returns
	    -------
	    stimuli1 : array, int, shape(n_dims, n_feats_per_dim)
	        Rows are dimensions and columns are features. Non-expanded coding.

	    stimuli2 : array, int, shape(n_dims, n_feats_per_dim)
	    	Same as above, expanded coding (features are labeled 1-n_feats).

	    single_stim : array, int, shape(1, n_feats_per_dim)
	   		Randomly selected single stimulus.

		"""

		stimuli1 = np.random.permutation(np.arange(self.n_feats_per_dim)+1)
		stimuli2 = stimuli1

		## Loop through dimensions.
		for d in np.arange(self.n_dims-1)+1:

			## Permute features.
			new = np.random.permutation(np.arange(self.n_feats_per_dim)+1)

			## Stack on top previous features.
			stimuli1 = np.vstack((stimuli1,new))
			stimuli2 = np.vstack((stimuli2, new + self.n_feats_per_dim*(d)))

		## Draw one stimulus at random.
		a_rand = np.random.randint(1,self.n_feats_per_dim+1)
		single_stim = stimuli2[:,a_rand-1]	

		return stimuli1, stimuli2, single_stim

	def make_observations(self, n_trials):
		""" Convenience function for generating random observation sequences of a 
			given length. Each observation consists of a [stimulus, outcome] pair. 
		"""

		stimuli = np.empty((0,self.n_feats_per_dim))
		outcomes = np.empty(n_trials)

		## Loop through trials. 
		for t in np.arange(n_trials):
			
			## Generate a random stimulus. 
			d, e, stimulus = self.make_stimuli()
			stimuli = np.vstack((stimuli, stimulus))

			## Compute outcome. 
			outcomes[t] = self.generate_outcome(stimulus)

		return stimuli, outcomes

	def plot_observations(self, stimuli, outcomes, target=None):
		""" Plots sequence of observations (choices and outcomes).
			Can be used with both simulated and real data.
		"""

		import matplotlib.pyplot as plt
		import seaborn as sns

		n_trials, d = stimuli.shape

		## If target is not provided, get default from world.
		if target is None: target = self.target

		fig, ax = plt.subplots(1, 1, figsize=(30,6));
		win_col = '#4dac26'
		loss_col = '#bababa'
		sz_targ = 16
		sz_nontarg = 16
		# plt.axhline(y=target, color='#f0f0f0', linestyle='-',linewidth='18')
		for t in np.arange(n_trials):
		    if outcomes[t] == 1:
		        for d in np.arange(self.n_dims):
		            if stimuli[t,d] == target:
		                plt.plot(t+1, stimuli[t,d],'s', color=win_col, markersize=sz_targ, markeredgecolor='#252525', markeredgewidth=2)
		            else:
		                plt.plot(t+1, stimuli[t,d],'s', color=win_col, markersize=sz_nontarg)    
		    else:
		        for d in np.arange(self.n_dims):
		            if stimuli[t,d] == target:
		                plt.plot(t+1, stimuli[t,d],'s', color=loss_col, markersize=sz_targ, markeredgecolor='#252525', markeredgewidth=2)      
		            else:
		                plt.plot(t+1, stimuli[t,d],'s', color=loss_col, markersize=sz_nontarg) 

		ax.set_xlim((0, n_trials+1));
		ax.set_ylim((0, self.n_feats+2));
		ax.set_xticks(np.arange(n_trials)+1);
		ax.set_yticks(np.arange(self.n_feats)+1);
		xl = ax.set_xlabel('Trial',fontsize = 30);
		yl = ax.set_ylabel('Feature',fontsize = 30);                
		ax.tick_params(labelsize=30)
		plt.ylim([0,self.n_feats+1])
		# sns.despine()

		return fig, ax

	def generate_outcome(self, stimulus, target=None):

		"""Generates binary outcome given stimulus. Requires "stimuli2" coding from
		make_stimuli

		Parameters
	    -------
	    stimulus : array, int, shape(n_dims, 1)
	    	Expanded coding.

	    target: int
	    	If target feature not provided, uses default from self.

		Returns
	    -------
	    outcome : int, 0 or 1
	        
		"""

		## If target is not provided, get default from world.
		if target is None: target = self.target
		
		## Compute outcome.
		if target in stimulus:
			outcome = int((np.random.rand() < self.p_high))
		else:
			outcome = int((np.random.rand() < self.p_low))

		return outcome





		

