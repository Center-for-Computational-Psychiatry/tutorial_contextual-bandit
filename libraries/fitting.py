# Generic class for storing and loading a fit result.
# Note that if you change the naming convention (e.g. fititng-> Fit) 
# this will break the pickled files that were made using the library
# under its previous name. 

class Fit(object):
	""" Container for fit properties and methods.
	
	Parameters
	----------
	subj_id: int
		Subject id.
	n_iter: int
		Number of iterations.
	initial_conditions: float, shape(n_iter, n_params)
		Initial_conditions of parameters.
	results: float
		Softmax choice temperature.
	elapsed_time: float
		Time it took to fit.
	----------
	
	"""

	def __init__(self, subj_id, n_iter, initial_conditions, results, elapsed_time):
		""" 
		Sets fit parameters.
		"""

		self.subj_id = subj_id
		self.n_iter = n_iter
		self.initial_conditions = initial_conditions
		self.results = results 
		self.elapsed_time = elapsed_time