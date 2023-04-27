import numpy as np
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import support
from decimal import Decimal

cache = {}


class _MetaLogisticMonoFit(stats.rv_continuous):
	"""
	This class should generally only be called inside its user-facing subclass MetaLogistic.
	It attempts to fit the data using only one fit method and one number of terms. The other class,
	MetaLogistic, handles the logic for attempting multiple fit methods.
	Here, we subclass scipy.stats.rv_continuous so we can make use of all the nice SciPy methods. We redefine the private methods
	_cdf, _pdf, and _ppf, and SciPy will make calls to these whenever needed.
	"""

	def __init__(
			self,
			cdf_ps=None,
			cdf_xs=None,
			term=None,
			fit_method=None,
			lbound=None,
			ubound=None,
			a_vector=None,
			feasibility_method='SmallMReciprocal',
			super_class_call_only=False):
		"""
		This class should only be called inside its user-facing subclass MetaLogistic.
		"""
		super().__init__()
		if super_class_call_only:
			return



		if lbound == -np.inf:
			print("Infinite lower bound was ignored")
			lbound = None

		if ubound == np.inf:
			print("Infinite upper bound was ignored")
			ubound = None

		if lbound is None and ubound is None:
			self.boundedness = False
		if lbound is None and ubound is not None:
			self.boundedness = 'upper'
			self.a, self.b = -np.inf, ubound
		if lbound is not None and ubound is None:
			self.boundedness = 'lower'
			self.a, self.b = lbound, np.inf
		if lbound is not None and ubound is not None:
			self.boundedness = 'bounded'
			self.a, self.b = lbound, ubound
		self.lbound = lbound
		self.ubound = ubound

		self.fit_method = fit_method
		self.numeric_ls_solver_used = None
		self.feasibility_method = feasibility_method

		self.cdf_ps = cdf_ps
		self.cdf_xs = cdf_xs
		if cdf_xs is not None and cdf_ps is not None:
			self.cdf_len = len(cdf_ps)
			self.cdf_ps = np.asarray(self.cdf_ps)
			self.cdf_xs = np.asarray(self.cdf_xs)

		# Special case where a MetaLogistic object is created by supplying the a-vector directly.
		if a_vector is not None:
			self.a_vector = a_vector
			if term is None:
				self.term = len(a_vector)
			else:
				self.term = term
			return

		if term is None:
			self.term = self.cdf_len
		else:
			self.term = term

		self.construct_z_vec()
		self.construct_y_matrix()

		if self.fit_method == 'Linear least squares':
			self.fit_linear_least_squares()
		elif self.fit_method == 'numeric':
			self.fit_linear_least_squares()  # This is done to generate an initial guess for the numeric method
			self.fit_numeric_least_squares(feasibility_method=feasibility_method)
		else:
			raise ValueError('Must supply a fit method to _MetaLogisticMonoFit (when called without an a-vector)')

	def is_feasible(self):
		if self.feasibility_method == 'QuantileMinimumIncrement':
			s = self.quantile_minimum_increment()
			if s < 0:
				self.valid_distribution_violation = s
				self.valid_distribution = False
			else:
				self.valid_distribution_violation = 0
				self.valid_distribution = True

		if self.feasibility_method == 'QuantileSumNegativeIncrements':
			s = self.infeasibility_score_quantile_sum_negative_increments()
			self.valid_distribution_violation = s
			self.valid_distribution = s == 0

		if self.feasibility_method == 'SmallMReciprocal':
			s = self.infeasibility_score_m_reciprocal()
			self.valid_distribution_violation = s
			self.valid_distribution = s == 0

		return self.valid_distribution

	def fit_linear_least_squares(self):
		"""
		Constructs the a-vector by linear least squares, as defined in Keelin 2016, Equation 7 (unbounded case), Equation 12 (semi-bounded and bounded cases).
		"""
		left = np.linalg.inv(np.dot(self.Y_matrix.T, self.Y_matrix))
		right = np.dot(self.Y_matrix.T, self.z_vec)

		self.a_vector = np.dot(left, right)

	def fit_numeric_least_squares(self, feasibility_method, avoid_extreme_steepness=True):
		"""
		Constructs the a-vector by attempting to approximate, using numerical methods, the feasible a-vector that minimizes least squares on the CDF.
		`feasibility_method` is the method by which we check whether an a-vector is feasible. It often has an important impact on the correctness and speed
		of the numerical minimization results.
		"""
		bounds_kwargs = {}  # bounds on the metalog specified by user
		if self.lbound is not None:
			bounds_kwargs['lbound'] = self.lbound
		if self.ubound is not None:
			bounds_kwargs['ubound'] = self.ubound

		def loss_function(a_candidate):
			# Setting a_vector in this MetaLogistic call overrides the cdf_ps and cdf_xs arguments, which are only used
			# for meanSquareError().
			return _MetaLogisticMonoFit(self.cdf_ps, self.cdf_xs, **bounds_kwargs, a_vector=a_candidate).mean_square_error()

		# Choose the method of determining feasibility.
		def feasibility_via_cdf_sum_negative(a_candidate):
			return _MetaLogisticMonoFit(a_vector=a_candidate, **bounds_kwargs).infeasibility_score_quantile_sum_negative_increments()

		def feasibility_via_quantile_minimum_increment(a_candidate):
			return _MetaLogisticMonoFit(a_vector=a_candidate, **bounds_kwargs).quantile_minimum_increment()

		def feasibility_via_small_m_reciprocal(a_candidate):
			return _MetaLogisticMonoFit(a_vector=a_candidate, **bounds_kwargs).infeasibility_score_m_reciprocal()

		if self.cdf_len == 3:
			k0 = 1.66711

			def a_ratio(a_candidate):
				a2 = a_candidate[1]
				a3 = a_candidate[2]
				return a3/a2

			def feasibility_bool(a_candidate):
				return -k0 < a_ratio(a_candidate) < k0 and a_candidate[1] > 0

			# As laid out in Keelin 2016, in the proof of Proposition 2 (page 270, left column)
			feasibility_constraint = optimize.NonlinearConstraint(a_ratio,-k0,k0)

			# The generalization to any three-term metalog implies the restriction that a2>0
			bounds_a = [(None,None),(0, np.inf)] + [(None,None)]*(self.term-2)

		else:
			bounds_a = None  # Bounds on a-vector
			if feasibility_method == 'SmallMReciprocal':
				def feasibility_bool(a_candidate):
					return feasibility_via_small_m_reciprocal(a_candidate) == 0

				feasibility_constraint = optimize.NonlinearConstraint(feasibility_via_small_m_reciprocal, 0, 0)

			if feasibility_method == 'QuantileSumNegativeIncrements':
				def feasibility_bool(a_candidate):
					return feasibility_via_cdf_sum_negative(a_candidate) == 0

				feasibility_constraint = optimize.NonlinearConstraint(feasibility_via_cdf_sum_negative, 0, 0)

			if feasibility_method == 'QuantileMinimumIncrement':
				def feasibility_bool(a_candidate):
					return feasibility_via_quantile_minimum_increment(a_candidate) >= 0

				feasibility_constraint = optimize.NonlinearConstraint(feasibility_via_quantile_minimum_increment, 0, np.inf)

		shifted = self.find_shifted_value((tuple(self.cdf_ps), tuple(self.cdf_xs), self.lbound, self.ubound), cache)
		if shifted:
			optimize_result, shift_distance = shifted
			self.a_vector = np.append(optimize_result.x[0] + shift_distance, optimize_result.x[1:])
			self.numeric_leastSQ_OptimizeResult = None
			if avoid_extreme_steepness:
				self.avoid_extreme_steepness(feasibility_bool)
			return

		a0 = self.a_vector

		optimize_results = []

		# First, try the default solver, which is often fast and accurate
		options = {}
		optimize_results_default = optimize.minimize(
											loss_function,
											a0,
											constraints=feasibility_constraint,
											bounds=bounds_a,
											options=options)
		if feasibility_bool(optimize_results_default.x):
			optimize_results_default.optimization_method_name = 'SLSQP'
			optimize_results.append(optimize_results_default)

		# If the mean square error is too large or distribution invalid, try the trust-constr solver (todo potentially move this into MetaLogistic class)
		if optimize_results_default.fun > 0.01 or not feasibility_bool(optimize_results_default.x):
			options = {
				'xtol': 1e-6,  # this improves speed considerably vs the default of 1e-8
				'maxiter': 300  # give up if taking too long
				}
			optimize_results_alternate = optimize.minimize(
															loss_function,
															a0,
															constraints=feasibility_constraint,
															bounds=bounds_a,
															method='trust-constr',
															options=options)
			if feasibility_bool(optimize_results_alternate.x):
				optimize_results_alternate.optimization_method_name = 'trust-constr'
				optimize_results.append(optimize_results_alternate)


		# Choose the best optimize result
		if not optimize_results:
			# We failed to find a solution
			return
		optimize_result_selected = sorted(optimize_results, key=lambda r: r.fun)[0]
		cache[(tuple(self.cdf_ps), tuple(self.cdf_xs), self.lbound, self.ubound)] = optimize_result_selected
		self.numeric_ls_solver_used = optimize_result_selected.optimization_method_name

		self.a_vector = optimize_result_selected.x

		if avoid_extreme_steepness:
			self.avoid_extreme_steepness(feasibility_bool)
		self.numeric_leastSQ_OptimizeResult = optimize_result_selected

	def avoid_extreme_steepness(self,feasibility_bool_callable):
		"""
		Since we are using numerical approximations to determine feasibility,
		the feasible distribution that most closely fits the data may actually be just barely infeasible.
		A hint that this is the case is that an extremely large spike in PDF will result,
		with densities in the tens of thousands! (If we were able to evaluate the PDF
		everywhere, we would see that it is actually negative in a tiny region).
		We can avoid secretly-infeasible a-vectors by biasing the *last* a-parameter very slightly (by 1%) towards zero when
		PDF steepness is extreme, which will usually guarantee that the distribution is feasible,
		at a very small cost to goodness of fit.
		"""
		steepness = self.pdf_max()
		if steepness > 10:
			candidate = np.append(self.a_vector[0:-1], self.a_vector[-1] * 99 / 100)
			if feasibility_bool_callable(candidate):
				self.a_vector = candidate

	def pdf_max(self):
		"""
		Find a very rough approximation of the max of the PDF. Used for penalizing extremely steep distributions.
		"""
		check_ps_from = 0.001
		number_to_check = 100
		ps_to_check = np.linspace(check_ps_from, 1 - check_ps_from, number_to_check)
		return max(self.density_m(ps_to_check))

	@staticmethod
	def find_shifted_value(input_tuple, cache):
		if not cache:
			return False
		for cache_tuple, cache_value in cache.items():
			shifted = _MetaLogisticMonoFit.is_same_shifted(support.tuple_to_dict(cache_tuple), support.tuple_to_dict(input_tuple))
			if shifted is not False:
				return cache_value, shifted
		return False

	@staticmethod
	def is_same_shifted(dict1, dict2):
		bounds = [dict1['lbound'], dict2['lbound'], dict1['ubound'], dict2['ubound']]
		if any([i is not None for i in bounds]):
			return False

		if not dict1['cdf_ps'] == dict2['cdf_ps']:
			return False

		dict1_x_sorted = sorted(dict1['cdf_xs'])
		dict2_x_sorted = sorted(dict2['cdf_xs'])

		diffdelta = np.abs(np.diff(dict1_x_sorted) - np.diff(dict2_x_sorted))
		diffdelta_relative = diffdelta / dict1_x_sorted[1:]
		if np.all(diffdelta_relative < .005):  # I believe this is necessary because of the imprecision of dragging in d3.js
			return dict2_x_sorted[0] - dict1_x_sorted[0]
		else:
			return False

	def mean_square_error(self):
		ps_on_fitted_cdf = self.cdf(self.cdf_xs)
		sum_sq_error = np.sum((self.cdf_ps - ps_on_fitted_cdf) ** 2)
		return sum_sq_error / self.cdf_len

	def infeasibility_score_quantile_sum_negative_increments(self):
		check_ps_from = 0.001
		number_to_check = 200  # This parameter is very important to both performance and correctness.
		ps_to_check = np.linspace(check_ps_from, 1 - check_ps_from, number_to_check)
		xs_to_check = self.quantile(ps_to_check)
		prev = -np.inf
		infeasibility_score = 0
		for item in xs_to_check:
			diff = item - prev
			if diff < 0:
				# Logarithm of the difference, to keep this scale-free
				infeasibility_score += np.log(1 - diff)
			prev = item
		return infeasibility_score

	def infeasibility_score_m_reciprocal(self):
		"""
		Checks whether the a-vector is feasible using the reciprocal of the small-m PDF, as defined in Keelin 2016, Equation 5 (see also equation 10).
		This check is performed for an array of probabilities; all reciprocals-of-densities that are negative (corresponding to an invalid distribution, since a density
		must be non-negative) are summed together into a score that attempts to quantify to what degree the a-vector is infeasible. This captures how much of the
		PDF is negative, and how far below 0 it is.
		As Keelin notes, Equations 5 and 10 are equivalent, i.e. reciprocals of densities will be positive whenever densities are positive, and so this method would return
		0 for all and only the same inputs, if we checked densities instead of reciprocals (i.e. if we omitted `densities_reciprocal = 1/densities_to_check`). However, using the reciprocal
		seems to be more amenable to finding feasible a-vectors with scipy.optimize.minimize; this might be because the reciprocal (Equation 5) is linear in the elements of the
		a-vector.
		"""
		check_ps_from = 0.001
		number_to_check = 100  # Some light empirical testing suggests 100 may be a good value here.
		ps_to_check = np.linspace(check_ps_from, 1 - check_ps_from, number_to_check)

		densities_to_check = self.density_m(ps_to_check)
		densities_reciprocal = 1 / densities_to_check
		infeasibility_score = np.abs(np.sum(densities_reciprocal[densities_reciprocal < 0]))

		return infeasibility_score

	def quantile_slope_numeric(self, p):
		"""
		Gets the slope of the quantile function by simple finite-difference approximation.
		"""
		epsilon = 1e-5
		if not np.isfinite(self.quantile(p + epsilon)):
			epsilon = -epsilon
		cdf_slope = optimize.approx_fprime(p, self.quantile, epsilon)
		return cdf_slope

	def quantile_minimum_increment(self):
		"""
		Nice idea but in practise the worst feasibility method, I might remove it.
		"""
		# Get a good initial guess
		check_ps_from = 0.001
		number_to_check = 100
		ps_to_check = np.linspace(check_ps_from, 1 - check_ps_from, number_to_check)
		xs = self.quantile(ps_to_check)
		xs_diff = np.diff(xs)
		i = np.argmin(xs_diff)
		p0 = ps_to_check[i]

		# Do the minimization
		r = optimize.minimize(self.quantile_slope_numeric, x0=p0, bounds=[(0, 1)])
		return r.fun

	def construct_z_vec(self):
		"""
		Constructs the z-vector, as defined in Keelin 2016, Section 3.3 (unbounded case, where it is called the `x`-vector),
		Section 4.1 (semi-bounded case), and Section 4.3 (bounded case).
		This vector is a transformation of cdf_xs to account for bounded or semi-bounded distributions.
		When the distribution is unbounded, the z-vector is simply equal to cdf_xs.
		"""
		if not self.boundedness:
			self.z_vec = self.cdf_xs
		if self.boundedness == 'lower':
			self.z_vec = np.log(self.cdf_xs - self.lbound)
		if self.boundedness == 'upper':
			self.z_vec = -np.log(self.ubound - self.cdf_xs)
		if self.boundedness == 'bounded':
			self.z_vec = np.log((self.cdf_xs - self.lbound) / (self.ubound - self.cdf_xs))

	def construct_y_matrix(self):
		"""
		Constructs the Y-matrix, as defined in Keelin 2016, Equation 8.
		"""

		# The series of Y_n matrices. Although we only use the last matrix in the series, the entire series is necessary to construct it
		Y_ns = {}
		ones = np.ones(self.cdf_len).reshape(self.cdf_len, 1)
		column_2 = np.log(self.cdf_ps / (1 - self.cdf_ps)).reshape(self.cdf_len, 1)
		column_4 = (self.cdf_ps - 0.5).reshape(self.cdf_len, 1)
		Y_ns[2] = np.hstack([ones, column_2])
		Y_ns[3] = np.hstack([Y_ns[2], column_4 * column_2])
		Y_ns[4] = np.hstack([Y_ns[3], column_4])

		if self.term > 4:
			for n in range(5, self.term + 1):
				if n % 2 != 0:
					new_column = column_4 ** ((n - 1) / 2)
					Y_ns[n] = np.hstack([Y_ns[n - 1], new_column])

				if n % 2 == 0:
					new_column = (column_4 ** (n / 2 - 1)) * column_2
					Y_ns[n] = np.hstack([Y_ns[n - 1], new_column])

		self.Y_matrix = Y_ns[self.term]

	def quantile(self, probability, force_unbounded=False):
		"""
		The metalog inverse CDF, or quantile function, as defined in Keelin 2016, Equation 6 (unbounded case), Equation 11 (semi-bounded case),
		and Equation 14 (bounded case).
		`probability` must be a scalar.
		"""

		# if not 0 <= probability <= 1:
		# 	raise ValueError("Probability in call to quantile() must be between 0 and 1")

		if support.is_list_like(probability):
			return np.asarray([self.quantile(i) for i in probability])

		if probability <= 0:
			if (self.boundedness == 'lower' or self.boundedness == 'bounded') and not force_unbounded:
				return self.lbound
			else:
				return -np.inf

		if probability >= 1:
			if (self.boundedness == 'upper' or self.boundedness == 'bounded') and not force_unbounded:
				return self.ubound
			else:
				return np.inf

		# Note that `self.a_vector` is 0-indexed, while in Keelin 2016 the a-vector is 1-indexed.
		# To make this method easier to read if following along with the paper, I subtract 1 when
		# subscripting the self.a_vector

		# The series of quantile functions. Although we only return the last result in the series, the entire series is necessary to construct it
		ln_p_term = np.log(probability / (1 - probability))
		p05_term = probability - 0.5
		quantile_functions = {2: self.a_vector[1 - 1] + self.a_vector[2 - 1] * ln_p_term}

		if self.term > 2:
			quantile_functions[3] = quantile_functions[2] + self.a_vector[3 - 1] * p05_term * ln_p_term
		if self.term > 3:
			quantile_functions[4] = quantile_functions[3] + self.a_vector[4 - 1] * p05_term

		if self.term > 4:
			for n in range(5, self.term + 1):
				if n % 2 != 0:
					quantile_functions[n] = quantile_functions[n - 1] + self.a_vector[n - 1] * p05_term ** ((n - 1) / 2)

				if n % 2 == 0:
					quantile_functions[n] = quantile_functions[n - 1] + self.a_vector[n - 1] * p05_term ** (n / 2 - 1) * ln_p_term

		quantile_function = quantile_functions[self.term]

		if not force_unbounded:
			if self.boundedness == 'lower':
				quantile_function = self.lbound + np.exp(quantile_function)  # Equation 11
			if self.boundedness == 'upper':
				quantile_function = self.ubound - np.exp(-quantile_function)
			if self.boundedness == 'bounded':
				quantile_function = (self.lbound + self.ubound * np.exp(quantile_function)) / (1 + np.exp(quantile_function))  # Equation 14

		return quantile_function

	def density_m(self, cumulative_prob, force_unbounded=False):
		"""
		This is the metalog PDF as a function of cumulative probability, as defined in Keelin 2016, Equation 9 (unbounded case),
		Equation 13 (semi-bounded case), Equation 15 (bounded case).
		Notice the unusual definition of the PDF, which is why I call this function density_m in reference to the notation in
		Keelin 2016.
		"""

		if support.is_list_like(cumulative_prob):
			return np.asarray([self.density_m(i) for i in cumulative_prob])

		if not 0 <= cumulative_prob <= 1:
			raise ValueError("Probability in call to density_m() must be between 0 and 1")
		if not self.boundedness and (cumulative_prob == 0 or cumulative_prob == 1):
			raise ValueError("Probability in call to density_m() cannot be equal to 0 and 1 for an unbounded distribution")

		# The series of density functions. Although we only return the last result in the series, the entire series is necessary to construct it
		density_functions = {}

		# Note that `self.a_vector` is 0-indexed, while in Keelin 2016 the a-vector is 1-indexed.
		# To make this method easier to read if following along with the paper, I subtract 1 when
		# subscripting the self.a_vector

		ln_p_term = np.log(cumulative_prob / (1 - cumulative_prob))
		p05_term = cumulative_prob - 0.5
		p1p_term = cumulative_prob * (1 - cumulative_prob)

		density_functions[2] = p1p_term / self.a_vector[2 - 1]
		if self.term > 2:
			density_functions[3] = 1 / (1 / density_functions[2] + self.a_vector[3 - 1] * (p05_term / p1p_term + ln_p_term))
		if self.term > 3:
			density_functions[4] = 1 / (1 / density_functions[3] + self.a_vector[4 - 1])

		if self.term > 4:
			for n in range(5, self.term + 1):
				if n % 2 != 0:
					density_functions[n] = 1 / (1 / density_functions[n - 1] + self.a_vector[n - 1] * ((n - 1) / 2) * p05_term ** ((n - 3) / 2))

				if n % 2 == 0:
					density_functions[n] = 1 / (1 / density_functions[n - 1] + self.a_vector[n - 1] * (p05_term ** (n / 2 - 1) / p1p_term +
																		 (n / 2 - 1) * p05_term ** (n / 2 - 2) * ln_p_term))

		density_function = density_functions[self.term]
		if not force_unbounded:
			if self.boundedness == 'lower':  # Equation 13
				if 0 < cumulative_prob < 1:
					density_function = density_function * np.exp(-self.quantile(cumulative_prob, force_unbounded=True))
				elif cumulative_prob == 0:
					density_function = 0
				else:
					raise ValueError("Probability in call to density_m() cannot be equal to 1 with a lower-bounded distribution.")

			if self.boundedness == 'upper':
				if 0 < cumulative_prob < 1:
					density_function = density_function * np.exp(self.quantile(cumulative_prob, force_unbounded=True))
				elif cumulative_prob == 1:
					density_function = 0
				else:
					raise ValueError("Probability in call to density_m() cannot be equal to 0 with a upper-bounded distribution.")

			if self.boundedness == 'bounded':  # Equation 15
				if 0 < cumulative_prob < 1:
					x_unbounded = np.exp(self.quantile(cumulative_prob, force_unbounded=True))
					density_function = density_function * (1 + x_unbounded) ** 2 / ((self.ubound - self.lbound) * x_unbounded)
				if cumulative_prob == 0 or cumulative_prob == 1:
					density_function = 0

		return density_function

	def get_cumulative_prob(self, x):
		"""
		The metalog is defined in terms of its inverse CDF or quantile function. In order to get probabilities for a given x-value,
		like in a traditional CDF, we invert this quantile function using a numerical equation solver.
		`x` must be a scalar
		"""
		f_to_zero = lambda probability: self.quantile(probability) - x

		# We need a smaller `xtol` than the default value, in order to ensure correctness when
		# evaluating the CDF or PDF in the extreme tails.
		# todo: consider replacing brent's method with Newton's (or other), and provide the derivative of quantile, since it should be possible to obtain an analytic expression for that
		return optimize.brentq(f_to_zero, 0, 1, xtol=1e-24, disp=True)

	def _cdf(self, x):
		"""
		This is where we override the SciPy method for the CDF.
		`x` may be a scalar or list-like.
		"""
		if support.is_list_like(x):
			return [self._cdf(i) for i in x]
		if support.is_numeric(x):
			return self.get_cumulative_prob(x)

	def _ppf(self, probability):
		"""
		This is where we override the SciPy method for the inverse CDF or quantile function (ppf stands for percent point function).
		`probability` may be a scalar or list-like.
		"""
		return self.quantile(probability)

	def _pdf(self, x):
		"""
		This is where we override the SciPy method for the PDF.
		`x` may be a scalar or list-like.
		"""
		if support.is_list_like(x):
			return [self._pdf(i) for i in x]

		if support.is_numeric(x):
			cumulative_prob = self.get_cumulative_prob(x)
			return self.density_m(cumulative_prob)

	def print_summary(self):
		print("Fit method used:", self.fit_method)
		print("Distribution is valid:", self.valid_distribution)
		print("Method for determining distribution validity:", self.feasibility_method)
		if not self.valid_distribution:
			print("Distribution validity constraint violation:", self.valid_distribution_violation)
		if not self.fit_method == 'Linear least squares':
			print("Solver for numeric fit:", self.numeric_ls_solver_used)
			print("Solver convergence:", self.numeric_leastSQ_OptimizeResult.success)
		# print("Solver convergence message:", self.numeric_leastSQ_OptimizeResult.message)
		print("Mean square error:", self.mean_square_error())
		print('a vector:', self.a_vector)

	def create_cdf_plot_data(self, p_from_to=(.001, .999), x_from_to=(None, None), n=100):
		p_from, p_to = p_from_to
		x_from, x_to = x_from_to
		if x_from is not None and x_to is not None:
			p_from = self.get_cumulative_prob(x_from)
			p_to = self.get_cumulative_prob(x_to)

		cdf_ps = self.get_linspace_for_plot(p_from, p_to, n)
		cdf_xs = self.quantile(cdf_ps)

		return {'X-values': cdf_xs, 'Probabilities': cdf_ps}

	def create_pdf_plot_data(self, p_from_to=(.001, .999), x_from_to=(None, None), n=100):
		p_from, p_to = p_from_to
		x_from, x_to = x_from_to
		if x_from is not None and x_to is not None:
			p_from = self.get_cumulative_prob(x_from)
			p_to = self.get_cumulative_prob(x_to)

		pdf_ps = self.get_linspace_for_plot(p_from, p_to, n)
		pdf_xs = self.quantile(pdf_ps)
		pdf_densities = self.density_m(pdf_ps)

		return {'X-values': pdf_xs, 'Densities': pdf_densities}

	def get_linspace_for_plot(self, p_from, p_to, n):
		"""
		Solve the following system:
		N = MIDDLE + 2*TAIL
		TAIL = (8/84)*MIDDLE*FACTOR
		"""

		factor = 8
		n_middle = round(n / (1 + 2 * factor * (8 / 84)))
		n_tail = round((n - n_middle) / 2)

		p_range = p_to - p_from
		p_08 = p_from + .08 * p_range
		p_92 = p_from + .92 * p_range

		ps = np.hstack([
			np.linspace(p_from, p_08, n_tail),
			np.linspace(p_08, p_92, n_middle),
			np.linspace(p_92, p_to, n_tail)
		])

		return ps

	def display_plot(self, p_from_to=(.001, .999), x_from_to=(None, None), n=100, hide_extreme_densities=50):
		"""
		The parameter `hide_extreme_densities` is used on the PDF plot, to set its y-axis maximum to `hide_extreme_densities` times
		the median density. This is because, when given extreme input data, the resulting metalog might have very short but extremely tall spikes in the PDF
		(where the maximum density might be in the hundreds), which would make the PDF plot unreadable if the entire spike was included.
		If both `p_from_to` and `x_from_to` are specified, `p_from_to` is overridden.
		"""

		p_from, p_to = p_from_to
		x_from, x_to = x_from_to
		if x_from is not None and x_to is not None:
			p_from = self.get_cumulative_prob(x_from)
			p_to = self.get_cumulative_prob(x_to)

		fig, (cdf_axis, pdf_axis) = plt.subplots(2)
		# fig.set_size_inches(6, 6)

		cdf_data = self.create_cdf_plot_data(p_from_to=(p_from, p_to), n=n)
		cdf_axis.plot(cdf_data['X-values'], cdf_data['Probabilities'])
		if self.cdf_xs is not None and self.cdf_ps is not None:
			cdf_axis.scatter(self.cdf_xs, self.cdf_ps, marker='+', color='red')
		cdf_axis.set_title('CDF')

		pdf_data = self.create_pdf_plot_data(p_from_to=(p_from, p_to), n=n)
		pdf_axis.set_title('PDF')

		if hide_extreme_densities:
			density50 = np.percentile(pdf_data['Densities'], 50)
			pdf_max_display = min(density50 * hide_extreme_densities, 1.05 * max(pdf_data['Densities']))
			pdf_axis.set_ylim(top=pdf_max_display)

		pdf_axis.plot(pdf_data['X-values'], pdf_data['Densities'])

		fig.show()
		return fig


class MetaLogistic(_MetaLogisticMonoFit):
	"""
	This is the only user-facing class in this package. It handles the logic that attempts to
	fit the distribution using an initial fit method and a number of terms, and falls back to
	less desirable methods in case of failure.
	It makes calls to _MetaLogisticMonoFit. That class does all the mathematical work of
	fitting the distribution, and also has methods for printing a summary and displaying plots.
	Since _MetaLogisticMonoFit is a superclass of MetaLogistic, all its methods are available
	in an instance of MetaLogistic.
	"""
	def __init__(
			self,
			cdf_ps=None,
			cdf_xs=None,
			term=None,
			fit_method=None,
			allow_fallback=True,
			lbound=None,
			ubound=None,
			a_vector=None,
			feasibility_method='SmallMReciprocal',
			validate_inputs=True):
		"""
		You must either provide CDF data or directly provide an a-vector. All other parameters are optional.
		:param cdf_ps: Probabilities of the CDF input data.
		:param cdf_xs: X-values of the CDF input data (the pre-images of the probabilities).
		:param term: Produces a `term`-term metalog. Cannot be greater than the number of CDF points provided. By default, it is equal to the number of CDF points.
		:param fit_method: Set to 'Linear least squares' to allow linear least squares only. By default, numerical methods are tried if linear least squares fails.
		:param lbound: Lower bound
		:param ubound: Upper bound
		:param a_vector: You may supply the a-vector directly, in which case the input data `cdf_ps` and `cdf_xs` are not used for fitting.
		:param feasibility_method: The method used to determine whether an a-vector corresponds to a feasible (valid) probability distribution. Its most important use is in the numerical solver, where it can have an impact on peformance and correctness. The options are: 'SmallMReciprocal' (default),'QuantileSumNegativeIncrements','QuantileMinimumIncrement'.
		"""

		super().__init__(super_class_call_only=True)

		if term is not None:
			allow_fallback = False

		if validate_inputs:
			self.validate_inputs(
				cdf_ps=cdf_ps,
				cdf_xs=cdf_xs,
				term=term,
				fit_method=fit_method,
				lbound=lbound,
				ubound=ubound,
				a_vector=a_vector,
				feasibility_method=feasibility_method
			)

		self.cdf_ps = cdf_ps
		self.cdf_xs = cdf_xs
		self.cdf_len = len(cdf_ps)

		self.lbound = lbound
		self.ubound = ubound


		user_kwargs = {
			'cdf_ps': cdf_ps,
			'cdf_xs': cdf_xs,
			'term': term,
			'lbound': lbound,
			'ubound': ubound,
			'feasibility_method': feasibility_method,
		}

		self.fit_method_requested = fit_method
		self.term_requested = term

		self.candidates_valid = []
		self.candidates_all = []

		#  Try linear least squares
		candidate = _MetaLogisticMonoFit(**user_kwargs, fit_method='Linear least squares')

		# If linear least squares works, we are done
		self.candidates_all.append(candidate)
		if candidate.is_feasible():
			self.candidates_valid.append(candidate)

		# todo implement timeouts
		# Otherwise, try other methods if allowed
		else:
			# if the user allows it, use numerical least squares
			if not fit_method == 'Linear least squares':
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=UserWarning, module='scipy.optimize')

					candidate = _MetaLogisticMonoFit(**user_kwargs, fit_method='numeric')

					self.candidates_all.append(candidate)
					if candidate.is_feasible():
						self.candidates_valid.append(candidate)

			if allow_fallback:
				if term is not None:
					term_fallback = term-1
				else:
					term_fallback = len(cdf_ps)-1
				kwargs = user_kwargs.copy()
				kwargs['term'] = term_fallback
				candidate = _MetaLogisticMonoFit(**kwargs, fit_method='Linear least squares')

				self.candidates_all.append(candidate)
				if candidate.is_feasible():
					self.candidates_valid.append(candidate)

				elif not fit_method == 'Linear least squares':
					if self.candidates_valid:
						current_winner_error = sorted(self.candidates_valid, key=lambda c: c.mean_square_error())[0].mean_square_error()
					else:
						current_winner_error = np.inf

					# Only do the N-1 numeric fit if the current winner is poor (todo make global constant)
					if current_winner_error > 0.01:
						with warnings.catch_warnings():
							warnings.filterwarnings("ignore", category=UserWarning, module='scipy.optimize')

							candidate = _MetaLogisticMonoFit(**kwargs, fit_method='numeric')

							self.candidates_all.append(candidate)
							if candidate.is_feasible():
								self.candidates_valid.append(candidate)

		# The below should be handled much more nicely
		if self.candidates_valid:
			self.valid_distribution = True
			winning_candidate = sorted(self.candidates_valid, key=lambda c: c.mean_square_error())[0]
		else:
			self.valid_distribution = False
			winning_candidate = self.candidates_all[0]

		self.__dict__.update(winning_candidate.__dict__.copy())
		self.term_used = winning_candidate.term
		self.fit_method_used = winning_candidate.fit_method


	def validate_inputs(
			self,
			cdf_ps,
			cdf_xs,
			term,
			fit_method,
			lbound,
			ubound,
			a_vector,
			feasibility_method):

		def check_Ps_Xs(array, name):
			if support.is_list_like(array):
				for item in array:
					if not support.is_numeric(item):
						raise ValueError(name + " must be an array of numbers")
			else:
				raise ValueError(name + " must be an array of numbers")

		if cdf_xs is not None:
			check_Ps_Xs(cdf_xs, 'cdf_xs')
			cdf_xs = np.asarray(cdf_xs)

		if cdf_ps is not None:
			check_Ps_Xs(cdf_ps, 'cdf_ps')
			cdf_ps = np.asarray(cdf_ps)

			if np.any(cdf_ps < 0) or np.any(cdf_ps > 1):
				raise ValueError("Probabilities must be between 0 and 1")

		if cdf_ps is not None and cdf_xs is not None:
			if len(cdf_ps) != len(cdf_xs):
				raise ValueError("cdf_ps and cdf_xs must have the same length")

			if len(cdf_ps) < 2:
				raise ValueError("Must provide at least two CDF data points")

			ps_xs_sorted = sorted(zip(cdf_ps, cdf_xs))
			prev = -np.inf
			for tuple_ in ps_xs_sorted:
				p, x = tuple_
				if x <= prev:
					print("Warning: Non-increasing CDF input data. Are you sure?")
				prev = x

			if term is not None:
				if term > len(cdf_ps):
					raise ValueError("term cannot be greater than the number of CDF data points provided")

		if term is not None:
			if term < 2:
				raise ValueError("term cannot be less than 2.")

		if a_vector is not None and term is not None:
			if term > len(a_vector):
				raise ValueError("term cannot be greater than the length of the a_vector")

		if fit_method is not None:
			if fit_method not in ['Linear least squares']:
				raise ValueError("Unknown fit method")

		if lbound is not None:
			if lbound > min(cdf_xs):
				raise ValueError("Lower bound cannot be greater than the lowest data point")

		if ubound is not None:
			if ubound < max(cdf_xs):
				raise ValueError("Upper bound cannot be less than the greatest data point")

		feasibility_methods = ['SmallMReciprocal', 'QuantileSumNegativeIncrements', 'QuantileMinimumIncrement']
		if feasibility_method not in feasibility_methods:
			raise ValueError("feasibility_method must be one of: " + str(feasibility_methods))