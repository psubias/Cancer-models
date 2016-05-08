#===============================================================#
#								#
#   Evolutionary algorithm.           				#
#								#
#   There is a population of X individuals (constant for the	#
#   same type of individuals) with different configurations.	#
#   Each individual is formed by N cells. Each one is		#
#   defined by its number of oncogenes (wild type and mutants), #
#   quantity of supressor-genes (wild type and mutants), 	#
#   replication rate and mutation rate.				#
#   There are two possible structures: spatial or non-spatial.  #
#   The genes of this individual evolve by means of the 	#
#   Moran Process.						#
#								#
#   Comments: apoptosis not included.				#
#   								#
#   Author: Paula Subias Beltran                                #
#   Date:   01-03-2016						#
#                                                               #
#===============================================================#
 
import sys
import os
import numpy as np
import copy
import math
from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd
import timeit



moves = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 0), (0, 1), (-1, -1), (-1, 0), (-1, 1)]
fitness_global = []
prob_cancer = []

class EA():
	
	""" Simulates the evolution of the cells of an individual. """
 
	def __init__(self, lifespan = 100, N = 10000, Na = 6, Ni = 5, mu = 0.05, alpha= 0.3, \
		rho = 0.9, spatial = True, output_dir = os.path.abspath(__file__)):
        	""" Initializes the simulation. """
		self.lifespan = lifespan        # The lifespan (expected) of the individual (number of cell regeneration)
	        self.evo_r = []		        # Results of the EA (r)
		self.evo_Nam = []		# Results of the EA (Nam)
		self.evo_Nim = []		# Results of the EA (Nim)
		self.evo_Nm = []		# Results of the EA (Nm) (number mutated genes)
		self.N = N			# Number of cells
		self.Na = Na			# Number of activator genes
		self.Ni = Ni			# Number of inhibitor genes
		self.tseries_Nm = []		# Results: evolution #mutated cells
		self.mu = mu			# Mutation rate
		self.alpha = alpha		# Basal replication rate
		self.rho = rho			# Replication (penalty) rate -in case of mutation
		self.spatial = spatial		# Is an spatial configuration?
		self.L = int(np.sqrt(N))	# Lenght side square (structure) 
		self.output_dir = output_dir	# Output directory
		#self.evo_fitness = []		# Fitness over time

	def compute_r(self, Nam, Nim):
		""" Update of the replication probability. """
		r = self.alpha * ((self.Na - Nam)- (self.Ni - Nim)) * 1. / (self.Na + self.Ni) + \
			self.rho * (Nam + Nim) * 1. / (self.Na + self.Ni)
		return(r)

	def mutate_individual(self, Nam, Nim):
		""" Mutation of an individual. """
		if (Nam < self.Na):
			Na_wild = int(self.Na - Nam)
			for gene in range(Na_wild):
				if (np.random.uniform() < self.mu):
					Nam = Nam + 1
		if (Nim < self.Ni):
			Ni_wild = int(self.Ni - Nim)
			for gene in range(Ni_wild):
				if (np.random.uniform() < self.mu):
					Nim = Nim + 1
		return([Nam, Nim])

	def update_individual_spatial(self, population, row, col):
		""" Update the treats of an individual. 
		    - spatial configuration.
		"""
		Nam = population[1][row][col]
		Nim = population[2][row][col]
		population[1][row][col], population[2][row][col] = self.mutate_individual(Nam, Nim)
		population[0][row][col] = self.compute_r(population[1][row][col], population[2][row][col])
		return(population)

	def update_individual_nospatial(self, population, cell):
		""" Update the treats of an individual.
		    - Non-spatial configuration.
		"""
		Nam = population[1][cell]
		Nim = population[2][cell]
		population[1][cell], population[2][cell] = self.mutate_individual(Nam, Nim)
		population[0][cell] = self.compute_r(population[1][cell], population[2][cell])
		return(population)

	def initialize_population_cellbycell(self):
		""" Creation of the initial population cell by cell. """
		if (self.spatial == True):
			initial_population = [[EA.Cell(self.Na, self.Ni, self.alpha, self.rho, self.mu) for j in range(self.L)] \
						for i in range(self.L)]
		else:
			initial_population = [EA.Cell(self.Na, self.Ni, self.alpha, self.rho, self.mu) for i in range(self.N)]
	
		return(initial_population)
	
	def initialize_population_traitbytrait(self):
		""" Creation of the initial population trait by trait. """
		if (self.spatial == True):
			initial_r = np.ones([self.L, self.L]) * self.compute_r(0, 0) * 1.
			initial_Nam = np.zeros([self.L, self.L])
			initial_Nim = np.zeros([self.L, self.L])
		else:
			initial_r = [self.compute_r(0, 0) * 1. for i in range(self.N)]
			initial_Nam = [0 for i in range(self.N)]
			initial_Nim = copy.copy(initial_Nam)

		return(initial_r, initial_Nam, initial_Nim)

	#def store_results(self, population):
		#""" Store the results of the simulation to analyze them later. """
		#if (self.spatial == True):
			#self.evo_Nm.append(copy.copy(population[1]) + copy.copy(population[2]))
			#self.tseries_Nm.append([(self.evo_Nm[-1] == i).sum() for i in range(self.Na + self.Ni + 1)])
		#else:
			#self.evo_Nm.append([x + y for x, y in zip(population[1], population[2])])
			#self.tseries_Nm.append([sum(1 for x in self.evo_Nm[-1] if x == i) for i in range(self.Na + self.Ni + 1)])
		#self.evo_fitness.append(copy.copy(self.compute_fitness()))	

	def run(self):
		""" Run the simulation. """
		#self.evo_fitness.extend([self.N, self.lifespan, self.mu, self.Na, self.Ni])
		population = self.initialize_population_traitbytrait()
		k = 0
		while (k < self.lifespan):
			#self.store_results(population)
			population = self.MoranProcess(population)
			k = k + 1

		#if (self.spatial == True):
			#self.saveAnimation()

		# Save the fitness in a global variable.
		#fitness_global.append(self.evo_fitness)
		
		# Save the probability of cancer
		if (self.spatial == True):
			aux = (copy.copy(population[1]) + copy.copy(population[2]))
			self.tseries_Nm.append([(aux == i).sum() for i in range(self.Na + self.Ni + 1)])
		else:
			aux = ([x + y for x, y in zip(population[1], population[2])])
			self.tseries_Nm.append([sum(1 for x in aux if x == i) for i in range(self.Na + self.Ni + 1)])
			
		prob_cancer.append([self.N, self.lifespan, self.mu, self.Ni, self.compute_probC(), \
		  self.compute_fitness_approach1(), self.compute_fitness_approach2()])
		
	def compute_probC(self):
	      number_cancerigenCells = sum(self.tseries_Nm[-1][1:]) # with at least 1 mutation
	      return(number_cancerigenCells * 1. / self.N)
	
	def MoranProcess(self, population):
		""" Moran Process. """
		if (self.spatial == True):
			for i in range(self.N):
				cell2rep = [np.random.randint(0, self.L), np.random.randint(0, self.L)]
				# Replication rate must be large enough to reproduce
				if (np.random.uniform() < population[0][cell2rep[0]][cell2rep[1]]):
					done = False
					while (not done):
						move = moves[np.random.randint(0, 9)]
						cell2sub_row = cell2rep[0] + move[0]
						cell2sub_col = cell2rep[1] + move[1]
						if (cell2sub_row >= 0 and cell2sub_row < self.L and \
							cell2sub_col >= 0 and cell2sub_col < self.L):
							done = True

					population[0][cell2sub_row][cell2sub_col] = population[0][cell2rep[0]][cell2rep[1]]
					population[1][cell2sub_row][cell2sub_col] = population[1][cell2rep[0]][cell2rep[1]]	
					population[2][cell2sub_row][cell2sub_col] = population[2][cell2rep[0]][cell2rep[1]]
				
					population = self.update_individual_spatial(population, cell2sub_row, cell2sub_col)
						
		else:
			for i in range(self.N):
				cell2rep = np.random.randint(0, self.N)
				# Replication rate must be large enough to reproduce
				if (np.random.uniform() < population[0][cell2rep]):
					cell2sub = np.random.randint(0, self.N)
					population[0][cell2sub] = population[0][cell2rep]
					population[1][cell2sub] = population[1][cell2rep]
					population[2][cell2sub] = population[2][cell2rep]
					
					population = self.update_individual_nospatial(population, cell2sub)
		return(population)
		
	def compute_fitness_approach1(self):
		number_cancerigenCells = sum(self.tseries_Nm[-1][1:]) # with at least 1 mutation
		fitness = ((self.N - number_cancerigenCells) * 1./ self.N) * (1. / self.Ni) 
		return fitness
		
	def compute_fitness_approach2(self):
		number_cancerigenCells = sum(self.tseries_Nm[-1][1:]) # with at least 1 mutation
		Ng = self.Na + self.Ni
		fitness = ((self.N - number_cancerigenCells) * 1./ self.N) * ((Ng - self.Ni) * 1. / Ng)
		return fitness
		
		
def frange(start, stop, step):
	i = start
	while i < stop:
		yield i
		i += step

def compute_lifespan(N):
	L = int(1e7 * (1. / (60 * 24)) * math.pow(N * 1e-12, 1./4))
	return(L)
  
if __name__ == "__main__":
	start = timeit.default_timer()
	#lifespan   = 500         # Lifetime
	#N	   = 10000	 # Number of cells
	Na         = 6           # Number of activator genes
	#mu	   = 0.05	 # Mutation rate
	alpha	   = 0.3   	 # Basal replication rate
	rho  	   = 0.6	 # Replication (penalty) rate -in case of mutation
	#spatial    = True	 # Is an spatial simulation?

	for LSpan in ["AllometricLaw", "Proportional"]:
		Slist = [False, True]
		for S in Slist:
			spatial = S

			if (spatial == True):
				output_dir = os.path.dirname(os.path.abspath(__file__)) + "/Results EvolvePopulation.py/" + \
					"spatial_t" + "_Na" + str(Na) + "_alpha" + str(alpha).replace('.','') + \
					"_rho" + str(rho).replace('.','') + "/"
			else:
				output_dir = os.path.dirname(os.path.abspath(__file__)) + "/Results EvolvePopulation.py/" + \
					"nospatial_t" + "_Na" + str(Na) + "_alpha" + str(alpha).replace('.','') + \
					  "_rho" + str(rho).replace('.','') + "/"

			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

			#size_individuals = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18, 1e20]
			size_individuals = [1e2, 1e3, 1e4, 1e5]
			mu_list = [0.001, 0.01, 0.03, 0.05]
			
			for N in size_individuals:
				N = int(N)
				if (LSpan == "AllometricLaw"):
					lifespan = compute_lifespan(N)
				else:
					lifespan = int(N*1./4)
				
				#for mu in frange(0.005, 0.400, 0.005):
				for mu in mu_list:
					for i in range(1, Na):
						Ni = i # Number of inhibitor genes
						sys.stderr.write("\nRunning simulation with %d cells: %d activator genes and %d inhibitor genes \n lifespan = %d, mu = %.3f, alpha = %.2f, rho = %.2f, spatial configuration = %r and lifespan computed by %r..." % \
							  (N, Na, Ni, lifespan, mu, alpha, rho, spatial, LSpan))
						for it in range(5):
							sys.stderr.write("\nCreating individual %d..." % it)
							sim = EA(lifespan, N, Na, Ni, mu, alpha, rho, spatial, output_dir)
							sim.run()
										
						df = pd.DataFrame(prob_cancer)
						df.to_csv(output_dir + "prob_cancer&fitness_" + LSpan + ".csv", \
						  index = False, header = False, sep = ";", decimal = ".")
						
	sys.stderr.write("[DONE]\n")

	stop = timeit.default_timer()

	print stop - start 

