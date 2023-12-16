import random 
import numpy as np
import json
import  pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, bernoulli
from itertools import combinations

SUSCEPTIBLE = 0
INCUBATED = 1
INFECTED = 2
VACCINATED = 3
REMOVED = 4
RECOVERED = 4
IMMUNE = 4
DECEASED = 5

AVTIVITY_GROUP_NUMBER = 4
POPULATION_SIZE = 1000


class Consumer:
    def __init__(self, gender):
        self.gender = gender
        self.activity_group_paid = np.zeros(AVTIVITY_GROUP_NUMBER)


consumers = []
for i in range(0, POPULATION_SIZE):
    consumers



scalar_factors_distribution = {
    "age_group": {
        "values": ["children", "adolescent", "adult"],
        "weights": [0.25,0.25,0.5]
    }
}



contact_matrix = {
            "scalar_factor": "age_group",
            "values": {
                "children": {"children": 0.2, 
                            "adolescent": 2.0, 
                            "adult": 1.2},
                "adolescent": {"children": 2.0,
                            "adolescent": 3.6,
                            "adult": 1.5},
                "adult": {"children": 1.2,
                        "adolescent": 1.5,
                        "adult": 5.3}
            }
        }
weight_dist = {
    "family_members": {
        "weights": [0.163, 
                    0.403, 
                    0.17, 
                    0.026, 
                    0.1, 
                    0.138],
        "values": [9.7, 
                    8.3,
                    5.6,
                    4.9,
                    1.3,
                    1
                    ]
    },
    "acquaintances": {
        "weights": [1],
        "values": [2.45],
    },  
}



n_pop = 200

pop = Population(N=n_pop, 
                family_size=4, 
                acquaintance_size=10,
                scalar_factors_distribution=scalar_factors_distribution)
agents = pop.population_sample()


edge_sampler = EdgeSampler(agents,non_hh_contact_matrix=contact_matrix)
weight_sampler = WeightSampler(weight_dist)
vaccine = Vaccine(num_vaccine=30, effciency=0.998, reach_rate=0.7, immune_period=9)
ebola = Ebola()
simulation = DiseaseSimulation(agents,
                            200,
                            edge_sampler,
                            weight_sampler,
                            ebola,
                            vaccine)
simulation.initialize_seed_cases(10)

import time

start_time = time.time() #------------------------
res = simulation.run_simulation() 
end_time = time.time() #--------------------------

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
print(res)
return res
res.to_csv("result.csv", sep=";")

plot = res[['susceptible', 'incubated', 'infected', 'deceased', 'removed']].div(n_pop).plot()
plot.set_xlabel("Day")
plot.set_ylabel("Ratio of population")
plot.set_title('Simulation')
plot.set_xticks(range(0, len(res), 20))
plt.show()


# n_pop = 2000

# pop = Population(N=n_pop, 
#                  family_size=4, 
#                  acquaintance_size=10,
#                  scalar_factors_distribution=scalar_factors_distribution)
# agents = pop.population_sample()


# edge_sampler = EdgeSampler(agents,non_hh_contact_matrix=contact_matrix)
# weight_sampler = WeightSampler(weight_dist)
# vaccine = Vaccine(num_vaccine=30, effciency=0.998, reach_rate=0.7, immune_period=9)
# ebola = Ebola()
# simulation = DiseaseSimulation(agents,
#                                200,
#                                edge_sampler,
#                                weight_sampler,
#                                ebola,
#                                vaccine)
# simulation.initialize_seed_cases(10)

# import time

# start_time = time.time() #------------------------
# res = simulation.run_simulation() 
# end_time = time.time() #--------------------------

# elapsed_time = end_time - start_time

# print(f"Elapsed time: {elapsed_time} seconds")

# res.to_csv("result.csv", sep=";")

# plot = res[['susceptible', 'incubated', 'infected', 'deceased', 'removed']].div(n_pop).plot()
# plot.set_xlabel("Day")
# plot.set_ylabel("Ratio of population")
# plot.set_title('Simulation')
# plot.set_xticks(range(0, len(res), 20))
# plt.show()
