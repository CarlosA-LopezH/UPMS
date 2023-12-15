from methods import *
from random import random, choice, seed
from math import exp
from time import time
from statistics import mean, stdev
import matplotlib.pyplot as plt

seed(0)

class Solution_SA:
    def __init__(self, representation, num_machines, data):
        # Initialize variables
        self.rep = representation
        self.n_machines = num_machines
        self.c = []
        self.c_max = ()
        self.fitness = 0.0
        self.neighbours = 0
        self.movements = -1
        self.history = []
        self.best = {'fitness': 0}

        self.update_current(data, True)

    def update_current(self, data, initial=False):
        self.c = generate_cvalues(self.n_machines, self.rep, data)
        self.c_max = get_cmax(self.c)
        self.fitness = get_fitness(self.c_max[0], len(self.c_max[1]))
        self.history.append(self.fitness)
        self.movements += 1
        if self.fitness <= self.best['fitness'] or initial:
            self.best = {'Rep': self.rep,
                         'c': self.c,
                         'c_max': self.c_max,
                         'fitness': self.fitness,
                         'neighbors': self.neighbours,
                         'movements': self.movements}


def cooling(current_temp, beta=0.0020):
    """
    Cooling method for the control of temperatures. This function was taken for
        Glass, 94
    :param float current_temp: Current temperature to be modified.
    :param float beta: Constant value for modification.
    :return:
    """
    return current_temp / (1 + (beta * current_temp))


def acceptance(diff, temp):
    """
    Probability of acceptance.
    :param float diff: Difference between values
    :param float temp: Current temperature
    :return:
    """
    return exp(diff / temp)


def neighbourhood_SA(rep, c, origin, temp, data, best):
    """ This is a simplified version of the neighborhood function of HC"""
    move = None
    visited = 1
    best_c = c[origin]
    origin_task = choice(rep[origin])  # Select a random task from origin
    target = choice([i for i, _ in enumerate(c) if i != origin])  # Select a random target machine different than origin
    if c[target] == c[origin]:  # If the target machine is critical, then do interchange
        target_task = choice(rep[target])  # Select a random task from target
        origin_c = c[origin] - data[origin_task][origin]  # Decrement the C value of origin machine with the origin task
        target_c = c[target] - data[target_task][target]  # Decrement the C value of target machine with the target task
        origin_c += data[target_task][origin]  # Augment the C value of origin machine with the target task
        target_c += data[origin_task][target]  # Augment the C value of target machine with the origin task
        difference = [best_c - origin_c, best_c - target_c]  # Calculate the difference of both moves
        accept = [acceptance(difference[0], temp), acceptance(difference[1], temp)]  # Calculate the acceptance value
        if (difference[0] > 0 and difference[1]) > 0 or (random() < accept[0] and random() < accept[1]):  # If the difference is good for either moves or the acceptance value is met for both moves
            move = (origin_task, target, target_task)  # Define the move
    else:  # Else-If the target machine is not critical, do insertion
        origin_c = c[origin] - data[origin_task][origin]  # Decrement the C value of origin machine with the origin task
        target_c = c[target] + data[origin_task][target]  # Augment the C value of target machine with the origin task
        difference = [best_c - origin_c, c[target] - target_c]  # Calculate the difference of the move
        accept = [acceptance(difference[0], temp), acceptance(difference[1], temp)]  # Calculate the acceptance value
        if (difference[0] > 0 and difference[1] > 0) or (random() < accept[0] and random() < accept[1]):
            move = (origin_task, target)  # Define the move
    return move, visited


def SA_run(data, n_machines, n_tasks, max_temp, min_temp, max_count=10000, trials=1, beta=0.0020):
    # Set trials
    solutions = []
    for trial in range(trials):  # Loop for tries: Halting criterion (1)
        print('> Trial: ', trial)  # >>>>>>>>>>>>>>>
        # Initialize solution
        initial_rep, _ = random_min(data,
                                    [task for task in range(n_tasks)],
                                    [machine for machine in range(n_machines)],
                                    [0 for _ in range(n_machines)])
        sol = Solution_SA(initial_rep, n_machines, data)
        current_temp = max_temp  # Initialize the temperature value
        while current_temp > min_temp:  # Halting criterion: Loop for Temperature.
            print('> Temperature: ', current_temp)  # >>>>>>>>>>>>>>>
            stuck = False  # Set the stuck indicator to False
            count = 0
            while count < max_count:  # Termination condition:
                stuck = True  # Stuck value will only change if not stuck
                origin_machine = choice(sol.c_max[1])  # Choose a (random) critical machine
                move, visited = neighbourhood_SA(sol.rep, sol.c, origin_machine, current_temp, data, sol.best['c_max'][0])
                sol.neighbours += visited
                if move:  # If move is not None, then a better solution was found or acceptance was applied
                    stuck = False
                    # Performed the move
                    if len(move) > 2:
                        origin_task, target_machine, target_task = move
                        sol.rep[origin_machine].remove(origin_task)  # Remove origin task from origin machine
                        sol.rep[target_machine].remove(target_task)  # Remove target task from target machine
                        sol.rep[target_machine].append(origin_task)  # Add origin task to target machine
                        sol.rep[origin_machine].append(target_task)  # Add target task to origin machine
                    else:
                        origin_task, target_machine = move
                        sol.rep[origin_machine].remove(origin_task)  # Remove origin task from origin machine
                        sol.rep[target_machine].append(origin_task)  # Add origin task to target machine
                    sol.update_current(data)  # Update all values of solution
                count += 1
            current_temp = cooling(current_temp, beta)  # Update temperature
        solutions.append(sol)  # Append the solution
    return solutions


if __name__ == '__main__':
    folder_path = '../Instances/Instances/'
    u_instances = listdir(folder_path)
    # Testing only on 1a100 (u_instances[4]) and 111.txt instances[50]
    instances = all_instances(folder_path + u_instances[4])[50]
    n_machines, n_tasks, data = get_instance(instances, folder_path + u_instances[4])
    st = time()
    max_temp = 10
    min_temp = 1
    final = SA_run(data, n_machines, n_tasks, max_temp, min_temp, max_count=100, trials=5, beta=0.0020)
    print('Run Time: ', time() - st)
    resume = [f.best['fitness'] for f in final]
    print('Results: ', resume)
    print('Min: ', min(resume), resume.index(min(resume)))
    print('Max: ', max(resume), resume.index(max(resume)))
    print('Mean: ', mean(resume))
    print('Stdev: ', stdev(resume))