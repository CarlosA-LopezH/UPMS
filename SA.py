from methods import *
from random import random, choice, seed
from math import exp, log
from time import time

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
    target = choice([i for i, _ in enumerate(c) if i != origin])  # Select a random target machine different from origin
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


def neighbourhood_SA_1(rep, c, origin, temp, data, best):
    """ Different version than HC"""
    move = None
    visited = 1
    best_difference = 0
    best_c1 = c[origin]
    origin_c = c[origin]
    target = sorted([i for i in range(len(c))], key=lambda x: c[x])[0]  # Select the fastest target machine
    best_c2 = c[target]
    target_c = c[target]
    for i, machine in enumerate(rep):
        if machine:
            origin_c = c[i]
            origin_task = choice(machine)
            target = choice([j for j in range(len(c)) if j != i])
            target_c = c[target]
            origin_c -= data[origin_task][i]
            target_c += data[origin_task][target]
            difference = (c[i] - origin_c) + (c[target] - target_c)
            accept = acceptance(difference, temp)
            if difference > best_difference or random() < accept:
                move = (i, origin_task, target)
    return move, visited


def neighbourhood_SA_2(rep, c, origin, temp, data):
    """ Different version than HC"""
    move = None
    visited = 0
    best_difference = 0
    best_c = c[origin]
    for origin_task in rep[origin]:  # Iterate over all tasks in origin
        for target, machine in enumerate(rep):  # Iterate over all machines
            if target != origin:  # Avoid origin machine
                if c[target] == c[origin]:  # If the target machine is critical, then do interchange
                    reference_c = c[target]
                    for target_task in machine:  # Iterate over the tasks from target machine
                        visited += 1
                        origin_c = c[origin] - data[origin_task][origin]  # Decrement the C value of origin machine with the origin task
                        target_c = c[target] - data[target_task][target]  # Decrement the C value of target machine with the target task
                        origin_c += data[target_task][origin]  # Augment the C value of origin machine with the target task
                        target_c += data[origin_task][target]  # Augment the C value of target machine with the origin task
                        all_diff = [best_c - origin_c, reference_c - target_c]
                        difference = all_diff[0] if all_diff[0] < all_diff[1] else all_diff[1]
                        accept = acceptance(difference, temp)
                        if difference > best_difference or random() < accept:  # Accept the neighbor if there is an improvement on both C values
                            best_c = origin_c
                            reference_c = target_c
                            best_difference = difference if difference > best_difference else best_difference
                            move = (origin_task, target, target_task)  # Define the move
                else:  # Else-If the target machine is not critical, do insertion
                    visited += 1
                    origin_c = c[origin] - data[origin_task][origin]  # Decrement the C value of origin machine with the origin task
                    target_c = c[target] + data[origin_task][target]  # Augment the C value of target machine with the origin task
                    all_diff = [best_c - origin_c, best_c - target_c]
                    difference = all_diff[0] if all_diff[0] < all_diff[1] else all_diff[1]
                    accept = acceptance(difference, temp)
                    if difference > best_difference or random() < accept:  # Accept the neighbor if there is an improvement on origin C and target C dont become bigger than origin C (Critical)
                        best_c = origin_c
                        best_difference = difference if difference > best_difference else best_difference
                        move = (origin_task, target)
    return move, visited


def SA_run_1(data, n_machines, n_tasks, max_temp, min_temp, max_count=10000, trials=1, beta=0.0020):
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
            stuck = False  # Set the stuck indicator to False
            count = 0
            while count < max_count and not stuck:  # Termination condition:
                stuck = True  # Stuck value will only change if not stuck
                origin_machine = choice(sol.c_max[1])  # Choose a (random) critical machine
                move, visited = neighbourhood_SA_2(sol.rep, sol.c, origin_machine, current_temp, data)
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
                        sol.rep[target_machine].append(origin_task)
                    sol.update_current(data)  # Update all values of solution
                count += 1
                if stuck:
                    print('Stuck')
            current_temp = cooling(current_temp, beta)  # Update temperature
        solutions.append(sol)  # Append the solution
    return solutions


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
    print(u_instances)
    # Testing only on 1a100 (u_instances[4]) and 111.txt instances[50] (50 for 111, 0 for 1011, 40 for 1051)
    instances = all_instances(folder_path + u_instances[4])
    print(instances)
    n_machines, n_tasks, data = get_instance(instances[40], folder_path + u_instances[4])
    # Calculate Max temp, min temp and beta
    init_acceptance = 0.90
    final_acceptance = 0.10
    n_temps = 200
    max_count = 10
    max_diff = data.max()
    min_diff = data.min()
    max_temp = (-1 * max_diff) / log(init_acceptance)
    min_temp = (-1 * min_diff) / log(final_acceptance)
    beta = (max_temp - min_temp) / ((n_temps - 1) * max_temp * min_temp)
    st = time()
    # max_temp = 615.3
    # min_temp = 0.83
    final = SA_run_1(data, n_machines, n_tasks, max_temp, min_temp, max_count=max_count, trials=1, beta=beta)
    print('Run Time: ', time() - st)
    resume = [f.best['fitness'] for f in final]
    print('Results: ', resume)