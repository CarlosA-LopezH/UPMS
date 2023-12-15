from methods import *
from random import choice, seed
from time import time
from statistics import mean, stdev

seed(0)


class Solution_HC:
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

        self.update_current(data)

    def update_current(self, data):
        self.c = generate_cvalues(self.n_machines, self.rep, data)
        self.c_max = get_cmax(self.c)
        self.fitness = get_fitness(self.c_max[0], len(self.c_max[1]))
        self.history.append(self.fitness)
        self.movements += 1


def neighbourhood_HC(rep, c, origin, data):
    move = None
    visited = 0
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
                        if origin_c < reference_c and target_c < reference_c:  # Accept the neighbor if there is an improvement on both C values
                            best_c = origin_c
                            reference_c = target_c
                            move = (origin_task, target, target_task)  # Define the move
                else:  # Else-If the target machine is not critical, do insertion
                    visited += 1
                    origin_c = c[origin] - data[origin_task][origin]  # Decrement the C value of origin machine with the origin task
                    target_c = c[target] + data[origin_task][target]  # Augment the C value of target machine with the origin task
                    if origin_c < best_c and target_c < c[origin]:  # Accept the neighbor if there is an improvement on origin C and target C dont become bigger than origin C (Critical)
                        best_c = origin_c
                        move = (origin_task, target)
    return move, visited


def HC_run(data, n_machines, n_tasks, trials=1):
    # Set trials
    solutions = []
    # Set counter
    for trial in range(trials):
        # Initialize solution
        initial_rep, _ = random_min(data,
                                    [task for task in range(n_tasks)],
                                    [machine for machine in range(n_machines)],
                                    [0 for _ in range(n_machines)])
        sol = Solution_HC(initial_rep, n_machines, data)
        stuck = False  # Set the stuck indicator to False
        while not stuck:
            stuck = True  # Stuck value will only change if not stuck
            origin_machine = choice(sol.c_max[1])  # Choose a (random) critical machine
            move, visited = neighbourhood_HC(sol.rep, sol.c, origin_machine, data)
            sol.neighbours += visited
            if move:  # If move is not None, then a better solution was found
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
        solutions.append(sol)  # Append the solution
    return solutions


if __name__ == '__main__':
    folder_path = '../Instances/Instances/'
    u_instances = listdir(folder_path)
    # Testing only on 1a100 (u_instances[4]) and 111.txt instances[50]
    instances = all_instances(folder_path + u_instances[4])[50]
    n_machines, n_tasks, data = get_instance(instances, folder_path + u_instances[4])
    st = time()
    final = HC_run(data, n_machines, n_tasks, 50)
    print('Run Time: ', time() - st)
    resume = [f.fitness for f in final]
    print('Results: ', resume)
    print('Min: ', min(resume), resume.index(min(resume)))
    print('Max: ', max(resume), resume.index(max(resume)))
    print('Mean: ', mean(resume))
    print('Stdev: ', stdev(resume))

    # all_instances('../Instances/Instances')
