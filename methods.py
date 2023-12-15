from pandas import read_csv
from numpy import uint8
from math import floor
from os import listdir
from random import random, randint, sample

class Solution_PSO:
    def __init__(self, representation, num_machines, data):
        self.rep = representation
        self.v = [random() for _ in range(num_machines)]
        self.num_machines = num_machines
        self.data = data
        self.c = []
        self.cmax = 0
        self.cl_mach = 0
        self.fitness = 0
        self.num_cl = 0
        self.gen = 0
        self.update_current(0)
        self.best_local = {}
        self.update_local()
        # self.best_global = {}

    def update_current(self, gen):
        self.c = generate_cvalues(self.num_machines, self.rep, self.data)
        self.cmax, self.cl_mach = get_cmax(self.c)
        self.fitness, self.num_cl = get_fitness(self.c)
        self.gen = gen

    def update_local(self):
        self.best_local = {'rep': self.rep,
                           'c': self.c,
                           'cmax': self.cmax,
                           'cl_mach': self.cl_mach,
                           'fitness': self.fitness,
                           'num_cl': self.num_cl,
                           'gen': self.gen}



def all_instances(folder_path):
    """
    Return all the instances in path.

    :param str folder_path: Results containing all instances
    :return list list_instances: List of all instances found in path
    """
    # Get all instances in folder
    list_instances = listdir(folder_path)
    # Drop 'list.txt'
    list_instances.remove('list.txt')
    return list_instances


def get_instance(name, folder_dir):
    """
    Return the instance data.

    :param str name: Name of the instance
    :param str folder_dir: Results containing all instances
    :return int n: number of machines
    :return int m: number of tasks
    :return list instance: Instance data
    """
    # Getting #Machines (m) & #Taks (n) from file name
    m = int(name[-6]) * 10
    n = int(name[:-6]) * 100
    # Get DF of instance
    data_ = read_csv(f'{folder_dir}/{name}', sep="\t", skiprows=[0, 1], header=None)
    # Drop columns of number of machines & NaN's
    data_.drop(data_.columns[[0] + [i for i in range(1, (m * 2) + 2, 2)]], inplace=True, axis=1)
    # Converting data to matrix
    instance_ = data_.to_numpy(dtype=uint8)

    return m, n, instance_


def rpd(cplex_val, val):
    """Relative percentage deviation"""
    return (val - cplex_val) / cplex_val

def get_min(task, data):
    task_values = [value for value in data[task]]
    min_value = min(task_values)
    min_index = task_values.index(min_value)
    return min_index


def generate_cvalues(machines, rep, data):
    # Generate a zero C array
    c_values = [0 for _ in range(machines)]
    # Fill the C values of each machine:
    # Iterate over the machines
    for m_i, machine in enumerate(rep):
        # Iterate over the task of a machine
        for t_i in machine:
            # Update the value of C of the machine with the task
            c_values[m_i] = c_values[m_i] + data[t_i][m_i]
    return c_values


def get_cmax(c_values):
    # Get the Cmax
    max_val = max(c_values)
    # Obtain all indexes that cotains the max value
    indexes = [i for i, v in enumerate(c_values) if v == max_val]
    # Return Cmax and its position
    return max_val, indexes


def count_similars(c_values, c):
    # Start the count in 0
    count = 0
    # Iterate over the C values
    for v in c_values:
        # If C value is Cmax, add 1
        if v == c:
            count += 1
    # Return the count minus 1 to eliminate the first occurrence
    return count


def get_fitness(cmax, repetition, divisor=10):
    # Get the Cmax value
    part_integer = cmax
    # Get the decimal part
    part_decimal = (repetition - 1) / divisor
    # Report the fitness and the index of Cmax
    return part_integer + part_decimal


def heuristic_min(data, task, machines, c_values):
    # Get the processing times
    c = [data[task][machine] + c_values[machine] for machine in machines]
    # Get min and index
    min_val = min(c)
    indx = c.index(min_val)

    return min_val, indx


def random_min(data, tasks, machines, c_values):
    # Permute the tasks randomly
    per_tasks = sample(tasks, k=len(tasks))
    # Initialize solution
    solution = [[] for _ in machines]
    # Iterate over tasks
    for task in per_tasks:
        # Get min value and index from Min() heuristic
        min_val, indx = heuristic_min(data, task, machines, c_values)
        # Add task in machine
        solution[indx].append(task)
        # Update processing time
        c_values[indx] = min_val

    return solution, c_values


def distance(current, reference):
    set_current = set(current)
    set_reference = set(reference)
    similarity = len(set_current & set_reference) / (len(set_current | set_reference) + 1)
    distance = 1 - similarity
    return distance


def shared(v, len_group):
    return floor((1 - v) * len_group)