from pandas import read_csv
from numpy import uint8
from math import floor
from os import listdir
from random import random, randint, sample
from copy import deepcopy as dcopy
from statistics import mean, stdev

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
    # Getting #Machines (m) & #Taks (n) from file name & (c) case
    c = int(name[-5])
    m = int(name[-6]) * 10
    if c == 0:
        m -= 10
    n = int(name[:-6]) * 100
    # Get DF of instance
    data_ = read_csv(f'{folder_dir}/{name}', sep="\t", skiprows=[0, 1], header=None)
    # Drop columns of number of machines & NaN's
    data_.drop(data_.columns[[0] + [i for i in range(1, (m * 2) + 2, 2)]], inplace=True, axis=1)
    # Converting data to matrix
    instance_ = data_.to_numpy()
    # Get cplex value from list.txt
    with open(f'{folder_dir}/list.txt') as file:
        lines = [read_line for read_line in file]
        for line in lines:
            split_line = line.split()
            if split_line[0] == name:
                cplex = int(split_line[1])
                break



    return m, n, c, cplex, instance_


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

def print_results(results, name='0-dictResults.txt', filepath='Results/'):
    with (open(f'{filepath}/{name}', 'w') as file):
        print('-----------------------------------------------------------------\n')
        print('-----------------------------------------------------------------\n')
        print('Individual Summary Dump------------------------------------------\n')
        file.write('-----------------------------------------------------------------\n')
        file.write('-----------------------------------------------------------------\n')
        file.write('Individual Summary Dump------------------------------------------\n')
        for key_0, u_dist in results.items():
            for key_1, tasks in u_dist.items():
                for key_2, machines in tasks.items():
                    print(f'{key_0}//{key_1}//{key_2}:::')
                    print(' >> Name: Mean <> Max-Min <> Stdev <> Values \n')
                    file.write(f'{key_0}//{key_1}//{key_2}:::')
                    file.write(' >> Name: Mean <> Max-Min <> Stdev <> Values \n')
                    for key_3, values in machines.items():
                        mean_ = mean(values)
                        max_ = [max(values)]
                        max_.append(values.index(max_[0]))
                        min_ = [min(values)]
                        min_.append(values.index(min_[0]))
                        stdev_ = stdev(values)
                        print(f'\t{key_3}: {mean_} <> {max_}-{min_} <> {stdev_} <> {values}\n')
                        file.write(f'\t{key_3}: {mean_} <> {max_}-{min_} <> {stdev_} <> {values}\n')
                        values.append([mean_, max_, min_, stdev_])
        u_dist = results.keys()
        tasks = ['100', '200', '500', '1000']
        machines = ['10', '20', '30', '40', '50']
        metrics = ['cplex', 'fitness', 'cm', 'time', 'moves', 'visit']
        print('\n\n-----------------------------------------------------------------\n')
        print('-----------------------------------------------------------------\n')
        print('Complete Results-------------------------------------------------\n')
        file.write('\n\n-----------------------------------------------------------------\n')
        file.write('-----------------------------------------------------------------\n')
        file.write('Complete Results-------------------------------------------------\n')
        # Per task
        r = [0.0 for _ in metrics]
        for task in tasks:
            count = 0
            for u in u_dist:
                for machine in machines:
                    for i, m in enumerate(metrics):
                        count += 1
                        r[i] += results[u][f't{task}'][f'm{machine}'][m][-1][0]
            print(f'\tT -> {task}: {[v / count for v in r]}\n')
            file.write(f'\tT -> {task}: {[v / count for v in r]}\n')
        print('-----------------------------------------------------------------\n')
        file.write('-----------------------------------------------------------------\n')
        # Per machine
        r = [0.0 for _ in metrics]
        for machine in machines:
            count = 0
            for u in u_dist:
                for task in tasks:
                    for i, m in enumerate(metrics):
                        count += 1
                        r[i] += results[u][f't{task}'][f'm{machine}'][m][-1][0]
            print(f'\tM -> {machine}: {[v / count for v in r]}\n')
            file.write(f'\tM -> {machine}: {[v / count for v in r]}\n')
        print('-----------------------------------------------------------------\n')
        file.write('-----------------------------------------------------------------\n')
        # Per U
        r = [0.0 for _ in metrics]
        for u in u_dist:
            count = 0
            for machine in machines:
                for task in tasks:
                    for i, m in enumerate(metrics):
                        count += 1
                        r[i] += results[u][f't{task}'][f'm{machine}'][m][-1][0]
            print(f'\tU -> {u}: {[v / count for v in r]}\n')
            file.write(f'\tU -> {u}: {[v / count for v in r]}\n')
        print('-----------------------------------------------------------------')
        file.write('-----------------------------------------------------------------')

def test_excel():
    rep = [[36, 56, 77, 83, 69, 12, 89, 30], [13, 42, 6, 5, 68, 75, 24, 22, 60, 11, 87, 54],
           [7, 78, 70, 28, 16, 53, 80, 66, 4], [84, 91, 51, 81, 20, 46, 99],
           [33, 25, 3, 96, 37, 93, 45, 76, 90, 27, 61, 19], [67, 29, 85, 23, 94, 52, 58, 35, 82, 49],
           [34, 0, 79, 59, 44, 48, 72, 57, 40, 55], [32, 74, 43, 98, 18, 63, 50, 26, 86, 2, 31],
           [1, 47, 21, 38, 62, 9, 92, 95, 10, 15, 39], [17, 8, 14, 88, 65, 73, 71, 97, 64, 41]]
    a = []
    for i, m in enumerate(rep):
        a += m
    print(len(a))
    print(i)

    letters = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    for l, r in zip(letters, rep):
        s = ''
        for t in r:
            s += f'{l}{t + 1}' + ', '
        print(f'=SUMA({s[:-2]})')

    present = [0 for _ in range(100)]
    for m in rep:
        for t in m:
            present[t] += 1
    print(present)

def chunker(seq, size):
    # Taken from Stackoverflow: How to iterate over list in chunks
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


if __name__ == '__main__':
    # Prepare results
    results = {}
    dict_results = dict(cplex=[],
                        fitness=[],
                        cm=[],
                        time=[],
                        moves=[],
                        visit=[])
    dict_machines = dict(m10=dcopy(dict_results),
                         m20=dcopy(dict_results),
                         m30=dcopy(dict_results),
                         m40=dcopy(dict_results),
                         m50=dcopy(dict_results))
    dict_tasks = dict(t100=dcopy(dict_machines),
                      t200=dcopy(dict_machines),
                      t500=dcopy(dict_machines),
                      t1000=dcopy(dict_machines))
    # Load data
    folder_path = '../Instances/Instances/'
    u_instances = listdir(folder_path)  # U() distributions
    for u in u_instances:
        print('> Doing ', u)
        results[u] = dcopy(dict_tasks)
        instances = all_instances(folder_path + u)
        for i in instances:
            n_machines, n_tasks, case, cplex, data = get_instance(i, folder_path + u)
            print(f'>> On instance {i} (Case {case})')
            results[u][f't{n_tasks}'][f'm{n_machines}']['cplex'].append(random())
            results[u][f't{n_tasks}'][f'm{n_machines}']['fitness'].append(random())
            results[u][f't{n_tasks}'][f'm{n_machines}']['cm'].append(random())
            results[u][f't{n_tasks}'][f'm{n_machines}']['time'].append(random())
            results[u][f't{n_tasks}'][f'm{n_machines}']['moves'].append(random())
            results[u][f't{n_tasks}'][f'm{n_machines}']['visit'].append(random())
    # Print results
    # for key_0, u_dist in results.items():
    #     print('>', key_0)
    #     print('>', u_dist)
    #     for key_1, tasks in u_dist.items():
    #         print('>>>', key_1)
    #         print('>>>', tasks)
    #         for key_2, machines in tasks.items():
    #             print('>>>>>', key_2)
    #             print('>>>>>', machines)
    #             for key_3, values in machines.items():
    #                 print('>>>>>', key_3)
    #                 print('>>>>>', values)
    print_results(results)

    print('End')

