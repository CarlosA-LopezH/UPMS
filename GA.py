from methods import *
from random import seed, sample, choice
from copy import deepcopy
from time import time
import json

seed(0)

def rank_sort(population, individuals):
    # Sort individuals based on fitness and generation
    individuals.sort(key=lambda x: (population[x].fitness, population[x].gen[0]))
    mark = individuals[-1]
    # Move all repeated values to end of population
    values = []
    ppl = []
    repeat = []
    for i in individuals:
        if population[i].fitness in values:
            repeat.append(i)
        else:
            values.append(population[i].fitness)
            ppl.append(i)
    individuals = ppl + repeat

    return individuals, mark

def print_population(population):
    return [p.fitness for p in population]

# Variation operators
def crossover(data, n_machines, tasks, machines, parents, population):
    # >>> Initialize offsprings
    c1 = [[] for _ in machines]
    c2 = [[] for _ in machines]
    # >>> Create first offspring
    choice_p1 = choice([0, 1])
    choice_p2 = 0 if choice_p1 == 1 else 1
    parent1 = parents[choice_p1]
    parent2 = parents[choice_p2]
    # Track registered tasks & machines
    reg_tasks = [0 for _ in tasks]
    reg_machines = [0 for _ in machines]
    # Sort the machines by C value for both parents
    p1 = [i for i in machines]
    p1.sort(key=lambda x: population[parent1].c[x])
    p2 = [i for i in machines]
    p2.sort(key=lambda x: population[parent2].c[x])
    # Iterate over the machines from parent 1 (m1) and parent 2 (m2) in order
    for m1, m2 in zip(p1, p2):
        # > Gen from p1
        # If m1 is not already in child, use gen
        if reg_machines[m1] == 0:
            # Register the machine
            reg_machines[m1] = 1
            # Iterate over tasks in machine m1
            for task in population[parent1].rep[m1]:
                # If task is not already in child, add it
                if reg_tasks[task] == 0:
                    c1[m1].append(task)
                    # Register the task
                    reg_tasks[task] = 1
        # > Gen from p2
        # If m2 is not already in child, use gen
        if reg_machines[m2] == 0:
            # Register the machine
            reg_machines[m2] = 1
            # Iterate over tasks in machine m2
            for task in population[parent2].rep[m2]:
                # If task is not already in child, add it
                if reg_tasks[task] == 0:
                    c1[m2].append(task)
                    # Register the task
                    reg_tasks[task] = 1
    # Re-insert the missing tasks (tasks no registered) by random insertion
    missing = [t for t in tasks if reg_tasks[t] == 0]
    missing = sample(missing, k=(len(missing)))
    c = generate_cvalues(n_machines, c1, data)
    for task in missing:
        # index = choice(machines)
        val, index = heuristic_min(data, task, machines, c)
        c1[index].append(task)
        c[index] = val
        reg_tasks[task] += 1

    # >>> Create second offspring - This will consist in a comparison machine by machine bw both parents
    # Track registered tasks & machines
    reg_tasks = [0 for _ in tasks]
    for m in machines:
        if population[parents[0]].c[m] < population[parents[1]].c[m]:
            winner = 0
        else:
            winner = 1
        for task in population[parents[winner]].rep[m]:
            if reg_tasks[task] == 0:
                c2[m].append(task)
                reg_tasks[task] += 1

    # Re-insert the missing tasks (tasks no registered) - Not sure if this applies for this process
    missing = [t for t in tasks if reg_tasks[t] == 0]
    missing = sample(missing, k=(len(missing)))
    c = generate_cvalues(n_machines, c2, data)
    for task in missing:
        # index = choice(machines)
        val, index = heuristic_min(data, task, machines, c)
        c2[index].append(task)
        c[index] = val
        reg_tasks[task] += 1

    return c1, c2
    # return c2
def crossover_old(data, n_machines, tasks, machines, parents, population):
    # >>> Initialize offsprings
    c1 = [[] for _ in machines]
    c2 = [[] for _ in machines]
    # >>> Create first offspring
    parent1 = parents[0]
    parent2 = parents[1]
    # Track registered tasks & machines
    reg_tasks = [0 for _ in tasks]
    reg_machines = [0 for _ in machines]
    # Sort the machines by C value for both parents
    p1 = [i for i in machines]
    p1.sort(key=lambda x: population[parent1].c[x])
    p2 = [i for i in machines]
    p2.sort(key=lambda x: population[parent2].c[x])
    # Iterate over the machines from parent 1 (m1) and parent 2 (m2) in order
    for m1, m2 in zip(p1, p2):
        # > Gen from p1
        # If m1 is not already in child, use gen
        if reg_machines[m1] == 0:
            # Register the machine
            reg_machines[m1] = 1
            # Iterate over tasks in machine m1
            for task in population[parent1].rep[m1]:
                # If task is not already in child, add it
                if reg_tasks[task] == 0:
                    c1[m1].append(task)
                    # Register the task
                    reg_tasks[task] = 1
        # > Gen from p2
        # If m2 is not already in child, use gen
        if reg_machines[m2] == 0:
            # Register the machine
            reg_machines[m2] = 1
            # Iterate over tasks in machine m2
            for task in population[parent2].rep[m2]:
                # If task is not already in child, add it
                if reg_tasks[task] == 0:
                    c1[m2].append(task)
                    # Register the task
                    reg_tasks[task] = 1
    # Re-inserte  the missing tasks (tasks no registered) by random insertion
    missing = [t for t in tasks if reg_tasks[t] == 0]
    missing = sample(missing, k=(len(missing)))
    c = generate_cvalues(n_machines, c1, data)
    for task in missing:
        # index = choice(machines)
        _, index = heuristic_min(data, task, machines, c)
        c1[index].append(task)
        reg_tasks[task] += 1

    # >>> Create second offspring
    # parent1 = parents[1]
    # parent2 = parents[0]
    # # Track registered tasks & machines
    # reg_tasks = [0 for _ in tasks]
    # reg_machines = [0 for _ in machines]
    # # Sort the machines by C value for both parents
    # p1 = [i for i in machines]
    # p1.sort(key=lambda x: population[parent1].c[x])
    # p2 = [i for i in machines]
    # p2.sort(key=lambda x: population[parent2].c[x])
    # # Iterate over the machines from parent 1 (m1) and parent 2 (m2) in order
    # for m1, m2 in zip(p1, p2):
    #     # > Gen from p1
    #     # If m1 is not already in child, use gen
    #     if reg_machines[m1] == 0:
    #         # Register the machine
    #         reg_machines[m1] += 1
    #         # Iterate over tasks in machine m1
    #         for task in population[parent1].rep[m1]:
    #             # If task is not already in child, add it
    #             if reg_tasks[task] == 0:
    #                 c2[m1].append(task)
    #                 # Register the task
    #                 reg_tasks[task] += 1
    #     # > Gen from p2
    #     # If m2 is not already in child, use gen
    #     if reg_machines[m2] == 0:
    #         # Register the machine
    #         reg_machines[m2] += 1
    #         # Iterate over tasks in machine m2
    #         for task in population[parent2].rep[m2]:
    #             # If task is not already in child, add it
    #             if reg_tasks[task] == 0:
    #                 c2[m2].append(task)
    #                 # Register the task
    #                 reg_tasks[task] += 1
    # # Re-inserte  the missing tasks (tasks no registered)
    # missing = [t for t in tasks if reg_tasks[t] == 0]
    # missing = sample(missing, k=(len(missing)))
    # c = generate_cvalues(n_machines, c1, data)
    # for task in missing:
    #     # index = choice(machines)
    #     _, index = heuristic_min(data, task, machines, c)
    #     c2[index].append(task)
    #     reg_tasks[task] += 1

    return c1, c2


def survivor_xorver(population, individuals, offsprings, nc, set_r):
    # First nc/2 offsprings replace the R set
    for child, r in zip(offsprings[:int(nc / 2)], set_r):
        population[r] = child
    # Rest (nc/2) offsprings replace the duplicates and worst fitness
    replace = [i for i in reversed(individuals) if i not in set_r]
    for child, w in zip(offsprings[int(nc / 2):], replace):
        population[w] = child

    return population


def mutation(data, machines, rep, c, kc=2):
    # Re-allocate all task from a critical, a random one and the busiest  machine
    critical = sorted(machines, key=lambda x: c[x])[-kc:]
    # busiest = sorted(machines, key=lambda x: [len(m) for m in rep][x])[-2:]
    # busy = busiest[1] if busiest[1] != critical else busiest[0]
    # rand = choice([m for m in machines if m not in critical])
    # >> Empty the critical, busy & random machine + [t for t in rep[rand]] + [t for t in rep[critical[1]]]
    r_tasks = []
    for cm in critical:
        r_tasks += rep[cm]
        rep[cm].clear()
        c[cm] = 0
    # rep[critical[0]].clear()
    # rep[critical[1]].clear()
    # rep[busy].clear()
    # rep[rand].clear()

    # c[critical[0]] = 0
    # c[critical[1]] = 0
    # c[busy] = 0
    # c[rand] = 0
    # >> Re-allocate tasks using Min()
    for t in r_tasks:
        val, index = heuristic_min(data, t, machines, c)
        rep[index].append(t)
        c[index] = val



    return c


def selection_xover(individuals, nc, nb):
    # Select the best group (g) & the random group (r).
    set_g = individuals[:nc]
    set_r = individuals[nb:]
    g = sample(set_g, int(nc / 2))
    r = sample(set_r, int(nc / 2))
    # Ensure that no pair will have the same individual
    for i, j in enumerate(g):
        if j == r[i]:
            if i == int(nc / 2) - 1:
                r[i], r[0] = r[0], r[i]
            else:
                r[i], r[i + 1] = r[i + 1], r[i]
    # Create pair solutions as parents.
    parents = [(p1, p2) for p1, p2 in zip(g, r)]

    return parents, g, r


def selection_mut(individuals, nm, nb, pb=0.70):
    # Select the pb (%) of nm best individuals (g)
    set_g = individuals[:int(pb * nm)]
    # Select the rest of duplicated and worst fitness, except the ones that will
    #   be replaced by the mutation of elit group
    set_w = individuals[int(pb * nm) - nm - nb:-nb]

    return set_g + set_w


class Solution:
    def __init__(self, representation, num_machines, data, gen, source='Init'):
        self.rep = representation
        self.n_machines = num_machines
        self.c = generate_cvalues(self.n_machines, self.rep, data)
        self.c_max = get_cmax(self.c)
        self.fitness = get_fitness(self.c_max[0], len(self.c_max[1]))
        self.c = generate_cvalues(num_machines, self.rep, data)
        self.gen = (gen, source)

    def update(self, gen, source='M'):
        self.c_max = get_cmax(self.c)
        self.fitness = get_fitness(self.c_max[0], len(self.c_max[1]))
        self.gen = (gen, source)

    def report(self):
        print(f'Individual from Gen: {self.gen}')
        print(f'Cmax: {self.cmax} in machine {self.num_cl} machines')
        print(f' Fitness: {self.fitness} repeated in {self.num_cl}')
        print(f'Representation: {self.rep}')
        print(f'C values: {self.c}')

def GA(population_size, generations, nc, nm, nb, pb, tasks, machines, individuals, data):
    # Store information per instances
    avg_fitness = []
    best_fitness = []
    best_cmax = []
    stdev_fitness = []
    cms = []
    gen = []
    # ------------------------- Initialization
    # Initialize empty population
    population = []
    # Initial generation
    current_gen = 0
    # Create individuals
    for individual in range(population_size):
        # Initilize c_values
        c_values = [0 for _ in machines]
        # Generate individual using random_min strategy
        rep, c_values = random_min(data, tasks, machines, c_values)
        # Add individual to population
        s = Solution(rep, len(machines), data, current_gen)
        population.append(s)
    # Rank population
    individuals, mark = rank_sort(population, individuals)
    # Set the current best
    best_individual = population[individuals[0]]
    while current_gen < generations:
        # Augment generation
        current_gen += 1
        # print(current_gen)

        # ------------------------- Crossover
        # Selection for crossover
        pair_parents, g, r = selection_xover(individuals, nc, nb)
        # Crossover
        xover_set = []
        for parents in pair_parents:
            # Do crossover
            child1, child2 = crossover(data, len(machines), tasks, machines, parents, population)
            # child1, _ = crossover(data, len(machines), tasks, machines, parents, population)
            # Insert offsprings into population
            s1 = Solution(child1, len(machines), data, current_gen, source='X1')
            xover_set.append(s1)
            s2 = Solution(child2, len(machines), data, current_gen, source='X2')
            xover_set.append(s2)
        # Replace individuals with offspring
        population = survivor_xorver(population, individuals, xover_set, nc, r)
        # Rank population
        individuals, mark = rank_sort(population, individuals)

        # ------------------------- Mutation
        # Selection for mutation
        to_mutate = selection_mut(individuals, nm, nb, pb)
        # Set the replacement mutation from elit
        clone_b = individuals[-nb:]
        for i, b in enumerate(clone_b):
            # population[b].rep = [m for m in population[to_mutate[i]].rep]
            # population[b].c = [k for k in population[to_mutate[i]].c]
            population[b] = deepcopy(population[i])
        to_mutate[:nb] = clone_b
        # Mutation
        for individual in to_mutate:
            # Mutate the representation
            mutation(data, machines, population[individual].rep, population[individual].c, len(population[individual].c_max[1]))
            population[individual].update(current_gen)
        # Rank population
        individuals, mark = rank_sort(population, individuals)

        # Update best
        if best_individual.fitness > population[individuals[0]].fitness:
            best_individual = population[individuals[0]]

        # Add information of process
        best_fitness.append(best_individual.fitness)
        best_cmax.append(best_individual.c_max[0])
        gen.append(best_individual.gen)
        avg_fitness.append(mean(print_population(population)))
        stdev_fitness.append(stdev(print_population(population)))
        cms.append(len(best_individual.c_max[1]))

    return best_fitness, gen, avg_fitness, stdev_fitness, cms, best_cmax, best_individual, population


if __name__ == '__main__':
    method = 'GA'
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
    # Genetic parameters:
    population_size = 100
    generations = 300
    nc = 80
    nm = 60
    nb = 1
    pb = 0.8
    folder_path = '../Instances/Instances/'
    u_instances = listdir(folder_path)
    total_time = time()
    for u in u_instances:
        print('> Doing ', u)
        results[u] = dcopy(dict_tasks)
        instances = all_instances(folder_path + u)
        for i in instances:
            # Store information overall
            overall = []
            times = []
            gens = []
            source = []
            fitness = []
            n_machines, n_tasks, case, cplex, data = get_instance(i, folder_path + u)
            # Indices for machines, tasks, & individuals
            machines = [j for j in range(n_machines)]
            tasks = [j for j in range(n_tasks)]
            individuals = [j for j in range(population_size)]
            print(f'>> On instance {i} (Case {case})')
            st = time()
            all_results = GA(population_size, generations, nc, nm, nb, pb, tasks, machines, individuals, data)
            et = time()
            print('RPD: ', rpd(cplex, all_results[6].c_max[0]))
            print('Gen: ', all_results[6].gen)
            print('Time: ', et - st)
            # Store results
            final = [all_results[6]]
            results[u][f't{n_tasks}'][f'm{n_machines}']['cplex'].append(rpd(cplex, final[0].c_max[0]))
            results[u][f't{n_tasks}'][f'm{n_machines}']['fitness'].append(final[0].fitness)
            results[u][f't{n_tasks}'][f'm{n_machines}']['cm'].append(int(final[0].c_max[0]))
            results[u][f't{n_tasks}'][f'm{n_machines}']['time'].append(et - st)
            results[u][f't{n_tasks}'][f'm{n_machines}']['moves'].append(0)
            results[u][f't{n_tasks}'][f'm{n_machines}']['visit'].append(0)
            # Save raw file of instance results
            with open(f'Results/{u}-{n_tasks}-{n_machines}-{case}_{method}.txt', 'w') as file:
                for x, r in enumerate(final):
                    file.write(f'-------- Try {x} --------\n')
                    file.write('\t '.join(['%s = %s\n' % (k, v) for k, v in r.__dict__.items()]))
    print('Complete time: ', time() - total_time)
    # Save raw file of all results
    try:
        with open(f'Results/0-Raw_{method}.json', 'w') as file:
            json.dump(results, file)
    except:
        print('JSON not saved')
    try:
        with open(f'Results/0-Raw_{method}.txt', 'w') as file:
            file.write(json.dumps(results))
    except:
        print('TXT-JSON not saved')
    try:
        with open(f'Results/0-Raw_{method}.txt', 'w') as file:
            json.dumps(results)
    except:
        print('JSON-TXT not saved')

    print_results(results, name=f'0-dictResults_{method}.txt')