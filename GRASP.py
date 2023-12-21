from methods import *
from random import choice, seed
from os import listdir
from copy import deepcopy as dcopy
from time import time
from HC import HC_run
from SA import SA_run
import json
from math import log


seed(0)

def GRS_eval(c_values, remain, machines, data):
    # Initialize candidates: It will consist in elements of tuples: (tasks, machine, value)
    candidates = []
    # Check which task-machine increases less the processing time:
    for t in remain:
        # Create an array of increases for every machine
        diffs = [data[t][m] + c_values[m] for m in range(machines)]
        # Keep the minimum value of diffs
        best_val = min(diffs)
        best_m = diffs.index(best_val)
        candidates.append((t, best_m, best_val))

    return candidates

def GRS_Construction(machines, tasks, data, alpha=0.5):
    """Greedy Randomize Solution:"""
    # Generate an initial empty solution
    solution = [[] for _ in range(machines)]
    # Create a list of remaining tasks
    # remain = [t for t in range(tasks)]
    # Create a remaining list of tasks yet to be considered
    remain = [t for t in range(tasks)]
    remain = sample(remain, k=len(remain))  # Do a permutation
    # Create list of C values
    c_values = [0 for _ in range(machines)]
    # Create a candidate list
    candidates = GRS_eval(c_values, remain, machines, data)
    # print('--------------Initial------------')
    # print_log(solution, candidates, e, c_values)
    # While remain list is not empty, build the solution
    while remain:
        remain = sample(remain, k=len(remain))  # Do a permutation
        # print('--------------Iteration------------')
        # Find lower bound (lb) & upper bound (ub) as the min & max value of e
        lb = min(candidates, key=lambda value: value[2])[2]
        ub = max(candidates, key=lambda value: value[2])[2]
        # Create the restricted candidate list (rcl) w/ candidates based on lb & ub
        rcl = [c for c in candidates if c[2] <= lb + (alpha * (ub - lb))]
        # print('RCL', rcl)
        # Randomly choose a candidate
        selection = choice(rcl)
        # Add choice to the solution
        solution[selection[1]].append(selection[0])
        # Update the C value
        c_values[selection[1]] = selection[2]
        # Update remain list
        remain.remove(selection[0])
        # Update evaluation values
        candidates = GRS_eval(c_values, remain, machines, data)
        # print_log(solution, candidates, e, c_values)
    # print('--------------Final------------')
    # print_log(solution, candidates, e, c_values)
    return solution, c_values, candidates


if __name__ == '__main__':
    method = 'GRASP-HC'
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
    folder_path = '../Instances/Instances/'
    u_instances = listdir(folder_path)
    total_time = time()
    for u in u_instances:
        print('> Doing ', u)
        results[u] = dcopy(dict_tasks)
        instances = all_instances(folder_path + u)
        for i in instances:
            n_machines, n_tasks, case, cplex, data = get_instance(i, folder_path + u)
            print(f'>> On instance {i} (Case {case})')
            st = time()
            initial_rep, cs, cand = GRS_Construction(n_machines, n_tasks, data, 0.2)
            final = HC_run(data, n_machines, n_tasks, 5, initial_rep)
            et = time()
            resume = [f.fitness for f in final]
            best = resume.index(min(resume))
            print('RPD: ', rpd(cplex, final[best].c_max[0]))
            print('Time: ', et - st)
            # Store results
            results[u][f't{n_tasks}'][f'm{n_machines}']['cplex'].append(rpd(cplex, final[best].c_max[0]))
            results[u][f't{n_tasks}'][f'm{n_machines}']['fitness'].append(final[best].fitness)
            results[u][f't{n_tasks}'][f'm{n_machines}']['cm'].append(int(final[best].c_max[0]))
            results[u][f't{n_tasks}'][f'm{n_machines}']['time'].append(et - st)
            results[u][f't{n_tasks}'][f'm{n_machines}']['moves'].append(final[best].movements)
            results[u][f't{n_tasks}'][f'm{n_machines}']['visit'].append(final[best].neighbours)
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

    method = 'GRASP-SA'
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
    folder_path = '../Instances/Instances/'
    u_instances = listdir(folder_path)
    total_time = time()
    for u in u_instances:
        print('> Doing ', u)
        results[u] = dcopy(dict_tasks)
        instances = all_instances(folder_path + u)
        for i in instances:
            n_machines, n_tasks, case, cplex, data = get_instance(i, folder_path + u)
            print(f'>> On instance {i} (Case {case})')
            st = time()
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
            print("Configuraton:")
            print(f'MD: {max_diff} - mD: {min_diff} - MT: {max_temp} - mT: {min_temp} - beta: {beta}')
            initial_rep, cs, cand = GRS_Construction(n_machines, n_tasks, data, 0.2)
            final = SA_run(data, n_machines, n_tasks, max_temp, min_temp, max_count=max_count, trials=1, beta=beta, initial_rep=initial_rep)
            et = time()
            print('RPD: ', rpd(cplex, final[0].best['c_max'][0]))
            print('Time: ', et - st)
            results[u][f't{n_tasks}'][f'm{n_machines}']['cplex'].append(rpd(cplex, final[0].best['c_max'][0]))
            results[u][f't{n_tasks}'][f'm{n_machines}']['fitness'].append(final[0].best['fitness'])
            results[u][f't{n_tasks}'][f'm{n_machines}']['cm'].append(int(final[0].best['c_max'][0]))
            results[u][f't{n_tasks}'][f'm{n_machines}']['time'].append(et - st)
            results[u][f't{n_tasks}'][f'm{n_machines}']['moves'].append(final[0].best['movements'])
            results[u][f't{n_tasks}'][f'm{n_machines}']['visit'].append(final[0].best['neighbors'])
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