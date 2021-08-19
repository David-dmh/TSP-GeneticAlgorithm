import os
import copy
import time
import datetime
import numpy as np
import pandas as pd
import more_itertools as mi
import heapq
from collections import Counter
from operator import itemgetter 

os.chdir("~/PB1/")

def random_vectors(amt):
    """ Used to initialise unique random vectors. 'amt' must be < 720 to not 
    exhaust all permutations for 6 items.
    Args:
        amt (int): Number of vectors desired.
    Returns:
        vecs (list): List of specified vectors as permutations.
    """
    while True:
        # create permutations of length 6
        vecs = [list(np.random.permutation(6)) for i in range(amt)]
        # convert to hashable type to check for dupl
        as_tup_set = set([tuple(vec) for vec in vecs])
        len_lists = len(vecs)
        len_tup_set = len(as_tup_set)
        if len_lists == len_tup_set: # end when obtain valid permutations
            break
        
    return vecs


def tsp_distance(vector):
    """ dists for cities 0 - 5, e.g [0, 1, 2, 3, 4, 5] represents
    0 to 1 to 2 to 3 to 4 to 5 and back to 0. e.g. lookup_table[0, 1] is 
    dist for going FROM city 0 TO city 1.
    Args:
        vector (list): Vector for which distance should be calculated.
    Returns:
        distance (int): TSP distance travelled for given vector.
    """
    distance = 0
    # table containing distances
    lookup_table = np.array([[0, 32, 39, 42, 29, 35],
                             [32, 0, 36, 27, 41, 25],
                             [39, 36, 0, 28, 33, 40],
                             [42, 27, 28, 0, 27, 38],
                             [29, 41, 33, 27, 0, 26],
                             [35, 25, 40, 38, 26, 0]])
    # window the vector for easy pairwise computation of TSP trips
    # sample output: [(1, 2), (2, 3), (3, 4), (4, 0), (0, 5)]
    combs = [*mi.pairwise(vector)] 
    combs.append((vector[-1], vector[0]))
    for coord in combs: # sum trip distances
        distance += lookup_table[coord[0], coord[1]]
    
    return distance


def best_vector(vectors):
    """
    Args:
        vectors (list): Candidate vectors for determining best distance.
    Returns:
        best (list): Vector with lowest distance.
    """
    dists = [*map(tsp_distance, vectors)] # iteratively apply tsp_distance
    best_ind = dists.index(min(dists)) # get index of best dist
    best = vectors[best_ind] # subset to get corresponding best vector
    
    return best


def select(t_pop):
    """ Tournament selection procedure with k=3 performed three times to 
    return 3 pairs for reproduction in the crossover process. Selection is 
    proportional to vector fitness values (low dist is high fitness).
    Args: 
        t_pop (list): Population at iteration t.
    Returns: 
        mating_pairs (list): List of 3 tuples, each representing pairs to mate 
        for input into the crossover/mutate process.
    """
    mating_pairs = []
    t_pop_c = copy.deepcopy(t_pop)
    for i in range(3):
         # get distances representing fitnesses
        fitnesses = [tsp_distance(vec) for vec in t_pop_c]
        total = sum(fitnesses)
        # create a probability distribution of fitnesses
        p_distribution = [f/total for f in fitnesses] 
        # probabilistically select 3 contestants w/o replacement
        contestants = np.random.choice(fitnesses, 
                                       3, 
                                       p=p_distribution, 
                                       replace=False)
        # determine smallest 2 distances
        p_a, p_b = heapq.nsmallest(2, contestants)
        # get indices based on above values
        p_a_loc, p_b_loc = fitnesses.index(p_a), fitnesses.index(p_b)
        # two selected vectors reprenting a mating pair
        mating_pairs.append((t_pop_c[p_a_loc], t_pop_c[p_b_loc]))
        # new candidate population should exclude last selected mating pair
        indices = p_a_loc, p_b_loc
        t_pop_c = [i for j, i in enumerate(t_pop_c) if j not in indices]

    return mating_pairs


def crossover_mutate(mating_pairs):
    """
    Args: 
        mating_pairs (list): Parent pairs to produce offspring in a 2-point
        crossover process for permutations.
    Returns: 
        (offsprings, cutpoints_storage, mutated) (tuple): 3 Offspring vectors 
        having undergone crossover and mutation. Also returns cutpoints and 
        mutated offspring for printout purposes.
    """
    # 1. Crossover
    offsprings = []
    cutpoints_storage = []
    # generate one offspring per mating pair
    for pair in mating_pairs:
        # pair[0] (first parent) is used for cutpoints
        parent_1, parent_2 = pair[0], pair[1]
        # get indices of cutpointns
        cutpoints = sorted(np.random.choice([*range(1, len(pair[0]))], 
                                            2, 
                                            replace=False))
        cutpoints_storage.append(cutpoints) # add cutpoint for printout
        # subset for beginning, middle and end portions (2-point crossover)
        beginning = parent_1[0:cutpoints[0]]
        end = parent_1[cutpoints[1]:]
        # middle: between cutpoints but in order they appear in parent 2
        middle_cands = parent_1[cutpoints[0]:cutpoints[1]]
        middle = []
        for i in parent_2:
            if i in middle_cands:
                middle.append(i)
            
        offspring = beginning + middle + end
        offsprings.append(offspring)
    
    # 2. Mutate
    to_mutate = np.random.randint(3) # randomly select index
    mutated = offsprings.pop(to_mutate) # get and remove offspring
    # randomly swop two vector values as mutation
    inds = sorted(np.random.choice([*range(6)], 2, replace=False))
    temp = mutated[inds[0]]
    mutated[inds[0]] = mutated[inds[1]]
    mutated[inds[1]] = temp
    # add mutated offspring
    offsprings.append(mutated)
    
    return (offsprings, cutpoints_storage, mutated)
    

def replace(t_pop):
    """" Elitism replacement procedure for selection of best 8 vectors amongst 
    combined parent-offspring population.
    Args: 
        t_pop (list): Population at iteration t.
    Returns: 
        remaining_pop (list): Remaining population of size 8 representing the 
        elite group.
    """
    # get distances
    dists = [*map(tsp_distance, t_pop)]
    # find best 8 while preserving order (unsorted)
    # use counter to check which elements to include in elite_dists
    c = Counter(heapq.nsmallest(8, dists))
    elite_dists = []
    for dist in dists:
        if c.get(dist, 0):
            elite_dists.append(dist)
            c[dist] -= 1
    
    # check inds to keep based on where elite dists in dists
    keep_inds = list(np.where(np.isin(dists, elite_dists))[0])
    # use itemgetter to subset t_pop according to indices to keep
    remaining_pop = list(itemgetter(*keep_inds)(t_pop))

    return remaining_pop


def update_incumbent(incumbent, t_pop, COUNTER):
    """
    Args: 
        incumbent (list): Current incumbent.
        t_pop (list): Population at iteration t.
        COUNTER (int): Counter for tracking stop condition.
    Returns: 
        (incumbent, COUNTER) (tuple): New incumbent and updated counter.
    """
    t_pop_best = best_vector(t_pop)
    # check if t_pop_best offers improvement, adjust counter accordingly
    if tsp_distance(t_pop_best) < tsp_distance(incumbent):       
        incumbent = t_pop_best # alter incumbent
        COUNTER = 0 # reset counter
        
        return (incumbent, COUNTER)
    
    COUNTER += 1 # increment counter
        
    return (incumbent, COUNTER)


def fitnesses_and_mean(t_pop):
    """ Helper function for printout purposes.
    Args:
        t_pop (list): Population at iteration t.
    Returns:
        (fitnesses, mean_fitness) (tuple): Computes tsp_distances and mean 
        distance representing fitnesses and mean fitness respectively.
    """
    fitnesses = [tsp_distance(vec) for vec in t_pop]
    mean_fitness = np.mean(fitnesses)
    
    return (fitnesses, mean_fitness)


def genetic_algorithm(initial_pop):
    """ 
    Args: 
        initial_pop (list): Initial population of city permutations.
    Returns: 
        incumbent (list): Best solution in obtained in final generation.
    """   
    t = -1 
    ts = []
    pops = []
    pop_fits = []
    pop_mpairs = []
    cp_locs = []
    mutated_offsprings = []
    pop_fits_mean = []
    COUNTER = 0
    
    t_pop = initial_pop
    incumbent = best_vector(t_pop)
    while True:
        t_mating_pairs = select(t_pop)
        t_offsprings, \
        t_cutpoints, \
        t_mutated_offspring = crossover_mutate(t_mating_pairs)
        
        # capture values for prinouts
        t_pop += t_offsprings # add offsprings to pop
        cp_locs.append(t_cutpoints) # add cutpoint locations
        mutated_offsprings.append(t_mutated_offspring) # add mutated offspring
        pops.append(t_pop) # add entire pop
        t_pop_fits, t_pop_mean = fitnesses_and_mean(t_pop)
        pop_fits.append(t_pop_fits) # add pop fitnesses
        pop_fits_mean.append(t_pop_mean) # add mean pop fitness
        
        t_pop = replace(t_pop)
        incumbent, COUNTER = update_incumbent(incumbent, t_pop, COUNTER)
        
        # more captures
        t += 1
        ts.append(t)
        pop_mpairs.append(t_mating_pairs)
        
        if COUNTER == 8: # end after 8 generations have no improvement
            print("Incumbent:\n", incumbent)
            print("Distance:\n", tsp_distance(incumbent))
            record_df = pd.DataFrame.from_dict({
                "Generation": ts, 
                "Entire population": pops, 
                "Entire population fitness values": pop_fits, 
                "Parents selected for crossover": pop_mpairs, 
                "Cutpoint location": cp_locs, 
                "Mutated offspring": mutated_offsprings, 
                "Mean population fitness": pop_fits_mean
                })
            now = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
            time.sleep(1)
            # save dataframe to working dir for inspection
            record_df.to_csv("{time}.csv".format(time=now))
            
            return (incumbent, record_df)


if __name__ == "__main__":   
    initial_pop = random_vectors(8)
    incumbent, record_df = genetic_algorithm(initial_pop)
