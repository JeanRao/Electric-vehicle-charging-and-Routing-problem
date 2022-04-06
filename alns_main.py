'''
Description: ALNS + EVRP
Author: BO Jianyuan
Date: 2022-02-15 13:38:21
LastEditors: BO Jianyuan
LastEditTime: 2022-02-24 19:43:57
'''

import argparse
import numpy as np
import numpy.random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from lxml import etree as LET

from evrp import *
from pathlib import Path

import sys
sys.path.append('./ALNS')
from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel


### draw and output solution ###
def save_output(YourName, evrp, suffix):
    '''Draw the EVRP instance and save the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file, 
            'initial' for random initialization
            and 'solution' for the final solution
    '''
    draw_evrp(YourName, evrp, suffix)
    generate_output(YourName, evrp, suffix)

### visualize EVRP ###
def create_graph(evrp):
    '''Create a directional graph from the EVRP instance
    Args:
        evrp::EVRP
            an EVRP object
    Returns:
        g::nx.DiGraph
            a directed graph
    '''
    g = nx.DiGraph(directed=True)
    g.add_node(evrp.depot.id, pos=(evrp.depot.x, evrp.depot.y), type=evrp.depot.type)
    for c in evrp.customers:
        g.add_node(c.id, pos=(c.x, c.y), type=c.type)
    for cs in evrp.CSs:
        g.add_node(cs.id, pos=(cs.x, cs.y), type=cs.type)
    return g

def draw_evrp(YourName, evrp, suffix):
    '''Draw the EVRP instance and the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file, 
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    g = create_graph(evrp)
    route = list(node.id for node in sum(evrp.route, []))
    edges = [(route[i], route[i+1]) for i in range(len(route) - 1) if route[i] != route[i+1]]
    g.add_edges_from(edges)
    colors = []
    for n in g.nodes:
        if g.nodes[n]['type'] == 0:
            colors.append('#0000FF')
        elif g.nodes[n]['type'] == 1:
            colors.append('#FF0000')
        else:
            colors.append('#00FF00')
    pos = nx.get_node_attributes(g, 'pos')
    fig, ax = plt.subplots(figsize=(24, 12))
    nx.draw(g, pos, node_color=colors, with_labels=True, ax=ax, 
            arrows=True, arrowstyle='-|>', arrowsize=12, 
            connectionstyle='arc3, rad = 0.025')

    plt.text(0, 6, YourName, fontsize=12)
    plt.text(0, 3, 'Instance: {}'.format(evrp.name), fontsize=12)
    plt.text(0, 0, 'Objective: {}'.format(evrp.objective()), fontsize=12)
    plt.savefig('{}_{}_{}.jpg'.format(YourName, evrp.name, suffix), dpi=300, bbox_inches='tight')
    
### generate output file for the solution ###
def generate_output(YourName, evrp, suffix):
    '''Generate output file (.txt) for the evrp solution, containing the instance name, the objective value, and the route
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file,
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    str_builder = ['{}\nInstance: {}\nObjective: {}\n'.format(YourName, evrp.name, evrp.objective())]
    for idx, r in enumerate(evrp.route):
        str_builder.append('Route {}:'.format(idx))
        j = 0
        for node in r:
            if node.type == 0:
                str_builder.append('depot {}'.format(node.id))
            elif node.type == 1:
                str_builder.append('customer {}'.format(node.id))
            elif node.type == 2:
                str_builder.append('station {} Charge ({})'.format(node.id, evrp.vehicles[idx].battery_charged[j]))
                j += 1
        str_builder.append('\n')
    with open('{}_{}_{}.txt'.format(YourName, evrp.name, suffix), 'w') as f:
        f.write('\n'.join(str_builder))

### Destroy operators ###
def destroy_worst(current, random_state):
    #worst destroy
    ''' Destroy operator sample (name of the function is free to change)
    Args:
        current::EVRP
            an EVRP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::EVRP
            the evrp object after destroying
    '''
    destroyed = current.copy()
    #choose a random vehicle
    #find and destroy the worst customer
    random_state.shuffle(destroyed.vehicles)
    v = destroyed.vehicles[0]
    depot_and_customers=[node for node in v.node_visited if (node.type==1 or node.type==0)]
    path_len=[]
    for i in range(len(depot_and_customers[1::-1])):
        dis=distance(depot_and_customers[i-1],depot_and_customers[i])+distance(depot_and_customers[i+1],depot_and_customers[i])
        path_len.append(dis)
    node_remove=depot_and_customers[path_len.index(max(path_len))+1]
    if node_remove in destroyed.customer_visited:
        destroyed.customer_visited.remove(node_remove)
        destroyed.customer_unvisited.append(node_remove)
        destroyed.vehicles[0].node_visited.remove(node_remove)
    return destroyed

### Repair operators ###
def repair_greedy(destroyed, random_state):
    ''' repair operator sample (name of the function is free to change)
    Args:
        destroyed::EVRP
            an EVRP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::EVRP
            the evrp object after repairing
    '''
    #get the destroyed node
    repaired=EVRP(parsed.name, parsed.depot, parsed.customers, parsed.CSs, parsed.vehicle)
    current_tour=copy.deepcopy(destroyed.customer_visited)
    for node_destroyed in destroyed.customer_unvisited:
    #find the shortest path
        path_dis=[]
        for i in range(len(current_tour)-1):
            dis=distance(current_tour[i],node_destroyed)+distance(node_destroyed,current_tour[i+1])
            path_dis.append(dis)
    #insert the destroyed node into the path, after current_route[path_dis.index(max(path_dis))]
        current_tour.insert(path_dis.index(min(path_dis))+1,node_destroyed)
    #rerun the split func
    repaired.split_route(current_tour)

    return repaired

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    parser.add_argument(dest='seed', type=int, help='seed')
    args = parser.parse_args()
    
    # instance file and random seed
    xml_file = args.data
    seed = int(args.seed)
    
    # load data and random seed
    parsed = Parser(xml_file)
    evrp = EVRP(parsed.name, parsed.depot, parsed.customers, parsed.CSs, parsed.vehicle)
    
    # construct random initialized solution
    evrp.random_initialize(seed)
    print("Initial solution objective is {}.".format(evrp.objective()))
    
    # visualize initial solution and gernate output file
    save_output('Ningzhen', evrp, 'initial')
    
    # ALNS
    random_state = rnd.RandomState(seed)
    alns = ALNS(random_state)
    # add destroy
    # You should add all your destroy and repair operators
    alns.add_destroy_operator(destroy_worst)
    # add repair
    alns.add_repair_operator(repair_greedy)
    
    # run ALNS
    # select cirterion
    criterion = HillClimbing()
    # assigning weights to methods
    omegas = [3,1,0,0.5]
    lambda_ =0.8
    result = alns.iterate(evrp, omegas, lambda_, criterion,
                          iterations=1000, collect_stats=True)

    # result
    solution = result.best_state
    objective = solution.objective()
    print('Best heuristic objective is {}.'.format(objective))
    
    # visualize final solution and gernate output file
    save_output('Ningzhen', solution, 'solution')
    