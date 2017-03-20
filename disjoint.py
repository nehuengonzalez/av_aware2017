from pulp import *
import igraph
import sys
sys.setrecursionlimit(10000)



def online_ra(graph, sour, d, weights=None, instance_name="NN"):
    if isinstance(weights, str):
        weight = graph.es[weights][:]
    elif isinstance(weights, list):
        weight = weights[:]
    else:
        weight = [1] * len(graph.es)

    assert isinstance(graph, igraph.Graph)

    prob = LpProblem('Disjoint RA instance: %s' % instance_name, LpMinimize)

    s_combs = range(len(graph.es))
    s = LpVariable.dicts('jointness variables s(e)', s_combs, lowBound=0
                           , upBound=1, cat=LpInteger)

    # Flow variables Xij
    x_combs = []


    for e_id, e in enumerate(graph.es):
        x_combs.append((e.source, e.target, e_id))
        x_combs.append((e.target, e.source, e_id))

    x = LpVariable.dicts('flow variables x(i,j,e)', x_combs, lowBound=0
                         , upBound=1, cat=LpInteger)

    alfa = 20.0 * len(graph.es)
    # Minimize sum of flow variables
    constr = ""
    for e_id, e in enumerate(graph.es):
        constr += ' + %f*s[%d]' % (alfa, e_id)
        constr += ' + x[(%d,%d,%d)] + x[(%d,%d,%d)]' % (e.source, e.target
                                                        , e_id, e.target
                                                        , e.source, e_id)
    prob += eval(constr)


    # Flow continuity constraint

    for i in range(len(graph.vs)):
        constraint = ""
        for e_id, e in enumerate(graph.es):
            if e.source == i:
                constraint += ' + x[(%d,%d,%d)]' % (i, e.target, e_id)
                constraint += ' - x[(%d,%d,%d)]' % (e.target, i, e_id)
            elif e.target == i:
                constraint += ' + x[(%d,%d,%d)]' % (i, e.source, e_id)
                constraint += ' - x[(%d,%d,%d)]' % (e.source, i, e_id)
        if constraint:
            if i == sour:
                constraint += ' == %d' % 2
            elif i == d:
                constraint += ' == -%d' % 2
            else:
                constraint += ' == 0'
            prob += eval(constraint)

    for e_id, e in enumerate(graph.es):
        prob += s[e_id] - x[(e.source, e.target, e_id)] - x[(e.target, e.source, e_id)] >= -1

    return prob
