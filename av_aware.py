from pulp import *
import igraph
import sys
sys.setrecursionlimit(10000)



"""
Este es el metodo de av_aware para partially disjoint paths sin restricción de disjunción

"""
def online_ra(graph, s, d, scenarios, scenarios_p, sla, weights=None, k_max=3, instance_name="NN"):
    if isinstance(weights, str):
        weight = graph.es[weights][:]
    elif isinstance(weights, list):
        weight = weights[:]
    else:
        weight = [1] * len(graph.es)

    assert isinstance(graph, igraph.Graph)

    prob = LpProblem('RA instance: %s' % instance_name, LpMinimize)

    x_combs = range(len(graph.es))
    xij = LpVariable.dicts('flow variables x(e)', x_combs, lowBound=0
                           , upBound=1, cat=LpInteger)

    # Flow variables Xij
    x_combs = []
    for k in range(k_max):
        for e_id, e in enumerate(graph.es):
            x_combs.append((k, e.source, e.target, e_id))
            x_combs.append((k, e.target, e.source, e_id))

    x = LpVariable.dicts('flow variables x(k,i,j,e)', x_combs, lowBound=0
                         , upBound=1, cat=LpInteger)
    P = LpVariable("total prob", lowBound=0, cat=LpContinuous)

    skg_combs = []
    for k in range(k_max):
        for g in range(len(scenarios)):
            skg_combs.append((k, g))
    skg = LpVariable.dicts('fail variables s(k,g)', skg_combs, lowBound=0
                           , upBound=1, cat=LpInteger)

    sg_combs = [g for g in range(len(scenarios))]
    sg = LpVariable.dicts('fail variables s(g)', sg_combs, lowBound=0
                          , upBound=1, cat=LpInteger)

    # Minimize sum of flow variables
    constr = ""
    for e_id, e in enumerate(graph.es):
        constr += ' + %f*xij[%d]' % (weight[e_id], e_id)
    prob += eval(constr)


    # Flow continuity constraint
    for k in range(k_max):
        for i in range(len(graph.vs)):
            constraint = ""
            for e_id, e in enumerate(graph.es):
                if e.source == i:
                    constraint += ' + x[(%d,%d,%d,%d)]' % (k, i, e.target, e_id)
                    constraint += ' - x[(%d,%d,%d,%d)]' % (k, e.target, i, e_id)
                elif e.target == i:
                    constraint += ' + x[(%d,%d,%d,%d)]' % (k, i, e.source, e_id)
                    constraint += ' - x[(%d,%d,%d,%d)]' % (k, e.source, i, e_id)
            if constraint:
                if i == s:
                    constraint += ' == 1'
                elif i == d:
                    constraint += ' == -1'
                else:
                    constraint += ' == 0'
                prob += eval(constraint)

    for k in range(k_max):
        for e_id, e in enumerate(graph.es):
            prob += xij[e_id] >= x[(k, e.source, e.target, e_id)] + x[(k, e.target, e.source, e_id)]

    for g, scen in enumerate(scenarios):
        for k in range(k_max):
            for e_id, e in enumerate(graph.es):
                if e_id in scen:
                    prob += (skg[(k, g)] - x[(k, e.source, e.target, e_id)]
                             - x[(k, e.target, e.source, e_id)] >= 0)

    for g, scen in enumerate(scenarios):
        constr = 'sg[%d]' % g
        for k in range(k_max):
            constr += ' - skg[(%d,%d)]' % (k, g)
        constr += ' >= %d' % (- k_max + 1)
        prob += eval(constr)
    constr = ''
    for g, scen in enumerate(scenarios):
        constr += ' +  %f - %f * sg[%d]' % (scenarios_p[g], scenarios_p[g], g)
    constr += ' -P == 0'
    prob += eval(constr)

    prob += P >= sla

    return prob




"""
Este es el metodo de av_aware para fully disjoint

"""

def online_ra_p(graph, s, d, scenarios, scenarios_p, sla, weights=None, k_max=2, instance_name="NN"):
    if isinstance(weights, str):
        weight = graph.es[weights][:]
    elif isinstance(weights, list):
        weight = weights[:]
    else:
        weight = [1] * len(graph.es)

    assert isinstance(graph, igraph.Graph)

    prob = LpProblem('RA instance: %s' % instance_name, LpMinimize)

    x_combs = range(len(graph.es))
    xij = LpVariable.dicts('flow variables x(e)', x_combs, lowBound=0
                           , upBound=1, cat=LpInteger)

    # Flow variables Xij
    x_combs = []
    for k in range(k_max):
        for e_id, e in enumerate(graph.es):
            x_combs.append((k, e.source, e.target, e_id))
            x_combs.append((k, e.target, e.source, e_id))

    x = LpVariable.dicts('flow variables x(k,i,j,e)', x_combs, lowBound=0
                         , upBound=1, cat=LpInteger)
    P = LpVariable("total prob", lowBound=0, cat=LpContinuous)

    skg_combs = []
    for k in range(k_max):
        for g in range(len(scenarios)):
            skg_combs.append((k, g))
    skg = LpVariable.dicts('fail variables s(k,g)', skg_combs, lowBound=0
                           , upBound=1, cat=LpInteger)

    sg_combs = [g for g in range(len(scenarios))]
    sg = LpVariable.dicts('fail variables s(g)', sg_combs, lowBound=0
                          , upBound=1, cat=LpInteger)

    # Minimize sum of flow variables
    constr = ""
    for e_id, e in enumerate(graph.es):
        constr += ' + %f*xij[%d]' % (weight[e_id], e_id)
    prob += eval(constr)


    # Flow continuity constraint
    for k in range(k_max):
        for i in range(len(graph.vs)):
            constraint = ""
            for e_id, e in enumerate(graph.es):
                if e.source == i:
                    constraint += ' + x[(%d,%d,%d,%d)]' % (k, i, e.target, e_id)
                    constraint += ' - x[(%d,%d,%d,%d)]' % (k, e.target, i, e_id)
                elif e.target == i:
                    constraint += ' + x[(%d,%d,%d,%d)]' % (k, i, e.source, e_id)
                    constraint += ' - x[(%d,%d,%d,%d)]' % (k, e.source, i, e_id)
            if constraint:
                if i == s:
                    constraint += ' == 1'
                elif i == d:
                    constraint += ' == -1'
                else:
                    constraint += ' == 0'
                prob += eval(constraint)

    for e_id, e in enumerate(graph.es):
        constraint = ""
        for k in range(k_max):
            constraint += " + x[(%d,%d,%d,%d)] " % (k, e.source, e.target, e_id)
            constraint += " + x[(%d,%d,%d,%d)] " % (k, e.target, e.source, e_id)
        constraint += " <= 1"
        prob += eval(constraint)

    for k in range(k_max):
        for e_id, e in enumerate(graph.es):
            prob += xij[e_id] >= x[(k, e.source, e.target, e_id)] + x[(k, e.target, e.source, e_id)]

    for g, scen in enumerate(scenarios):
        for k in range(k_max):
            for e_id, e in enumerate(graph.es):
                if e_id in scen:
                    prob += (skg[(k, g)] - x[(k, e.source, e.target, e_id)]
                             - x[(k, e.target, e.source, e_id)] >= 0)

    for g, scen in enumerate(scenarios):
        constr = 'sg[%d]' % g
        for k in range(k_max):
            constr += ' - skg[(%d,%d)]' % (k, g)
        constr += ' >= %d' % (- k_max + 1)
        prob += eval(constr)
    constr = ''
    for g, scen in enumerate(scenarios):
        constr += ' +  %f - %f * sg[%d]' % (scenarios_p[g], scenarios_p[g], g)
    constr += ' -P == 0'
    prob += eval(constr)

    prob += P >= sla

    return prob
