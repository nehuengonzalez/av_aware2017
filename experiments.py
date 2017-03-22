import av_aware
import paths
import scenarios
import igraph
import random
import timeit


def run1(g, s, d, scen, scen_p, av_obj ):
    start = timeit.default_timer()
    p = av_aware.online_ra(g, s, d, scen, scen_p, av_obj, 'weight')
    p.solve()
    stop = timeit.default_timer()
    print (stop - start)
    return p

def exp1(network_name, sla, s, d):
    cplx = pulp.solvers.CPLEX()
    g = igraph.read(network_name)
    g.es[COST] = [1.0]*len(g.es)
    cost = []
    comp_time = []


    for i in range(30):
        random_av(g, 2, 0.5, 8, 8)
        start = timeit.default_timer()
        scen, scen_p = scenarios.compute_alfa_scenarios(list(range(len(g.es))), g.es[AV], sla, 0.1)
        p = av_aware.online_ra(g, s, d, scen, scen_p, sla)
        p.solve(cplx)
        stop = timeit.default_timer()

        if p.status == 1:
            comp_time.append( stop - start)
            cost.append(0)
            for var in p.variables():
                if var.varValue > 0.5:
                    if var.name[:len(flow_var_name)] == flow_var_name:
                        e_id = int(var.name[var.name.index(")_") + 2 :])
                        cost[-1] += (g.es[e_id][COST])


    print(comp_time)
    print(cost)
    print(sum(comp_time) / len(comp_time))
    print(sum(cost) / len(cost))
