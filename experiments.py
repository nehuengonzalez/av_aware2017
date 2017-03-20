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

