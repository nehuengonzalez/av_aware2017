import igraph
import kpaths
from math import log
from kpaths import e2vpath, get_shortest_paths_comb, sort_epaths, get_shortest_paths_exclusion
from scenarios import scenarios2bin, get_fav
import sympy
import functools
from itertools import combinations


def paths_to_expr(paths):
    '''
    Construye una expresión proposicional que caracteriza los caminos dados por "path".

    Construye una expresión proposicional Sypmy que caracteriza los caminos dados por la lista
    "paths". Cada uno de los componentes de un camino es un símbolo de la expresión.
    Los componentes no pueden ser números, porque a 0 y 1 se los interpreta los valores booleanos.
    '''

    if len(paths) == 0:
        res = sympy.sympify(False)
    else:
        #label_paths = map(functools.partial(map, lambda i : str(i)), paths)
        label_paths = [[str(i) for i in p] for p in paths]
        #and_paths = map(functools.partial(reduce, lambda x,y : x+" & "+y), label_paths)
        and_paths = [functools.reduce(lambda x,y : x+" & "+y, p) for p in label_paths]
        or_paths = functools.reduce(lambda x,y : x+" | "+y, and_paths)
        res = sympy.sympify(or_paths)

    return res


def factorization(expr, availabilities):
    '''
    Calcula la disponiblidad límite por factorización.

    Calcula la disponibilidad del sistema caracterizado por la expresión booleana "expr" por
    factorización, utilizando las disponibilidades límite de cada componente dadas
    por "availabilites". Cada componente debe tener su disponibilidad definida.
    '''

    atoms = expr.atoms()
    if len(atoms) > 0:
        t = atoms.pop()
        try:
            a = availabilities[str(t)]
        except:
            raise Exception("No existen disponibilidad definida para el término " + str(t) + ".")
        res = a * factorization(expr.subs(t, True), availabilities) + (1-a) * factorization(expr.subs(t, False), availabilities)
    else:
        res = int(expr == True)

    return res


def epath_cost(graph, epath, weights='weight'):
    if isinstance(weights, str):
        we = graph.es[weights]
    else:
        we = weights
    return sum([we[ei] for ei in epath])


def large_path_set(graph, v, to, weights='weight', depth=1):
    if not isinstance(graph, igraph.Graph):
        raise TypeError("graph should be an igraph.Graph instance")

    base = kpaths.get_shortest_paths_exclusion(graph, v, to, weights=weights)
    if base == [[]]:
        return base

    extra = True
    while extra != [[]] :
        exclusion = [ei for path in base for ei in path]
        extra = kpaths.get_shortest_paths_exclusion(graph, v, to, excl_e=exclusion
                                                    , weights=weights)
        if extra != [[]]:
            base += extra

    #print (base)
    epaths = base[:]

    for n in range(depth):
        base = epaths[:]
        for base_path in base:
            vpath = kpaths.e2vpath(graph, base_path, v)

            for ei, e in enumerate(base_path):
                s = vpath[ei]
                d = vpath[ei+1]

                sp = kpaths.get_shortest_paths_exclusion(graph, s, d, excl_e=base_path
                                                         , weights=weights)
                excl_e = base_path[:]
                while sp != [[]]:
                    spi = base_path[:ei] + sp[0] + base_path[ei+1:]
                    if spi and spi not in epaths:
                        epaths.append(spi)
                    excl_e += sp[0]
                    sp = kpaths.get_shortest_paths_exclusion(graph, s, d, excl_e=excl_e
                                                             , weights=weights)



    return epaths


def large_path_set2(graph, v, to, weights='weight'):
    if not isinstance(graph, igraph.Graph):
        raise TypeError("graph should be an igraph.Graph instance")
    sp = kpaths.get_shortest_paths_exclusion(graph, v, to, weights=weights)
    if sp == [[]]:
        return sp
    sp = sp[0]

    base = kpaths.k_shortest_paths(graph, v, to, weights=weights, K=10)
    epaths = base
    exclusion = []
    while sp != []:
        exclusion += sp
        extra = kpaths.k_shortest_paths(graph, v, to, excl_e=exclusion, weights=weights, K=10)
        if extra != []:
            sp = extra[0]
        else:
            sp = []
        for ep in extra:
            if ep not in epaths:
                epaths.append(ep)
    return epaths


def get_av(graph, epath, availability='av'):
    p = 1
    for ei in epath:
        p *= graph.es[availability][ei]
    return p


def simple_path_finding(graph, v, to, av_obj, weights='weight', availability='a'):
    if not isinstance(graph, igraph.Graph):
        raise TypeError("graph should be an igraph.Graph instance")

    graph.es['log_av'] = [-log(ai) for ai in graph.es[availability]]
    sp_av = graph.get_shortest_paths(v, to, weights='log_av', output='epath')[0]

    av = 1
    for ei in sp_av:
        av *= graph.es[availability][ei]
    if av < av_obj:
        return []


    curr_epaths = []
    cand_epaths = []

    epath = graph.get_shortest_paths(v, to, weights, output='epath')[0]
    if get_av(graph, epath, availability) >= av_obj:
        return epath

    curr_epaths.append(epath)
    k = 0
    while True:
        k += 1
        vpath = e2vpath(graph, curr_epaths[k - 1], v)

        for i in range(len(vpath)-1):
            excl_e_p = []
            excl_v_p = []

            spur_node = vpath[i]
            root_epath = curr_epaths[k-1][:i]

            for curr_epath in curr_epaths:
                if len(curr_epath) > len(root_epath):
                    if root_epath == curr_epath[:len(root_epath)]:
                        eid = curr_epath[len(root_epath)]
                        if eid not in excl_e_p:
                            excl_e_p.append(eid)

            for eid in root_epath:
                excl_e_p.append(eid)

            root_vpath = e2vpath(graph, root_epath, v)

            for vid in root_vpath[:-1]:
                excl_v_p.append(vid)

            incl_v_p = []
            for vid in excl_v_p:
                if vid in incl_v_p:
                    incl_v_p.remove(vid)

            incl_e_p = []
            for eid in root_epath:
                if eid in incl_e_p:
                    incl_e_p.remove(eid)

            spur_epath = get_shortest_paths_comb(graph, spur_node, to, incl_v_p
                                                 , incl_e_p, excl_v_p, excl_e_p
                                                 , [], [], weights)[0]

            if spur_epath:
                cand_epath = root_epath + spur_epath
                if (cand_epath not in cand_epaths
                    and cand_epath not in curr_epaths):
                    cand_epaths.append(cand_epath)

        if cand_epaths:
            sort_epaths(graph, cand_epaths, weights)
            if get_av(graph, cand_epaths[0], availability) >= av_obj:
                return cand_epaths[0]
            curr_epaths.append(cand_epaths[0])
            cand_epaths = cand_epaths[1:]
        else:
            return []


def pair_partially_disjoint(graph, v, to, av_obj, weights='weight', availability='a', depth=1, superp=False):
    path_set = large_path_set(graph, v, to, weights=weights, depth=depth)
    best_pair = []
    best_cost = sum(graph.es[weights])
    best_av = 0
    i=0
    for comb in combinations(path_set, 2):
        #print(i)
        i+=1
        av = factor(list(comb), graph.es[availability])
        if av >= av_obj:
            if superp:
                cost = epath_cost(graph, list(set(comb[0]+comb[1])))
            else:
                cost = epath_cost(graph, list(comb[0]+comb[1]))

            if cost < best_cost or (cost == best_cost and av > best_av):
                best_pair = list(comb)
                best_cost = cost
                best_av = av
    return best_pair


def pair_partially_disjoint_scens(graph, v, to, av_obj, scen, scen_p, weights='weight', availability='a', depth=1
                                  , superp=False):
    path_set = large_path_set(graph, v, to, weights=weights, depth=depth)

    paths_surv = get_paths_surv(path_set, scen)


    best_pair = []
    best_cost = sum(graph.es[weights])
    best_av = 0
    i=0
    for comb in combinations(range(len(path_set)), 2):
        #print(i)
        i+=1

        av = get_service_av(list(comb), paths_surv, scen_p)

        if av >= av_obj:
            if superp:
                cost = epath_cost(graph, list(set(path_set[comb[0]]+path_set[comb[1]])))
            else:
                cost = epath_cost(graph, list(path_set[comb[0]]+path_set[comb[1]]))

            if cost < best_cost or (cost == best_cost and av > best_av):
                best_pair = list(comb)
                best_cost = cost
                best_av = av
    return [path_set[i] for i in best_pair]


def av_aware_route(graph, v, to ,av_obj, weights='weight', availability='a', depth=1):
    p = simple_path_finding(graph, v, to, av_obj, weights=weights, availability=availability)
    if not p:
        p = pair_partially_disjoint(graph, v, to, av_obj, weights=weights, availability='a', depth=depth)
    return p


def av_aware_route_scen(graph, v, to, av_obj, scen, scen_p, weights='weight', availability='a', depth=1):
    p = simple_path_finding(graph, v, to, av_obj, weights=weights, availability=availability)
    if not p:
        p = pair_partially_disjoint_scens(graph, v, to, av_obj, scen, scen_p, weights='weight', availability='a', depth=depth)
    return p


def are_disjoint(epath1, epath2):
    if [ei for ei in epath1 if ei in epath2]:
        return False
    return True


def factor(epaths, av):
    paths = []
    for epath in epaths:
        paths.append(["e%d"%ei for ei in epath])
    expr = paths_to_expr(paths)
    avs = {}
    for ei in [eid for p in epaths for eid in p]:
        avs["e%d" % ei] = av[ei]

    return factorization(expr, avs)

########################################################

def get_paths_surv(epaths, scen):
    bin_s = scenarios2bin(scen)
    surv=[]
    for epath in epaths:
        surv.append(get_fav(epath, bin_s))
    return surv


def get_service_av(service, epaths_surv, scen_p):
    scens = []
    for ep_id in service:
        scens += epaths_surv[ep_id]
    scens = list(set(scens))

    return sum([scen_p[i] for i in scens])






