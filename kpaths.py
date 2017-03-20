# -*- coding: utf-8 -*-

import itertools


EPATH_LBL = 0   
V_LBL = 1
FORWARD = 0
BACKWARD = 1


def e2vpath(graph, epath, v=None):
    """
    Toma un camino en formato de secuencia de indices de arcos (epath) y 
    devuelve un camino en formato de secuencia de indices de vertices (vpath).
    Al pasar al formato vpath se pierde informacion de caminos que pasan por
    vertices conectados por múltiples arcos.     
    """    
    vpath=[]
    
    if len(epath) == 0:
        return []
    elif  v != None:
        vpath.append(v) 
        if v in ([graph.es[epath[-1]].source, graph.es[epath[-1]].target]):
            epath = epath[::-1]
    elif len(epath) > 1:
        if graph.es[epath[0]].source in ([graph.es[epath[1]].source
                                          , graph.es[epath[1]].target]):
            vpath.append(graph.es[epath[0]].target)
        else:
            vpath.append(graph.es[epath[0]].source)
    elif len(epath)==1:
        vpath.append(graph.es[epath[0]].source)

    for eid in epath:
        if graph.es[eid].source == vpath[-1]:
            vpath.append(graph.es[eid].target)
        elif graph.es[eid].target == vpath[-1]:
            vpath.append(graph.es[eid].source)
        else:
            raise IndexError("Not valid epath")
    return vpath       


def _v2epath(graph, vpath):
    """
    Toma un camino en formato de secuencia de indices de vertices (vpath) y 
    devuelve un camino en formato de secuencia de indices de arcos (epath).
    """   
    epath=[]
    if len(vpath)>1:
        for i in range(len(vpath)-1):
            try:
                eid = graph.get_eid(vpath[i], vpath[i+1])
                epath.append(eid)
            except:
                raise IndexError("Invalid vpath")
        return epath
    elif len(vpath)==0:
        return [] 
    else:
        raise IndexError("Invalid vpath")


def _epath_cost(graph, epath, weights):
    """
    Calcula el costo aditivo total de un camino en base a los costos de cada 
    uno de los arcos que lo componen. 'weights' puede ser el nombre de un 
    atributo de los arcos del grafo o una lista de enteros o flotantes. 
    """
    if epath == []:
        return float('inf')

    if isinstance(weights, str):
        w = graph.es[weights][:]
        
    elif isinstance(weights, list):
        if len(weights) != len(graph.es):
            raise IndexError("'weight' must have the same length than \
                              graph's edge sequence")
        w = weights[:]
        
    else:
        raise TypeError("'weights' must be a label or a list")
    
    wsum = 0
    for ei in epath:
        wsum += w[ei]
    return wsum


def sort_epaths(graph, epaths, weights='weight'):
    """
    Ordena una lista de caminos de menor a mayor en función de los costos 
    representados por 'weight'. El camino más corto de la lista queda en la 
    posición con índice 0.    
    """
    paths_w = [_epath_cost(graph, epath, weights) for epath in epaths]
    for i in range(len(epaths)):
        for j in range(len(epaths)-1-i):
            if paths_w[j] > paths_w[j+1]:
                temp = epaths[j]
                temp_w = paths_w[j]
                epaths[j] = epaths[j+1]
                paths_w[j] = paths_w[j+1]
                epaths[j+1] = temp
                paths_w[j+1] = temp_w    
    return None  


def _get_inclusion_lists(graph, v, to, incl_v=[], incl_e=[]):
    
    if v  not in range(len(graph.vs)) or to not in range(len(graph.vs)):
        raise IndexError("'v' or 'to' index out of range")
     
    if v == to:
        raise IndexError("'v' and 'to' can't be same")

    
    adj_v = graph.incident(v,3)
    adj_to = graph.incident(to,3)   
    
    # Clean vertices inclusion list creation
    if isinstance(incl_v, int):
        incl_v = [incl_v]
    
    if isinstance(incl_e, int):
        incl_e = [incl_e]

    clean_v = [vid for vid in incl_v if vid not in [v,to]]
    clean_v = ([vid for vid in clean_v if vid not in 
                    ([vid2 for eid in incl_e for vid2 in 
                        [graph.es[eid].source, graph.es[eid].target]])
                ])

    source_edges = [eid for eid in adj_v if eid in incl_e]    
    target_edges = [eid for eid in adj_to if eid in incl_e]
    clean_e = [[eid] for eid in incl_e if eid not in (source_edges 
                                                    + target_edges)]
           
    if len(source_edges) > 1 or len(target_edges) > 1:
        raise ValueError("More than one edge in inclusion list adjacent to v")  
   
    if source_edges and source_edges == target_edges:
        target_edges = []
        if clean_e:
            raise ValueError("Not single direct edge between v and to in\
                              inclusion list")  
            
    clean_e = ([ep for ep in ([source_edges] + clean_e + [target_edges]) 
                                if ep != []])
            
    for vid in range(len(graph.vs)):
        adj_v_i = [eid for eid in graph.incident(vid,3) if eid in ([ei[0] for ei in clean_e])]
        if len(adj_v_i) > 2:
            raise ValueError("More than two edges adjacent to the same vertex\
                              in inclusion list") 
            

    w0_continue = clean_e != []
    while(w0_continue):       
        i = 0
        while(i < len(clean_e)):
            el1e1_s = graph.es[clean_e[i][0]].source
            el1e1_t = graph.es[clean_e[i][0]].target
            el1e2_s = graph.es[clean_e[i][-1]].source
            el1e2_t = graph.es[clean_e[i][-1]].target
            j = i + 1
            w1_break = False
            
            while(j < len(clean_e)):
                el2e1_s = graph.es[clean_e[j][0]].source
                el2e1_t = graph.es[clean_e[j][0]].target
                el2e2_s = graph.es[clean_e[j][-1]].source
                el2e2_t = graph.es[clean_e[j][-1]].target
                
                if (el1e1_s == el2e1_s or el1e1_s == el2e1_t 
                        or el1e1_t == el2e1_s or el1e1_t == el2e1_t):
                    w1_break = True
                    el = clean_e[i][::-1] + clean_e[j][:]
                    clean_e = clean_e[:j] + clean_e[j+1:]
                    clean_e[i] = el
                    break
                
                elif (el1e1_s == el2e2_s or el1e1_s == el2e2_t 
                        or el1e1_t == el2e2_s or el1e1_t == el2e2_t):
                    w1_break = True
                    el = clean_e[j][:] + clean_e[i][:]
                    clean_e = clean_e[:j] + clean_e[j+1:]
                    clean_e[i] = el
                    break
                
                elif (el1e2_s == el2e1_s or el1e2_s == el2e1_t 
                        or el1e2_t == el2e1_s or el1e2_t == el2e1_t):
                    w1_break = True
                    el = clean_e[i][:] + clean_e[j][:]
                    clean_e = clean_e[:j] + clean_e[j+1:]
                    clean_e[i] = el
                    break
                
                elif (el1e2_s == el2e2_s or el1e2_s == el2e2_t 
                        or el1e2_t == el2e2_s or el1e2_t == el2e2_t):
                    w1_break = True
                    el = clean_e[i][:] + clean_e[j][::-1]
                    clean_e = clean_e[:j] + clean_e[j+1:]
                    clean_e[i] = el
                    break
                
                j += 1
            
            if w1_break:
                break
            
            i += 1
            if i == len(clean_e):
                w0_continue = False
 

    el_start = [el for el in clean_e if (el[0] in adj_v or el[-1] in adj_v)]
    el_end = [el for el in clean_e if (el[0] in adj_to or el[-1] in adj_to)]
    el_end = [el for el in el_end if el not in el_start]
    if el_end:
        el_end = el_end[0]
    if el_start:
        el_start = el_start[0]

    clean_e = [el for el in clean_e if el not in [el_start,el_end]]    
   
    return clean_e, el_start, el_end, clean_v


def _are_disjoint(graph, incl_v=[], incl_e=[], excl_v=[], excl_e=[]):
    
    if isinstance(incl_v, int):
        incl_v = [incl_v]
    
    if isinstance(incl_e, int):
        incl_e = [incl_e]

    if isinstance(excl_v, int):
        excl_v = [excl_v]

    if isinstance(excl_e, int):
        excl_e = [excl_e]


    if [vi for vi in incl_v if vi in excl_v]:
        return False

    if [ei for ei in incl_e if ei in excl_e]:
        return False
    
   

    for vi in excl_v:
        for ei in incl_e:
            if vi in [graph.es[ei].source, graph.es[ei].target]:
                return False
    
    return True


def _one_way_greedy(graph, v, to, incl_v=[], incl_e=[], excl_v=[]
                    , excl_e=[], avoid_v=[], avoid_e=[], weights='weight'):
    
    if isinstance(excl_v, int):
        excl_v = [excl_v]
    if isinstance(excl_e, int):
        excl_e = [excl_e]
    

    if v in excl_v or to in excl_v:
        return []
    
    if not _are_disjoint(graph, incl_v, incl_e, excl_v, excl_e):
        return []
    
    
        
    try:
        (epaths
         , start_epath
         , end_epath
         , vertices) = _get_inclusion_lists(graph, v, to, incl_v, incl_e)
    except:
        return []
    
    s_vid = v
    if start_epath:
        start_vpath = e2vpath(graph, start_epath)
        sides = [start_vpath[0], start_vpath[-1]]
        sides.remove(v)
        if to in sides:
            return start_epath
        s_vid = sides[0]
    

             

    labels = ([(V_LBL,vid) for vid in vertices] 
              + [(EPATH_LBL, eid) for eid in range(len(epaths))])

    epath = start_epath[:]    
    excl_v_p = []
    
    
    
    p=0
    times = len(labels)
    while p < times:
        p+=1
        min_ep = []
        min_w = float('inf')
        next_s = s_vid
        min_lbl = ""


        for label in labels:

            excl_v_p=[]
            for ep in epaths:
                excl_v_p += e2vpath(graph, ep)[1:-1]
            
            excl_v_p += (e2vpath(graph, epath) + [v, to]
                         + e2vpath(graph, start_epath)
                         + e2vpath(graph, end_epath))
            excl_v_p = list(set(excl_v_p))
            excl_v_p = [vid for vid in excl_v_p if vid != s_vid]
            
            if label[0] == V_LBL:
                t_vids = [label[1]]

                n_vids = t_vids[:] 

            else:
                incl_epath = epaths[label[1]]

                incl_vpath = e2vpath(graph, incl_epath)
                t_vids = [incl_vpath[0], incl_vpath[-1]]
                n_vids = t_vids[::-1]  


            for t_vid in t_vids:
                excl_e_p = excl_e + epath + start_epath + end_epath
                for incl_epath in epaths:
                    excl_e_p += incl_epath 
                
                cand_ep = get_shortest_paths_exclusion(graph, s_vid, t_vid
                                                        , excl_v_p + excl_v
                                                        , excl_e_p
                                                        , avoid_v, avoid_e
                                                        , weights)[0]
                

                cand_w = _epath_cost(graph, cand_ep, weights)
                if cand_w < min_w:
                    min_w = cand_w
                    min_ep = cand_ep[:]
                    next_s = n_vids[t_vids.index(t_vid)]
                    min_lbl = label

        if not min_ep:
            return []           

        epath += min_ep
        
        if min_lbl[0] == EPATH_LBL:
            lbl_vpath = e2vpath(graph, epaths[min_lbl[1]])
            min_vpath = e2vpath(graph, min_ep)
            if lbl_vpath[0] != min_vpath[-1]:
                epath += epaths[min_lbl[1]][::-1]    
            else:
                epath += epaths[min_lbl[1]]
               

        s_vid = next_s
        labels.remove(min_lbl)
     
    t_vid = to    
    if end_epath:
        end_vpath = e2vpath(graph, end_epath)
        t_vid = [end_vpath[0], end_vpath[-1]]
        t_vid.remove(to)
        t_vid = t_vid[0]
        
    
    excl_v_p = (e2vpath(graph, epath) + [v, to]
                + e2vpath(graph, start_epath)
                + e2vpath(graph, end_epath))
    excl_v_p = [vid for vid in excl_v_p if vid not in [s_vid, t_vid]]
    
    excl_e_p = excl_e + epath
    for incl_epath in epaths:
        excl_e_p += incl_epath 
    

    cand_ep = get_shortest_paths_exclusion(graph, s_vid, t_vid
                                          , excl_v_p + excl_v, excl_e_p
                                          , avoid_v, avoid_e, weights)[0]
    
    if not cand_ep:
        return []
    
    return epath + cand_ep + end_epath


def _two_way_greedy(graph, v, to, incl_v=[], incl_e=[], excl_v=[]
                    , excl_e=[], avoid_v=[], avoid_e=[]
                    , weights='weight'):
    
    p1 = _one_way_greedy(graph, v, to, incl_v, incl_e, excl_v, excl_e, avoid_v
                         , avoid_e, weights)
    
    p2 = _one_way_greedy(graph, to, v, incl_v, incl_e, excl_v, excl_e, avoid_v
                         , avoid_e, weights)
    
    if _epath_cost(graph, p1, weights) <= _epath_cost(graph, p2, weights):
        return p1
    else:
        return p2[::-1]


def _elabels_expansion(labels):
    
    if not labels:
        return []
    
    elif len(labels) == 1:
        if labels[0][0] == EPATH_LBL:
            return [[(EPATH_LBL, labels[0][1], FORWARD)]
                    ,[(EPATH_LBL, labels[0][1], BACKWARD)]]
        else:
            return [labels]
        
    elif labels[0][0] == EPATH_LBL:
        ret_n = _elabels_expansion(labels[1:])
        return ([[(EPATH_LBL, labels[0][1], FORWARD)] 
                   + ret_i for ret_i in ret_n] 
                + [[(EPATH_LBL, labels[0][1], BACKWARD)] 
                   + ret_i for ret_i in ret_n])
    else:
        ret_n = _elabels_expansion(labels[1:])
        return [[labels[0]] + ret_i for ret_i in ret_n]


def _get_labels_perms(labels): 
    perms = itertools.permutations(labels)    
    perms_exp = []
    for p in perms:
        labels_i =_elabels_expansion(list(p))
        perms_exp += labels_i
    return perms_exp


def _single_target_comb(graph, v, to, incl_v=[], incl_e=[], excl_v=[]
                    , excl_e=[], avoid_v=[], avoid_e=[], weights='weight'):
    """
    Computa el camino más corto entre el vertice 'v' y el vertice 'to' que 
    incluye a los arcos y vertices listados en las listas de inclusión 
    (incl_e,incl_v), excluyendo a los arcos y vertices listados en las listas 
    de exclusión (excl_e,excl_v) y tratando de evitar siempre que se pueda los 
    arcos y vertices listados en las listas de evitar (avoid_v,avoid_e). Un 
    camino que cumpla con las listas de inclusion y con las listas de exclusion 
    de forma simultanea puede no existir, en cuyo caso el metodo devuelve una 
    lista vacía. Si existe, se devuelve el camino más corto que cumple con las 
    restricciones en formato de lista de indices de arcos 'epath'.
    El argumento 'weights', que hace referencia a los pesos de los arcos, puede 
    ser una etiqueta correspondiente a un atributo de los arcos del grafo o una
    lista de pesos (enteros o flotantes).
    
    Este método garantiza el computo del camino más corto que cumple con las 
    restricciones de inclusión y exclusión siempre que el mismo exista. 
    
    """
    
    if isinstance(incl_v, int):
        incl_v = [incl_v]
        
    if isinstance(incl_e, int):
        incl_e = [incl_e]        
        
    # En caso de inconsistencias entre las listas se devuelve una lista vacía    
    if not _are_disjoint(graph, incl_v+[v,to], incl_e, excl_v, excl_e):
        return []
    
    if isinstance(excl_v, int):
        excl_v = [excl_v]
    
    if isinstance(excl_e, int):
        excl_e = [excl_e]

    # Se intenta generar las listas de inclusion diferenciando segmentos de
    # incio, de finalización e intermedios de vetrices. 
    # En caso de inconsistencias se devuelve una lista vacía    
    try:
        (  epaths
         , start_epath
         , end_epath
         , vertices) = _get_inclusion_lists(graph, v, to, incl_v, incl_e)
    except:
        return []
    
    
    # Teniendo en cuenta que el método _get_inclusion_lists devuelve una lista
    # con las secuencias de arcos que estan relacionados con el origen 'v' se
    # genera el siguiente vertice para la busqueda como el último del camino 
    # inicial. En el caso de que la lista de inclusion contenga un camino 
    # único entre 'v' y 'to' se devuelve ese camino. 
    start = v
    if start_epath:
        start_vpath = e2vpath(graph, start_epath)
        sides = [vi for vi in [start_vpath[0], start_vpath[-1]] if vi != v]
        if to in sides:
            return start_epath
        start = sides[0]
        
    
    # Se genera una lista de etiquetas, una por cada vertice y una por cada
    # camino intermedio a incluir
    labels = ([(V_LBL, vid) for vid in vertices] 
              + [(EPATH_LBL, eid) for eid in range(len(epaths))])

   
    # Si no hay elementos intermedios para incluir se busca un camino que 
    # contemple el camino de incio (start_epath) y el de finalización 
    # (end_epath)
    if not labels:
        t_vid = to
        if end_epath:
            end_vpath = e2vpath(graph, end_epath)
            t_vid = [end_vpath[0], end_vpath[-1]]
            t_vid.remove(to)
            t_vid = t_vid[0]
            
        excl_e_p = excl_e + start_epath + end_epath
        excl_v_p = (excl_v + [v,to] + e2vpath(graph, start_epath)
                    + e2vpath(graph, end_epath))
      
        excl_v_p = [vi for vi in excl_v_p if vi not in [start,t_vid]]        
            
        path_section = get_shortest_paths_exclusion(graph, start, t_vid, excl_v_p
                                                   , excl_e_p, avoid_v, avoid_e
                                                   , weights)[0]
        
        if not path_section:
            return []
        return start_epath + path_section + end_epath
        
    # Si existen elementos intermedios para incluir, se generan las 
    # permutaciones de las etiquetas y se realizan las busuqedas intermedias
    # los caminos más cortos que corresponden a cada una de las permutaciones
    # se almacenan en la lista 'candidates'. El camino más corto es el menor 
    # de los caminos de la lista 'candidates'.    
    candidates = []
    labels_perms = _get_labels_perms(labels)

    for labels_p in labels_perms:
        epath_isgood = True
        s = start
        candidate = []

        for label in labels_p:
            if label[0] == V_LBL:
                t_vid = label[1]
                
                
                excl_v_p=[]
                for ep in epaths:
                    excl_v_p += e2vpath(graph, ep)[1:-1]
                    
                excl_v_p += (excl_v + vertices + [start,to]
                             + e2vpath(graph, start_epath)
                             + e2vpath(graph, end_epath))
                excl_v_p = list(set(excl_v_p))
                excl_v_p = [vi for vi in excl_v_p if vi not in [s,t_vid]]  
                    
                excl_e_p = excl_e + candidate + start_epath + end_epath  
                for epath in epaths:
                    excl_e_p += epath
                                                                                                  
                path_section = get_shortest_paths_exclusion(graph, s, t_vid
                                                           , excl_v_p, excl_e_p
                                                           , avoid_v, avoid_e
                                                           , weights)[0]
                
                if not path_section:
                    epath_isgood = False
                    break
                
                candidate += path_section
                s = t_vid
                      
            else:
                eid = label[1]
                vpath = e2vpath(graph, epaths[eid])
                
                if label[2] == FORWARD:
                    vpath = vpath[::-1]
                t_vid = vpath[0]


                excl_v_p=[]
                for ep in epaths:
                    excl_v_p += e2vpath(graph, ep)[1:-1]
                    
                excl_v_p += (excl_v + vertices + [start,to]
                             + e2vpath(graph, start_epath)
                             + e2vpath(graph, end_epath))
                excl_v_p = list(set(excl_v_p))
                excl_v_p = [vi for vi in excl_v_p if vi not in [s,t_vid]]        
            
                excl_e_p = excl_e + candidate + start_epath + end_epath  
                for epath in epaths:
                    for ei in epath:
                        excl_e_p.append(ei)

                path_section = get_shortest_paths_exclusion(graph, s, t_vid
                                                           , excl_v_p, excl_e_p
                                                           , avoid_v, avoid_e
                                                           , weights)[0]

                if not path_section:
                    epath_isgood = False
                    break
                
                s = vpath[-1]
                candidate += path_section 
                
                if e2vpath(graph, epaths[eid])[0] != e2vpath(graph, candidate)[-1]:
                    candidate += epaths[eid][::-1]
                else:
                    candidate += epaths[eid]
        
        if epath_isgood:               
            for epath in epaths:
                excl_e_p += epath
        
            t_vid = to
            if end_epath:
                vpath =e2vpath(graph, end_epath)
                t_vid = [vpath[0], vpath[-1]]
                t_vid.remove(to)
                t_vid = t_vid[0]

                
                
                
            excl_v_p = (excl_v + vertices + [start,to]
                        + e2vpath(graph, start_epath)
                        + e2vpath(graph, end_epath))
            
            excl_v_p = [vi for vi in excl_v_p if vi not in [s,t_vid]]        
            
            excl_e_p = excl_e + candidate + start_epath + end_epath  

            path_section = get_shortest_paths_exclusion(graph, s, t_vid
                                                       , excl_v_p, excl_e_p
                                                       , avoid_v, avoid_e
                                                       , weights)[0]
            
            if path_section:
                candidate += path_section + end_epath    
                candidates.append(candidate)
  
    # Se ordenan los caminos de la lista y, en caso de que exista, se devuelve 
    # el más corto.  
    sort_epaths(graph, candidates, weights)
    if not candidates:        
        return []
    
    return start_epath + candidates[0]
   
   
def get_shortest_paths_exclusion(graph, v, to=None, excl_v=[], excl_e=[]
                                , avoid_v=[], avoid_e=[], weights='weight'):    
     
    """
    Computa el camino más corto entre el vertice 'v' y una conjunto de 
    destinos. Si 'to' es un índice, se computa un único camino entre 'v' y 
    'to'. Si 'to' es una lista de índices, se computan lso caminos entre 'v' y
    cada uno de los índices en la lista. Si 'to' es None, se computan los 
    caminos entre 'v' y cada uno de los vertices en el grafo. En todos los 
    casos se excluyen los vetices y arcos explicitados en las listas de 
    exclusión (excl_v, excl_e). Además se evitan, siempre que sea posible, los 
    vertices y arcos cuyos índices se encuentren en las listas de evitar
    (avoid_v, avoid_e).
    'weights' puede ser una etiqueta correspondiente a un atributo de los arcos
    del grafo o una lista de costos (enteros o flotantes).
    """ 

    if isinstance(excl_e,int):
        excl_e = [excl_e]
      
    if isinstance(excl_v,int):
        excl_v = [excl_v]

    if isinstance(avoid_e,int):
        avoid_e = [avoid_e]

    if isinstance(avoid_v,int):
        avoid_v = [avoid_v]

    if isinstance(weights, str):
        w = graph.es[weights][:]
    elif isinstance(weights, list):
        w = weights[:]
        if len(w) != len(graph.es):
            raise IndexError("Inconsistent length: 'weights list'")
    else:
        raise TypeError("Invalid type: 'weights'")

    g = graph.copy()
    g.es['oid'] = range(len(g.es))
      
    max_w = max(w)    
    w = [wi / float(max_w) for wi in w]           
    M = 2 * sum(w)
  
    for eid, e in enumerate(graph.es):
        if eid in avoid_e or e.source in avoid_v or e.target in avoid_v:
                w[eid] += float(M)  
                    
    to_delete = excl_e[:]
    for vi in excl_v:
        to_delete += g.incident(vi)
    g.delete_edges(list(set(to_delete)))    
    
    w = [w[e['oid']] for e in g.es]   
    
    
    epaths = g.get_shortest_paths(v, to, mode=3, output="epath"
                                  , weights=w)
    
    for epi, ep in enumerate(epaths):
        epn = [g.es[eid]['oid'] for eid in ep]
        epaths[epi] = epn
        
    return epaths    


def get_shortest_paths_greedy(graph, v, to=None, incl_v=[], incl_e=[]
                              , excl_v=[], excl_e=[], avoid_v=[], avoid_e=[]
                              , weights='weight'):
    if to == None:
        t_vids = range(len(graph.vs))
    elif isinstance(to, int):
        t_vids = [to]
    elif isinstance(to,list):
        t_vids = to[:]
    else:    
        raise TypeError("'to' must be None, an integer or a list")    
        
    epaths=[]    
    for t_vid in t_vids:
        epath = _two_way_greedy(graph, v, t_vid, incl_v, incl_e, excl_v
                                , excl_e, avoid_v, avoid_e, weights)
        epaths.append(epath)
    return epaths


def get_shortest_paths_comb(graph, v, to=None, incl_v=[], incl_e=[], excl_v=[]
                            ,excl_e=[], avoid_v=[], avoid_e=[]
                            , weights='weight'):
    if to == None:
        targets = range(len(graph.vs))
    elif isinstance(to, int):
        targets = [to]
    elif isinstance(to,list):
        targets = to[:]
    else:    
        raise TypeError("'to' must be None, an integer or a list")    
        
    epaths = []    
    for t_vid in targets:
        epath = _single_target_comb(graph, v, t_vid, incl_v, incl_e, excl_v
                                    , excl_e, avoid_v, avoid_e, weights)
        epaths.append(epath)
    return epaths


def k_shortest_paths(graph, v, to, incl_v=[], incl_e=[], excl_v=[], excl_e=[]
                     , avoid_v=[], avoid_e=[], weights='weight', K=1
                     , sp_func = get_shortest_paths_comb ):


    if isinstance(to, list):
        raise TypeError("'to' must be a vertex id")     


    if isinstance(incl_v, int):
        incl_v = [incl_v]

    if isinstance(incl_e, int):
        incl_e = [incl_e]

    if isinstance(excl_v, int):
        excl_v = [excl_v]

    if isinstance(excl_e, int):
        excl_e = [excl_e]
              
    curr_epaths = []
    cand_epaths = []
    
    epath = sp_func(graph, v, to, incl_v, incl_e, excl_v, excl_e, avoid_v
                    , avoid_e, weights)[0]

    curr_epaths.append(epath)
    vpath = e2vpath(graph, epath, v)
    
    for k in range(1,K):        
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
                    
            incl_v_p = incl_v[:]
            for vid in excl_v_p:
                if vid in incl_v_p:
                    incl_v_p.remove(vid)
            
            incl_e_p = incl_e[:]
            for eid in root_epath:
                if eid in incl_e_p:
                    incl_e_p.remove(eid)
    
            spur_epath = sp_func(graph, spur_node, to, incl_v_p
                                                 ,incl_e_p, excl_v + excl_v_p
                                                 , excl_e + excl_e_p, avoid_v
                                                 , avoid_e, weights)[0]

            if spur_epath:
                cand_epath = root_epath + spur_epath
                if (cand_epath not in cand_epaths 
                        and cand_epath not in curr_epaths):
                    cand_epaths.append(cand_epath)
                             
        if cand_epaths:
            sort_epaths(graph, cand_epaths, weights)
            curr_epaths.append(cand_epaths[0])
            cand_epaths = cand_epaths[1:]
        else:
            break
        
    curr_epaths = [c_ep for c_ep in curr_epaths if c_ep!=[]]
    return curr_epaths


