
import itertools


def compute_scenarios(entities, entities_av, N=2):
    '''
    Computes the scenarios and their occurrence probability
    :param entities:
    :param entities_av:
    :param epsilon:
    :return:
    '''
    p_base = 1.0
    for ent_i, ent in enumerate(entities):
        p_base *= entities_av[ent_i]

    scenarios = [[]]
    scenarios_p = [p_base]
    #    dif_p = 1 - p_base
    n = 0
    while n < N and n < len(entities):
        n += 1
        #if n != N:
        for comb in itertools.combinations(entities, n):
            scenarios.append(list(comb))
            p = p_base
            for ent in comb:
                ent_i = entities.index(ent)
                p *= ((1 - entities_av[ent_i])/entities_av[ent_i])
            scenarios_p.append(p)
    """
        else:
            for comb in itertools.combinations(entities, n):
                scenarios.append(list(comb))
                p = 1
                for ent in comb:
                    ent_i = entities.index(ent)
                    p *= (1 - entities_av[ent_i])
                scenarios_p.append((p))
    """

            #            dif_p -= p
            #            if dif_p <= epsilon:
            #                break

    return scenarios, scenarios_p


def compute_scenarios2(entities, entities_av, N=2):
    '''
    Computes the scenarios and their occurrence probability
    :param entities:acen
    :param entities_av:
    :param epsilon:
    :return:
    '''
    p_base = 1.0
    for ent_i, ent in enumerate(entities):
        p_base *= entities_av[ent_i]

    scenarios = [[]]
    scenarios_p = [p_base]
    #    dif_p = 1 - p_base
    n = 1
    while n < N and n < len(entities):
        n += 1
        for comb in itertools.combinations(entities, n):
            scenarios.append(list(comb))
            p = p_base
            for ent in comb:
                ent_i = entities.index(ent)
                p *= ((1 - entities_av[ent_i])/entities_av[ent_i])
            scenarios_p.append(p)
    scen1 = [[i] for i in entities]
    scen1_p = [(1 - a) for a in entities_av]
    for i, sce in enumerate(scen1):
        for j, comb in enumerate(scenarios):
            if sce[0] in comb:
                scen1_p[i]-=scenarios_p[j]

    scenarios = scen1 + scenarios
    scenarios_p = scen1_p + scenarios_p

    return scenarios, scenarios_p


def compute_nh_scenarios(entities, entities_av, N=2, threshold=0.995):
    '''
    Computes the scenarios and their occurrence probability
    :param entities:
    :param entities_av:
    :param epsilon:
    :return:
    '''
    p_base = 1.0
    for ent_i, ent in enumerate(entities):
        p_base *= entities_av[ent_i]

    scenarios = [[]]
    scenarios_p = [p_base]
    #    dif_p = 1 - p_base
    n = 0
    while n < N and n < len(entities):
        n += 1
        for comb in itertools.combinations(entities, n):
            scenarios.append(list(comb))
            p = p_base
            for ent in comb:
                ent_i = entities.index(ent)
                p *= ((1 - entities_av[ent_i])/entities_av[ent_i])
            scenarios_p.append(p)

    for comb in itertools.combinations([ent for i, ent in enumerate(entities) if entities_av[i] <= threshold], N+1):
        scenarios.append(list(comb))
        p = p_base
        for ent in comb:
            ent_i = entities.index(ent)
            p *= ((1 - entities_av[ent_i])/entities_av[ent_i])
        scenarios_p.append(p)

    return scenarios, scenarios_p


def scenarios2bin(scenarios):
    bin_s = []
    for scen in scenarios:
        bin = 0
        for ei in scen:
            bin |= 1 << ei
        bin_s.append(bin)
    return bin_s


def elist2bin(sceanrio):
    bina = 0b0
    for e_i in sceanrio:
        bina |= 1 << e_i
    return bina


def get_affection(epath, scenarios_bin):
    bin_epath = elist2bin(epath)
    affection=[]
    for scen_i, scen in enumerate(scenarios_bin):
        if bin_epath & scen:
            affection.append(scen_i)
    return affection


def get_fav(epath, scenarios_bin):
    bin_epath = elist2bin(epath)
    fav = []
    for scen_i, scen in enumerate(scenarios_bin):
        if not(bin_epath & scen):
            fav.append(scen_i)
    return fav


def get_service_av(epaths, scenarios_bin, scenarios_p):
    fav = []
    for epath in epaths:
        fav += get_fav(epath, scenarios_bin)
    p = 0.0
    for scen_i in set(fav):
        p += scenarios_p[scen_i]
    return p
