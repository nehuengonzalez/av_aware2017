from decimal import *
from math import factorial


def ncr(n, r):
    f1 = Decimal(factorial(n))
    f2 = Decimal(factorial(r))
    f3 = Decimal(factorial(n-r))
    return f1 / f2 / f3



def av(N,a,n,l):
    suma = Decimal(0.0)
    for r in range(n+1):
        suma += ncr(N-l, r) * Decimal(a)**Decimal(N-r) * Decimal(1-a)**Decimal(r)
    return suma

def av_par(N,a,n,l):
    return Decimal(a)**Decimal(N) + Decimal(N) * Decimal(a)**Decimal(N-1) * Decimal(1-a) + (ncr(N, 2) - Decimal(l)**Decimal(2)) * Decimal(a)**Decimal(N-2) * Decimal(1-a)**Decimal(2)

############VIEJO##############3
def prop1nm(N, n, m):
    den=Decimal(0.0)
    num = ncr(N-1, m-1)

    for r in range(1, N-n+1):
        den += ncr(N, n+r)

    return num/den


def e_abs(N, n, A):
    error = Decimal(0.0)
    for m in range(n+1, N+1):
        error += ncr(N, m) * Decimal(A**(N - m)) * Decimal((1 - A)**m)
    return error


def e_rel(N, n, A):
    ref = Decimal(0.0)
    for r in range(1, n+1):
        ref += ncr(N, r) * Decimal(A**(N - r)) * Decimal((1 - A)**r)
    return e_abs(N, n, A) / ref


def Prop(N, n, l):
    tot = Decimal(0.0)
    for r in range(1,N-n+1):
        tot += ncr(N, n+r)

    fav = Decimal(0.0)
    for r in range(N-n-l):
        fav += ncr(N-l, n+r)

    return (tot-fav)/tot


def aporte(N,a):
    pr = []
    freq = []
    pr_tot=[]
    acu = 0
    pr_acu=[]
    for r in range(0,N+1):
        pr.append(Decimal(a**(N-r)) * Decimal((1-a)**r))
        freq.append(ncr(N,r))
        acu += pr[-1]*freq[-1]
        pr_tot.append(pr[-1]*freq[-1])
        pr_acu.append(acu)
    return pr, freq, pr_tot, pr_acu

def plot_acus(lista, labels):
    import matplotlib.pyplot as plt

    symbol=['ro-', 'bx-', 'g>-', 'y<-', 'kv-', 'm+-', 'cs-']
    for yi, data_y in enumerate(lista):
        x = range(len(data_y))
        plt.plot(x, data_y, (symbol[yi]), label=labels[yi])
    plt.ylim(-0.05, 1.05)
    plt.grid(True,which="both",ls="-")
    plt.legend(loc=4)
    plt.xscale('log')
    #xlabels= [str(xi) for xi in x]
    #plt.xticks(x, xlabels)
    plt.show()

def plot_acus2(N,avs):
    y = []
    labels = [str(av) for av in avs]
    for a in avs:
        pr, freq, pr_tot, pr_acu = aporte(N, a)
        y.append(pr_acu[:])
    plot_acus(y, labels)

def plot_acus3(Ns,a):
    y = []
    labels = [str(Nss) for Nss in Ns]
    for N in Ns:
        pr, freq, pr_tot, pr_acu = aporte(N, a)
        y.append(pr_acu[:])
    plot_acus(y, labels)

def plot_pis(N,avs):
    y = []
    labels = [str(av) for av in avs]
    for a in avs:
        pr, freq, pr_tot, pr_acu = aporte(N, a)
        y.append(pr[:])

    plot_acus(y, labels)

