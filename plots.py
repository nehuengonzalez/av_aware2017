import matplotlib.pyplot as plt


def plot_random_costs(input_file, yscale='lin', ylabel=None, xlabel=None, output_file='ou.png'):

    infile = open(input_file, 'r')
    plt.figure(figsize=(12, 6), dpi=120)
    x=[]
    y=[]
    for i, row in enumerate(infile):
        if i == 0:
            labels = row.split(" ")
        else:
            splitt = row.split(" ")

            if i==1:
                xlabels = splitt[:]
                x=range(len(xlabels))
            else:
                for di, d in enumerate(splitt):
                    print(d.replace(',', '.'))
                    splitt[di] = float(d.replace(',', '.'))
                y.append(splitt[:])

    symbol=['ro-', 'bx-', 'g>-', 'y<-', 'kv-']
    for yi, data_y in enumerate(y):
        plt.plot(x, data_y, (symbol[yi]), label=labels[yi])

    if yscale:
        plt.yscale(yscale)
    plt.grid()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xticks(x, xlabels, rotation=45)
    plt.legend(loc=2)
    plt.xlim(0, len(xlabels)-1)
    #plt.savefig(output_file)
    plt.show()





