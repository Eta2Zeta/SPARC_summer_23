import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_file(filename):
    df = pd.read_csv(filename, delim_whitespace=True, header=None, 
                     names=['n', 'r', 'phi', 'z', 'vr', 'vphi', 'vz', 'mass', 'charge'])
    df = df.drop(df.index[0])
    df = df.astype({
        'n': 'int',
        'r': 'float',
        'phi': 'float',
        'z': 'float',
        'vr': 'float',
        'vphi': 'float',
        'vz': 'float',
        'mass': 'float',
        'charge': 'int'
    })
    return df



def rzphi(vr,vphi,vz,vtot,Nmrk):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vr, vphi, vz, color='b')

    u = np.linspace(0, 2 * np.pi, Nmrk)
    v = np.linspace(0, np.pi, Nmrk)
    xs = vtot * np.outer(np.cos(u), np.sin(v))
    ys = vtot * np.outer(np.sin(u), np.sin(v))
    zs = vtot * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(xs, ys, zs, color='r', alpha=0.1, linewidth = 0.2)

    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0

    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set the aspect ratio of the plot to 1:1:1
    ax.set_box_aspect([1,1,1])

    # Add labels to the axes
    ax.set_xlabel('R')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Z')

    plt.show()


def plot_histogram(data, xlabel='', title='', bins=50, rwidth=1, color='c', pdf=None, graph_label = None):
    plt.figure(figsize=(8.,6.))
    plt.hist(data, bins=bins, rwidth=rwidth, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('')
    plt.tight_layout(pad=2)
    if pdf is not None:
        pdf.savefig()
    plt.show()



# Plots based on data input types
def group_go_data_plot():
    data = pd.read_csv('group_go_7000_vdistribution.txt', delim_whitespace=True, header=None)
    Nmrk = 100
    vmax = 12950000
    vr = data[0].to_numpy()
    vz = data[1].to_numpy()
    vphi = data[2].to_numpy()
    bhats = data[3].to_numpy()
    pitches = data[6].to_numpy()
    gyro_angles = data[7].to_numpy()
    rzphi(vr,vz,vphi,vmax,Nmrk)


def alexa_data_plot():
    # reading data
    df = read_file('alexa_marker_set_03.txt')

    n = df['n'].to_numpy()
    Nmrk = n.size

    vr = df['vr'].to_numpy()
    vphi = df['vphi'].to_numpy()
    vz = df['vz'].to_numpy()
    vtot = np.sqrt(vr**2 + vphi**2 + vz**2)
    print(np.max(vtot),np.min(vtot))
    vmax = np.max(vtot)



    # fraction of the data to plot
    percentage = 0.3
    random_ii = np.random.choice(Nmrk,size = int(percentage*Nmrk),replace=False)
    Nmrk_mini = len(random_ii)
    print(f"we are plotting {Nmrk_mini} markers")

    # Plot
    vr_mini = vr[random_ii]
    vphi_mini = vphi[random_ii]
    vz_mini = vz[random_ii]
    rzphi(vr_mini,vphi_mini,vz_mini,vmax,Nmrk_mini)

    plot_histogram(vtot, xlabel='vtot', title='Distribution of vtot', bins=100)






alexa_data_plot()