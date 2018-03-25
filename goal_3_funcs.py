from numba import jit
import matplotlib.pyplot as plt
import numpy as np

@jit
def circle_coordinates(coordinates, radius,density=20):
    """
    Parameters:
    --------------------
    coordinates: of x,y
    radius: given
    density: how sparse the points on the circle are

    Returns:
    --------------------
    Circle coordinates
    """
    latitude = coordinates[0] # latitude of circle center, decimal degrees
    longitude = coordinates[1]
    x,y=[],[]
    density = int(np.ceil(density*radius))
    for k in range(int(density)+1):
        a = np.pi*2*k/(density+1)
        dx = radius * np.cos(a) + latitude
        dy = radius * np.sin(a) + longitude
        x.append(dx)
        y.append(dy)
    return np.array(x),np.array(y)
@jit
def azimuth(x,y):
    """
    Parameters:
    --------------------
    coordinates: of x,y


    Returns:
    --------------------
    Azimuth angle values
    """
    r=np.zeros(len(x))
    for i in range(len(x)-1):
        if i == len(x):
            X1, X2 = x[i], x[0]
            Y1, Y2 = y[i], y[0]
        else:
            X1, X2 = x[i], x[i+1]
            Y1, Y2 = y[i], y[i+1]
        r[i]=np.arctan2(X2-X1,Y2-Y1)/np.pi
    return r
def to_magic_units(data,x):
    return (np.array(x)-0.5*np.array(data.shape[::-1]))/23

def single_frame(data, vortex = [2,2], radii = [1],savefig=False):
    '''
    PHASE VS AZIMUTH

    Parameters:
    --------------------
    data: phase data
    vortex: x and y coordinates
    radii: radius input of every circle you want

    Returns:
    --------------------
    single frame plot
    '''
    markers_list = ["<",">","1","2","3","4","8"]
    phases,azimuths = {}, {}

    plt.figure(figsize=(12,5))

    ##FIRST PLOT
    plt.subplot(121)
    p, q = to_magic_units(data,[[0,0],data.shape[::-1]])
    im=plt.imshow(data, extent=(p[0],q[0],p[1],q[1]))

    for i in range(len(radii)):
        x,y = circle_coordinates(vortex,radii[i]/2,
                                density=30)
        data_1 = np.vstack((x,y)).T
        print(data_1.shape)
        data_1 = to_magic_units(data,data_1)

        plt.plot(data_1[:,0],-data_1[:,1],"--o",linewidth=2)

        phase = [data[int(y[j]+0.5),
                      int(x[j]+0.5)] for j in range(len(x))]
        az = azimuth(x,y)
        phases.update({str(i):phase})
        azimuths.update({str(i):az})

    plt.colorbar(im)

    plt.xlabel("x ($\mu$m)",fontsize=25)
    plt.ylabel("y ($\mu$m)",fontsize=25)

    ##SECOND PLOT
    plt.subplot(122)

    for i in range(len(phases)):
        plt.plot(azimuths[str(i)],
                 phases[str(i)],
                 label = str(np.round(radii[i]/23,2)) + " $\mu$m",
                 marker=markers_list[i])
    plt.legend()
    plt.ylabel("Phase ($\pi$)",fontsize=25)
    plt.xlabel("Azimuthal angle ($\pi$)",fontsize=25)
    if savefig == True:
        import sys
        import os
        file_name =  os.path.basename(sys.argv[0]).split(".")[0]
        plt.savefig(file_name+".png")
    plt.show()
