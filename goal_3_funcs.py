from numba import jit
import matplotlib.pyplot as plt
import numpy as np

@jit
def circle_coordinates(coordinates, radius,density=20):
    latitude = coordinates[0] # latitude of circle center, decimal degrees
    longitude = coordinates[1]
    x,y=[],[]
    density = int(np.ceil(density*radius)) + 1
    for k in range(int(density)+1):
        a = np.pi*2*k/density
        dx = radius * np.cos(a) + latitude
        dy = radius * np.sin(a) + longitude
        x.append(dx)
        y.append(dy)
    return x,y
@jit
def azimuth(x,y):
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

def single_frame(data, vortex = [2,2], radii = [1],savefig=False):
    '''
    PHASE VS AZIMUTH

    Parameters:
    --------------------
    data: phase data
    vortex: x and y coordinates
    radii: radius input of every circle you want
    '''
    markers_list = ["<",">","1","2","3","4","8"]
    phases,azimuths = {}, {}

    plt.figure(figsize=(12,5))

    ##FIRST PLOT
    plt.subplot(121)
    im=plt.imshow(data, extent=(-4,4,-4,4))

    for i in range(len(radii)):
        x,y = circle_coordinates(vortex,radii[i]/2,
                                density=30)
        plt.plot(x,y,"--o")

        phase = [data[int(y[j]*23),
                      int(x[j]*23)] for j in range(len(x))]
        az = azimuth(x,y)
        phases.update({str(i):phase})
        azimuths.update({str(i):az})

    plt.colorbar(im)#,label="Phase ($\pi$)")
    s=0.1
    plt.axis([vortex[0] - max(radii)/2-s,
              vortex[0] + max(radii)/2+s,
              vortex[1] - max(radii)/2-s,
              vortex[1] + max(radii)/2+s])
    plt.xlabel("x ($\mu$m)",fontsize=25)
    plt.ylabel("y ($\mu$m)",fontsize=25)

    ##SECOND PLOT
    plt.subplot(122)

    for i in range(len(phases)):
        plt.plot(azimuths[str(i)],
                 phases[str(i)],
                 label = str(radii[i]) + " $\mu$m",
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
