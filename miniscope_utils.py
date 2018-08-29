def make_lenslet_surface(Xlist, Ylist, Rlist, xrng, yrng, samples,aperR):
    # Takes in Xlist, Ylist and Rlist: floating point center and radius values for each lenslet
    # xrng and yrng: x and y range (tuple) over which to define grid
    # samples: tuple of number of samplex in x and y, respectively
    #
    # Outputs: T, the aperture thickness function. 
    import numpy as np
    assert np.shape(Xlist) == np.shape(Ylist), 'Xlist and Ylist must be same shape'
    Nlenslets = np.shape(Xlist)[0]
    if np.shape(Rlist) == ():
        Rlist = np.tile(Rlist, np.shape(Xlist))
    else:
        assert np.shape(Rlist) == np.shape(Xlist), 'Xlist must be either a scalar or list matching shapes of Xlist and Ylist'
    assert type(xrng) is tuple, 'xrng must be tuple'
    assert type(yrng) is tuple, 'yrng must be tuple'
    assert type(samples) is tuple, 'samples must be tuple'
    

    
    T = np.zeros(samples)
    xg = np.linspace(xrng[0], xrng[1], samples[1])
    yg = np.linspace(yrng[0], yrng[1], samples[0])
    px = xg[1] - xg[0]
    py = yg[1] - yg[0]
    xg, yg = np.meshgrid(xg,yg)
    for n in range(Nlenslets):
        sph = np.real(np.sqrt(0j+Rlist[n]**2 - (xg-Xlist[n])**2 - (yg-Ylist[n])**2))-Rlist[n]+10
        T = np.maximum(T,sph)
    aper = np.sqrt(xg**2+yg**2) <= aperR

    return T-np.min(T[aper]), aper, px, py

def prop_field(Ui, z, lam, Fx, Fy):
    import numpy as np
    from numpy import fft    
    Hf = np.exp(1j*2*np.pi*z/lam*np.sqrt(1-(lam*Fx)**2 - (lam*Fy)**2))
    R = np.sqrt(Fx**2 + Fy**2)
    roi = R>1/lam
    Hf[roi] = 0


    Uf = fft.fftshift(fft.fft2(fft.ifftshift(Ui)))
    #plt.imshow(np.imag(Uf))
    Up = Uf*Hf
    uf = fft.fftshift(fft.ifft2(fft.ifftshift(Up)))
    return uf
    