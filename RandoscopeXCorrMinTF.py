import numpy as np
import tensorflow as tf
def make_lenslet_surface(Xlist, Ylist, Rlist, xrng, yrng, samples,aperR,r_lenslet):
    # Takes in Xlist, Ylist and Rlist: floating point center and radius values for each lenslet
    # xrng and yrng: x and y range (tuple) over which to define grid
    # samples: tuple of number of samplex in x and y, respectively
    #
    # Outputs: T, the aperture thickness function. 

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
        #sph = np.real(np.sqrt(0j+Rlist[n]**2 - (xg-Xlist[n])**2 - (yg-Ylist[n])**2))-Rlist[n]+10
        sph1 = np.real(np.sqrt(0j+Rlist[n]**2 - (xg-Xlist[n])**2 - (yg-Ylist[n])**2))-np.real(np.sqrt(0j+Rlist[n]**2-r_lenslet**2))

        T = np.maximum(T,sph1)
    aper = np.sqrt(xg**2+yg**2) <= aperR
    
    return T-np.min(T[aper]), aper, px, py

def poissonsampling_circular(model):
    EAperR=model.CA-model.mean_lenslet_CA
    xrng=[-(model.CA),model.CA] 
    yrng=xrng
    Rrng=[model.Rmin,model.Rmax]
    rlist = np.random.permutation(1/(np.linspace(1/model.Rmax, 1/model.Rmin, model.Nlenslets)))
    xpos = (xrng[1]-xrng[0])*np.random.rand(model.Nlenslets) + xrng[0]
    ypos = (yrng[1]-yrng[0])*np.random.rand(model.Nlenslets) + yrng[0]
    lens_curv = 1/rlist
    while (sum((xpos**2+ypos**2)>EAperR)!=0):
        xpos = (xrng[1]-xrng[0])*np.random.rand(model.Nlenslets) + xrng[0]
        ypos = (yrng[1]-yrng[0])*np.random.rand(model.Nlenslets) + yrng[0]
    # poission disc sampling (kindof)
    step=model.mean_lenslet_CA
    #check if points are too close and have them walk!
    i=0
    j=0
    while i<model.Nlenslets:
        while j<model.Nlenslets:
            if i!=j:
                d=np.sqrt((xpos[i]-xpos[j])**2+(ypos[i]-ypos[j])**2)
                if d<1.2*model.mean_lenslet_CA or xpos[j]**2+ypos[j]**2>EAperR**2 or xpos[i]**2+ypos[i]**2>EAperR**2:
                    vx=np.random.rand()
                    vy=np.random.rand()
                    if vx>0.5:
                        directionx=1
                    else:
                        directionx=-1
                    if vy>0.5:
                        directiony=1
                    else:
                        directiony=-1
                    xpos[j]=xpos[j]+directionx*step
                    ypos[j]=ypos[j]+directiony*step
                    while xpos[j]**2+ypos[j]**2>EAperR**2 or ypos[j]>yrng[1] or ypos[j]<yrng[0] or xpos[j]>xrng[1] or xpos[j]<xrng[0]:
                        while xpos[j]>xrng[1] or xpos[j]<xrng[0]:
                            if xpos[j]>xrng[1]:
                                #xpos(j)=xpos(j)-step;
                                xpos[j]=xrng[1]
                            elif xpos[j]<xrng[0]:
                                #xpos(j)=xpos(j)+step;
                                xpos[j]=xrng[0]
                        while ypos[j]>yrng[1] or ypos[j]<yrng[0]:
                            if ypos[j]>yrng[1]:
                                #ypos(j)=ypos(j)-step;
                                ypos[j]=yrng[1]
                            elif ypos[j]<yrng[0]:
                                #%ypos(j)=ypos(j)+step;
                                ypos[j]=yrng[0]

                        while xpos[j]**2+ypos[j]**2>EAperR**2:
                            step2=np.abs(xpos[j]**2+ypos[j]**2)-EAperR**2
                            directionx=-np.sign(xpos[j])
                            directiony=-np.sign(ypos[j])
                            xpos[j]=xpos[j]+directionx*step2
                            ypos[j]=ypos[j]+directiony*step2
                    i=0
            j=j+1
        j=0
        i=i+1
    return xpos, ypos, rlist

def make_lenslet_tf(model):
        T = tf.zeros([model.samples[0],model.samples[1]])
        for n in range(model.Nlenslets):
            sph1 = tf.real(tf.sqrt(tf.square(model.rlist[n]) - tf.square((model.xgm-
                                                                          model.xpos[n])) - tf.square((model.ygm-model.ypos[n]))))-tf.real(tf.sqrt(tf.square(model.rlist[n]
                                                                          )-tf.square(model.mean_lenslet_CA)))
            T = tf.maximum(T,sph1)
        aper = tf.sqrt(model.xgm**2+model.ygm**2) <= model.CA
        return T,aper
    
def crop2d(x, crop_size):
    cstart = (x.shape - crop_size)//2
    return x[cstart[0]:cstart[0]+crop_size[0], cstart[1]:cstart[1]+crop_size[1]]  


def tf_exp(arg):
    U_in_r = tf.cos(arg)
    U_in_i = tf.sin(arg)
    return tf.complex(U_in_r,U_in_i)
    


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
    
def propTF(u1,L,lam,z):
    #propagation-transfer function approach
    #assume same x and y side length
    # uniform sampling
    #u1-source plane field
    #L- source and observation plane side length
    # lambda-wavelength
    #z=prop distance
    #u2=observation plane field
    [M,N]=np.shape(u1) #get input field array size
    dx=L/M #sample interval
    k=2*np.pi/lam

    fx=np.linspace(-1/(2*dx),1/(2*dx)-1/L,T1.shape[1]) #freq coords
    fy=np.linspace(-1/(2*dx),1/(2*dx)-1/L,T1.shape[0])
    [FX,FY]=np.meshgrid(fx,fx)
    #ignoring exp(jkz) since it does not affect the transverse spatial
    #structure of observation plane
    H=np.exp(1j*k*z-1j*np.pi*lam*z*(FX**2+FY**2)) #trans function
    H=np.fft.fftshift (H); #shift trans funciton
    U1=np.fft.fft2(np.fft.fftshift(u1)) #shift fft src field
    U2=H*U1;
    u2=np.fft.ifftshift(np.fft.ifft2(U2))
    return u2