import numpy as np
import scipy as sc
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from bridson import poisson_disc_samples
import miniscope_utils as ms_utils





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




def make_lenslet_tf_zern(model):
    T = tf.zeros([model.samples[0],model.samples[1]],tf.float64)
    
    if np.shape(model.zernlist) != (): 
        T_orig = tf.zeros([model.samples[0],model.samples[1]],tf.float64)
    else:
        T_orig = []
    
    for n in range(model.Nlenslets):
        #sph1 = model.lenslet_offset[n]+tf.real(tf.sqrt(tf.square(model.rlist[n]) - tf.square((model.xgm- model.xpos[n]))- tf.square((model.ygm-model.ypos[n])))
                                          #)-tf.real(tf.sqrt(tf.square(model.rlist[n])-tf.square(model.mean_lenslet_CA)))
            
        sphere = tf.real(tf.sqrt(
            tf.square(model.rlist[n])
            - tf.square((model.xgm-model.xpos[n]))
            - tf.square((model.ygm-model.ypos[n]))))


        sag = tf.real(tf.sqrt(tf.square(model.rlist[n])-tf.square(model.mean_lenslet_CA)))

        sph1 = model.lenslet_offset[n] + sphere - sag
        
        
        if tf.not_equal(tf.size(model.zernlist),0):  # Including Zernike aberrations 
            # change to normalize by CA
            #Z = zernikecartesian(model.zernlist[n],  model.xnorm - model.xpos[n]/np.max(model.xgm)  ,model.ynorm - model.ypos[n]/np.max(model.xgm))
            x_coord = model.xnorm - model.xpos[n]/np.max(model.xgm)
            y_coord = model.ynorm - model.ypos[n]/np.max(model.xgm)
            Z =  zernike_evaluate(model.zernlist[n], model.zernikes, x_coord, y_coord)
            
            #r = 1. # Not sure about this
            #Z[(model.xnorm  - model.xpos[n]/np.max(model.xgm) )**2+(model.ynorm - model.ypos[n]/np.max(model.xgm))**2>r**2]=0 # Crop to be a circle
            sph2 = sph1 + Z
            
            
            T_orig = tf.maximum(T_orig,sph1)
            T = tf.maximum(T,sph2)
        else:
            T = tf.maximum(T,sph1)
            
        
    aper = tf.sqrt(model.xgm**2+model.ygm**2) <= model.CA
    return T, aper, T_orig


def make_lenslet_tf(model):
        T = tf.zeros([model.samples[0],model.samples[1]])
        for n in range(model.Nlenslets):
            sphere = tf.real(tf.sqrt(
                tf.square(model.rlist[n])
                - tf.square((model.xgm-model.xpos[n]))
                - tf.square((model.ygm-model.ypos[n]))))


            sag = tf.real(tf.sqrt(tf.square(model.rlist[n])-tf.square(model.mean_lenslet_CA)))

            sph1 = model.lenslet_offset[n] + sphere - sag
            T = tf.maximum(T,sph1)
                
        aper = tf.sqrt(model.xgm**2+model.ygm**2) <= model.CA
        return T,aper

def diff_tf(arr,ax):
    ndims = arr.ndim
    slicer_ = tuple(slice(0+int(n==ax),-1,1) for n in range(ndims))
    slicer = tuple(slice(0,-1-int(n==ax),1) for n in range(ndims))
    return(arr[slicer_] - arr[slicer])    
    
def crop2d(x, crop_size):
    cstart = (np.array(x.shape) - crop_size)//2  #there is an NP here!
    return x[cstart[0]:cstart[0]+crop_size[0], cstart[1]:cstart[1]+crop_size[1]]  


def tf_exp(arg):
    U_in_r = tf.cos(arg)
    U_in_i = tf.sin(arg)
    return tf.complex(U_in_r,U_in_i)
    
def tf_fftshift(spectrum):
    spec_in=fftshift(spectrum,axis=0)
    spec_out=fftshift(spec_in,axis=1)
    return spec_out
    
def fftshift(spectrum, axis=-1):
  try: 
    shape = spectrum.shape[axis].value
  except:
    shape = None
  if shape is None:
    shape = tf.shape(spectrum)[axis]
  # Match NumPy's behavior for odd-length input. The number of items to roll is
  # truncated downwards.
  b_size = shape // 2
  a_size = shape - b_size
  a, b = tf.split(spectrum, [a_size, b_size], axis=axis)
  return tf.concat([b, a], axis=axis)

def pad_frac_tf(x, padfrac=0):
    pads=np.ceil(tf.to_float(tf.shape(x))*padfrac).astype('int')
    paddings = tf.constant([[pads[0], pads[0]], [pads[1], pads[1]]])
    return tf.pad(x,paddings,'constant')

def propagate_field_freq_tf(model,U,padfrac=0):
    if padfrac != 0:
        shape_orig = np.shape(U)
        U = pad_frac_tf(U, padfrac)
        Fx, Fy = tf.meshgrid(tf.lin_space(tf.reduce_min(model.Fx), tf.reduce_max(model.Fx), U.shape[1]), tf.lin_space(tf.reduce_min(model.Fy), tf.reduce_max(model.Fy), U.shape[0]))
        Fxn=Fx
        Fyn=Fy
    else:
        Fxn=model.Fx
        Fyn=model.Fy
    Uf = tf_fftshift(tf.fft2d(tf_fftshift(U)))

    Hf = tf_exp(2.*np.pi*model.t/model.lam * tf.sqrt(1-tf.square(model.lam*Fxn) - tf.square(model.lam*Fyn)))
    Up = tf_fftshift(tf.ifft2d(tf_fftshift(Uf*Hf)))
    if padfrac != 0:
        Up = crop2d(Up, shape_orig)
    return Up

def gen_psf_ag_tf(T,model,z_dis, obj_def, pupil_phase=0, prop_pad = 0,Grin=0,normalize='l1'):  #check negatives in exponents
    # Inputs:
    # surface: single surface thickness function, units: mm
    # ior : index of refraction of bulk material
    # t : thickness of surface (i.e. distance to output plane)
    # z_obj : distance from object plane. +Inf means object at infinity
    # object_def : 'angle' for angular field definition, 'obj_height' for finite
    # field : tuple (x,y) wavefront field definition. (0,0) is on-axis. Interpreted in context of object_def
    # CA: radius of clear aperture in mm
    # pupil_aberration: additional pupil phase, in radians!
    # lmbda: wavelength in mm
    # xg and yg are the spatial grid (pixel spacing in mm)
    # Fx : frequency grid in 1/mm
    # Fy : same as Fx  
    # normalize = 'l2','l1','max'

    if obj_def is 'angle':
        ramp_coeff_x = -tf.tan(model.field_list[0]*np.pi/180.)
        ramp_coeff_y = -tf.tan(model.field_list[1]*np.pi/180.)
        ramp = model.xgm*ramp_coeff_x + model.ygm*ramp_coeff_y
        if z_dis is 'inf':
            U_in = tf_exp(model.k*(ramp))
        else:
            U_in = tf_exp(model.k*(Grin+z_dis -z_dis*tf.sqrt(1+tf.square(model.ygm/z_dis)+tf.square(model.xgm/z_dis)) + ramp)) #negative already included
    
    elif obj_def is 'obj_height':
        if z_dis is 'inf':
            raise Exception('cannot use obj_height and object at infinity')
        else:
            U_in = tf_exp(1*-z_dis*model.k*tf.sqrt(1-tf.square((model.xgm-model.field_list[0])/z_dis) - tf.square((model.ygm-model.field_list[1])/z_dis)))
    
    U_out = U_in * tf_exp((model.k*(model.ior-1)*T + pupil_phase))
    amp = tf.cast(tf.sqrt(tf.square(model.xgm) + tf.square(model.ygm)) <= model.CA, tf.float64)

    U_prop = propagate_field_freq_tf(model, tf.cast(tf.complex(tf.real(U_out)*amp,tf.imag(U_out)*amp),tf.complex128), prop_pad)    
    psf = tf.square(tf.abs(U_prop))
    if normalize == 'l2':
        return(psf/tf.sqrt(tf.reduce_sum(tf.square(psf)))) #DO WE NEED TO DO THIS????
    elif normalize == 'l1':
        return(psf/tf.reduce_sum(psf))*model.psf_scale
    elif normalize == 'max':
        return(psf/tf.reduce_max(tf.abs(psf)))
    


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

#have tf do everything for us
def loss (model, inputs):
    Rmat = model(inputs)
    return tf.reduce_sum(Rmat), Rmat

def loss_extra(model, inputs):
    Rmat, err_stack = model(inputs)
    return tf.reduce_sum(Rmat), Rmat, err_stack

def loss_sum(model, inputs):
    Rmat = model(inputs)
    return tf.reduce_sum(Rmat), Rmat

def loss_inf(model, inputs):
    Rmat = model(inputs)
    return tf.reduce_max(Rmat), Rmat

def loss_mixed(model, inputs):
    # max of off diags, sum of diags
    Rmat = model(inputs)
    diag_vec = Rmat[0:model.Nz]
    off_diag = Rmat[model.Nz+1:-1]
    return tf.reduce_sum(tf.abs(diag_vec)) + tf.reduce_max(off_diag), Rmat


def remove_nan_gradients(grads):
   # Get rid of NaN gradients
   for g in range(0,len(grads)):
           new_grad = tf.where(tf.is_nan(grads[g]), tf.zeros_like(grads[g]), grads[g])
           grads[g] = new_grad
   return grads

def project_to_aper_keras(model,CA = -1):
    if CA == -1:
        CA = model.CA
    lenslets_dist = tf.sqrt(tf.square(model.xpos) + tf.square(model.ypos))
   # print(lenslets_dist)
    dist_new = tf.minimum(lenslets_dist, CA)
    
    th = tf.atan2(model.ypos, model.xpos)
    x_new = dist_new * tf.cos(th)
    y_new = dist_new * tf.sin(th)
    model.xpos.assign(x_new)
    model.ypos.assign(y_new)
    
    
def find_best_initialization(model, num_trials = 2000, save_results =False):
    loss_list = []
    defocus_grid=  1./(np.linspace(1/model.zmin_virtual, 1./model.zmax_virtual, model.Nz * 1)) #mm or dioptres
    init_loss,_ = loss(model, defocus_grid)
    loss_list.append(init_loss.numpy())

    x_best = tfe.Variable(tf.zeros(model.Nlenslets));    
    y_best = tfe.Variable(tf.zeros(model.Nlenslets));       
    r_best = tfe.Variable(tf.zeros(model.Nlenslets));
    loss_best = loss_list[0]

    x_worst = tfe.Variable(tf.zeros(model.Nlenslets));  
    y_worst = tfe.Variable(tf.zeros(model.Nlenslets));       
    r_worst = tfe.Variable(tf.zeros(model.Nlenslets));
    loss_worst = loss_list[0]

    for i in range(0,num_trials):
        [xnew,ynew, rnew]=poissonsampling_circular(model)
        tf.assign(model.xpos, xnew)
        tf.assign(model.ypos, ynew)
        tf.assign(model.rlist, rnew)

        current_loss,_ = loss(model, defocus_grid)


        if i%1 == 0:
            print('iteration: ', i, ', min value: ', loss_best, ', max value: ', loss_worst , end="\r")
        if current_loss < np.min(loss_list):
            tf.assign(x_best, xnew)
            tf.assign(y_best, ynew)
            tf.assign(r_best, rnew)
            loss_best = current_loss.numpy()

        elif current_loss > np.max(loss_list):
            tf.assign(x_worst, xnew)
            tf.assign(y_worst, ynew)
            tf.assign(r_worst, rnew)
            loss_worst = current_loss.numpy()


        loss_list.append(current_loss.numpy())
    
    
    # Save 
    dict_best = {'x_best' : x_best.numpy(),
             'y_best': y_best.numpy(),
             'r_best': r_best.numpy(),
             'loss': loss_best}

    dict_worst = {'x_worst' : x_worst.numpy(),
             'y_worst': y_worst.numpy(),
             'r_worst': r_worst.numpy(),
              'loss': loss_worst}

    if save_results == True:
        sc.io.savemat('/media/hongdata/Kristina/MiniscopeData/best_init.mat', dict_best)
        sc.io.savemat('/media/hongdata/Kristina/MiniscopeData/worst_init.mat', dict_worst)

       
    
def zernike_evaluate(coefficients, indixes, x, y):
    zernike_polynomials = [
           lambda x,y,r: 1,                                    # 0: piston
           lambda x,y,r: 2.*x,                                 # 1: tilt
           lambda x,y,r: 2.*y,                                 # 2: tilt 
           lambda x,y,r: 2.*tf.sqrt(6.)*x*y,                   # 3: astigmatism 
           lambda x,y,r: tf.sqrt(3.)*(2.*tf.square(r)-1.),     # 4: defocus
           lambda x,y,r: tf.sqrt(6.)*(x**2-y**2),              # 5: astigmatism 
           lambda x,y,r: tf.sqrt(8.)*y*(3*x**2-y**2),          # 6: trefoil
           lambda x,y,r: tf.sqrt(8.)*x*(3*r**2-2),             # 7: coma
           lambda x,y,r: tf.sqrt(8.)*y*(3*r**2-2),             # 8: coma
           lambda x,y,r: tf.sqrt(8.)*x*(3*x**2-y**2),]          # 9: trefoil
    
    
    r = tf.sqrt(tf.square(x) + tf.square(y))
    
    ZN = 0
    for i in range(0, len(indixes)):
        ZN = ZN + coefficients[i]*zernike_polynomials[indixes[i]](x,y,r)
        
    return ZN


def bridson_poisson_N(r1 = .15, r2=.075,CA=.9, W=1.8,H=1.8,Nlenslets=1e9):
    p = np.array(poisson_disc_samples(width=W,height=H,r=r1))
    p_circ_x,p_circ_y = ms_utils.project_to_aperture(p[:,0]-W/2,p[:,1]-H/2,CA-r2,mode='delete')

    if Nlenslets < len(p_circ_x):
        inds = np.random.choice(len(p_circ_x), Nlenslets,replace=False)
    else:
        inds = range(len(p_circ_x))

    return p_circ_x[inds], p_circ_y[inds]


def tf_2d_conv(x,y,padstr):
    # Inputs x and y tensors (2d)
    x_tensor = tf.reshape(x,[1,tf.shape(x)[0], tf.shape(x)[1], 1])
    
    y_tensor = tf.reshape(y,[tf.shape(y)[0], tf.shape(y)[1],1, 1])
    return tf.squeeze(tf.nn.convolution(x_tensor,y_tensor,padstr))

