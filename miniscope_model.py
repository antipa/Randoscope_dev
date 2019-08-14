import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import scipy.special as scsp
from miniscope_utils_tf import *
import scipy.io
import cv2


class Model(tf.keras.Model):
    def __init__(self,target_res=0.005,lenslet_CA=0.165,zsampling = 'uniform_random', cross_corr_norm = 'log_sum_exp', aberrations = False,GrinAber='',GrinDictName='', zernikes = [],psf_norm = 'l1',logsumparam = 1e-2,psf_scale=1e2,psf_file = "../psf_meas/zstack_sim_test.mat",loss_type = "matrix_coherence",lenslet_spacing = 'poisson', Nlenslets = 'auto',incoherent_source = False):   #'log_sum_exp'
        # psf_scale: scale all PSFS by this constant after normalizing
        super(Model, self).__init__()
        target_option = 'airy'
        self.lenslet_spacing = lenslet_spacing
        self.loss_type = loss_type   #matrix_coherence: use cross corelations to design PSF
                                       #psf_error: fit to measured PSF
        self.incoherent_source = incoherent_source   #If True, convolve PSF with source shape
        self.source_size = 4.8   #um, diameter
        self.mag = 6.1   #System magnification

        #self.samples = (512,512)  #Grid for PSF simulation
        self.samples = (972,1296)  #Grid for PSF simulation
        #self.samples = (1024,1024)
        lam = 510e-3
        self.lam=tf.constant(lam,dtype=tf.float64)
        
        if GrinAber:
            #'/media/hongdata/Kristina/MiniscopeData/GrinAberrations.mat'
            # 'GrinAberrations'
            file=scipy.io.loadmat(GrinAber)
            GrinAber=file[GrinDictName]
            self.Grin=[]
            for i in range(len(GrinAber)):
                Grinpad=pad_frac_tf(GrinAber[i,:,:]*self.lam, padfrac=0.5)
                Grinresize=cv2.resize(Grinpad.numpy(),(self.samples[1],self.samples[0]))
                self.Grin.append(Grinresize)
                
        # min and max lenslet focal lengths in mm
        self.fmin = 5.5e3
        self.fmax = 29.0e3
        self.ior = 1.515
        
        self.psf_norm = psf_norm
        # Min and max lenslet radii
        self.Rmin = self.fmin*(self.ior-1.)
        self.Rmax = self.fmax*(self.ior-1.)
        self.psf_scale = tf.constant(psf_scale,tf.float64);
        # Convert to curvatures
        self.cmin = 1/self.Rmax
        self.cmax = 1/self.Rmin
        #self.xgrng = np.array((-2851.2, 2851.2)).astype('float32')    #Range, in mm, of grid of the whole plane (not just grin)
        #self.ygrng = np.array((-2138.4,2138.4)).astype('float32')
        self.xgrng = np.array((-2592*2.2/4, 2592*2.2/4)).astype('float64')
        self.ygrng = np.array((-2138.4/2,2138.4/2)).astype('float64')
        #self.xgrng = np.array((-1.8,1.8)).astype('float32')
        #self.ygrng = np.array((-1.8,1.8)).astype('float32')

        self.t = 8.74e3
        # 8.74    #Distance to sensor from mask in mm

        #Compute depth range of virtual image that mask sees (this is assuming an objective is doing some magnification)

        self.zmin_virtual = 1./(1./self.t - 1./self.fmin)
        self.zmax_virtual = 1./(1./self.t - 1./self.fmax)
        self.CA = .85e3; #semi clear aperature of GRIN
        self.mean_lenslet_CA = tf.constant(lenslet_CA,tf.float64) #average lenslest semi clear aperture in mm. 
            
        #Getting number of lenslets and z planes needed as well as defocus list
        self.ps = (self.xgrng[1] - self.xgrng[0])/(self.samples[1])
        if Nlenslets is 'auto':
            self.Nlenslets=np.int(np.floor((self.CA**2)/(self.mean_lenslet_CA**2)))
        else:
            self.Nlenslets = Nlenslets
        self.Nz = 10
        self.zsampling = zsampling
        self.grid_z_planes=10
        

        #self.defocus_grid=  1./(np.linspace(1/self.zmin_virtual, 1./self.zmax_virtual, self.grid_z_planes)) #mm or dioptres

        #if self.zsampling is 'fixed':
        #    self.defocus_list = 1./(np.linspace(1/self.zmin_virtual, 1./self.zmax_virtual, self.Nz)) #mm or dioptres
            
        self.min_offset= 0#-10e-3
        self.max_offset= 50
        #self.lenslet_offset=tfe.Variable(tf.zeros(self.Nlenslets),name='offset', dtype = tf.float64)
        #self.lenslet_offset=tfe.Variable(tf.zeros(self.Nlenslets),name='offset', dtype = tf.float64,constraint=lambda t: tf.clip_by_value(t,self.min_offset, self.max_offset))
        
        #initializing the x and y positions
        #[xpos,ypos, rlist]=poissonsampling_circular(self)
        if self.lenslet_spacing is 'poisson':
            [xpos, ypos] = bridson_poisson_N(r1 = self.mean_lenslet_CA, 
                                         r2=.075e3,CA=self.CA,H=2*self.CA,W=2*self.CA,Nlenslets=self.Nlenslets)
        elif self.lenslet_spacing is 'uniform':
            [xpos, ypos] = bridson_poisson_N(r1 = .05e3, 
                                         r2=.075e3,CA=self.CA,H=2*self.CA,W=2*self.CA,Nlenslets=self.Nlenslets)
            
        rlist=np.random.permutation(1/(np.linspace(1/self.Rmax, 1/self.Rmin, self.Nlenslets)))
        
        self.min_r= self.Rmin * 0.3
        self.max_r= self.Rmax
        
        
        
# Variables        
        self.rlist = tfe.Variable(rlist,name='rlist', dtype = tf.float64,
                                  constraint=lambda t: tf.clip_by_value(t,self.min_r, self.max_r))
        
        # Add variable for defocus as input to self.gen_psf_stack
        
        #self.rlist = tfe.Variable(rlist,name='rlist',dtype=tf.float64)
        #self.xpos = tfe.Variable(xpos, name='xpos', dtype = tf.float64, constraint=lambda t: tf.clip_by_value(t,-self.CA, self.CA))
        #self.ypos = tfe.Variable(ypos, name='ypos', dtype = tf.float64, constraint=lambda t: tf.clip_by_value(t,-self.CA, self.CA))
        self.xpos = tfe.Variable(xpos, name='xpos', dtype = tf.float64)
        self.ypos = tfe.Variable(ypos, name='ypos', dtype = tf.float64)
        
        self.defocus_offset = tfe.Variable(tf.zeros(self.Nz,tf.float64), name='defocus_offset', dtype = tf.float64,
                                          constraint = lambda t: tf.clip_by_value(t,-1./100000,1./100000))
        self.lenslet_offset=tf.Variable(tf.zeros(self.Nlenslets,tf.float64),name='lenslet_offset',dtype=tf.float64,
                                        constraint=lambda t: tf.clip_by_value(t,-4,5))
        #parameters for making the lenslet surface
        self.yg = tf.constant(np.linspace(self.ygrng[0], self.ygrng[1], self.samples[0]),dtype=tf.float64)
        self.xg=tf.constant(np.linspace(self.xgrng[0], self.xgrng[1], self.samples[1]),dtype=tf.float64)
        
        #Pixel size:
        self.px=tf.constant(self.xg[1] - self.xg[0],tf.float64)
        self.py=tf.constant(self.yg[1] - self.yg[0],tf.float64)
        
        # Setup grids
        self.xgm, self.ygm = tf.meshgrid(self.xg,self.yg)

        
        # Normalized coordinates
        self.xnorm =  self.xgm/np.max(self.xgm)
        self.ynorm =  self.ygm/np.max(self.ygm)

        #PSF generation parameters
        
        self.k = tf.constant(np.pi*2./self.lam)
        
        #Compute frequency grid
        fx = tf.constant(np.linspace(-1./(2.*self.ps),1./(2.*self.ps),self.samples[1]),dtype=tf.float64)
        fy = tf.constant(np.linspace(-1./(2.*self.ps),1./(2.*self.ps),self.samples[0]),dtype=tf.float64)
        self.Fx,self.Fy = tf.meshgrid(fx,fy)
        
        # Field point to consider during modeling
        self.field_list = tf.constant(np.array((0., 0.)).astype('float64'))
        
        
        M=6.   #Magnification
        
        self.corr_pad_frac = 0
        self.target_res = target_res*M# micron   

        D=tf.constant(10.,tf.float64)   #1.22*self.lam/(tf.sin(2*tf.atan(self.target_res/(2*self.t))))  #0.514 for half_max, 1.22 for first zero. 
        x_airy=self.k*D*tf.sin(tf.atan(self.xgm/self.t))/2
        y_airy=self.k*D*tf.sin(tf.atan(self.ygm/self.t))/2

        Ra = tf.sqrt(tf.square(x_airy) + tf.square(y_airy))

        target_airy = (2*scsp.j1(Ra)/Ra)**2
        self.target_airy = target_airy/tf.sqrt(tf.reduce_sum(tf.square(target_airy)))
        self.target_airy_pad = pad_frac_tf(self.target_airy, self.corr_pad_frac)

        sig_aprox=0.42*self.lam*self.t/D
        self.airy_aprox_target = tf.exp(-(tf.square(self.xgm) + tf.square(self.ygm))/(2*tf.square(sig_aprox)))

        self.airy_aprox_pad = pad_frac_tf(self.airy_aprox_target / tf.reduce_max(self.airy_aprox_target), self.corr_pad_frac)

        if target_option=='airy':
            self.target_F = tf.square(tf.abs(tf.fft2d(tf.complex(tf_fftshift(self.target_airy_pad), 
                                                                 tf.constant(0.,tf.float64)))))
        elif target_option=='gaussian':
            self.target_F = tf.abs(tf.fft2d(tf.complex(tf_fftshift(self.airy_aprox_pad), tf.constant(0.,tf.float64))))
        
        if self.loss_type is "psf_error":
            psf_in = psf_in = sc.io.loadmat(psf_file)
            Nz_in = np.shape(psf_in['zstack'])[0]
            assert(Nz_in == self.Nz,'Measured PSF has different number of zplanes than model')
            self.target_psf = [tf.constant(psf_in['zstack'][n].astype(np.float64)*self.psf_scale,tf.float64) for n in range(np.shape(psf_in['zstack'])[0])]

        
        # Set regularization. Problem will be treated as l1 of spectral error at each depth + tau * l_inf(cross_corr)
        self.cross_corr_norm = cross_corr_norm
        self.logsumexp_param = tf.constant(logsumparam, tf.float64)   #lower is better l-infinity approximation, higher is worse but smoother
        self.tau = tf.constant(30,tf.float64)    #Extra weight for cross correlation terms
        
        
        self.ignore_dc = True   #Ignore DC in autocorrelations when computing frequency domain loss
        
                                         
        #Mask for DC. Set to zero anywhere we want to ignore loss computation of power spectrum
        dc_mask = np.ones_like(self.target_F.numpy())  
        dc_mask[:3,:3] = 0
        dc_mask[-2:,:1] = 0
        dc_mask[:1,-2:] = 0
        dc_mask[-2:,-2:]= 0
        dc_mask = dc_mask * self.target_F.numpy()>(.001*np.max(self.target_F.numpy()))
        self.dc_mask = tf.constant(dc_mask,tf.float64)
        
        # Use Zernike aberrations with random initialization 
        # Zernike 
        self.zernikes = zernikes
        self.numzern = len(zernikes)

        if aberrations == True:
            zern_init = []
            for i in range(self.Nlenslets):
                aberrations = np.zeros(self.numzern)
                zern_init.append(aberrations)
    
            self.zernlist = tf.Variable(zern_init, dtype='float64', name='zernlist')
        else:
            self.zernlist = []
        
        
    def call(self, defocus_list):

        if np.size(defocus_list) == 1:
            defocus_list = tf.constant(1./(np.linspace(1/self.zmin_virtual, 1./self.zmax_virtual, self.Nz)),
                                       tf.float64)#mm or dioptres
            
            
        T, aper,_ = make_lenslet_tf_zern(self) 
        #T,aper=make_lenslet_tf(self) #offset added
        # Get psf stack
        zstack = self.gen_psf_stack(T, aper, 0.0, 1./(1./defocus_list + self.defocus_offset))
        
        
        ## Smooth with bead size
        if self.incoherent_source is True:
            
            source_size_sensor = np.ceil(self.source_size*self.mag/self.px)
            xkern = np.r_[-np.floor(source_size_sensor/2):np.ceil(source_size_sensor/2)]
            Xkern, Ykern = np.meshgrid(xkern,xkern)
            Rkern = np.sqrt(Xkern**2 + Ykern**2)
            kern_numpy = (Rkern<=(source_size_sensor/2)).astype(np.float64)
            kern_numpy /= np.sum(kern_numpy)
            self.source_kern = tf.constant(kern_numpy,tf.float64)
            zstack = [tf_2d_conv(zstack[n], self.source_kern,'SAME') for n in range(self.Nz)]
        
        
        
        #Fcorr_2dlist = self.gen_cross_spectra(psf_spect)
       
        Rmat_tf_diag = []
        Rmat_tf_off_diag = []
        #calculating Xcorr
        
        
        # This now computes diagonals and off-diagonals separately then concatenates them. 
        # this makes is easier to "find" the diagonals/off diagonals for separate treatment  later.
        if self.loss_type is "matrix_coherence":
            psf_spect = self.gen_stack_spectrum(zstack)
            normsize=tf.to_float(tf.shape(psf_spect)[0]*tf.shape(psf_spect)[1])
        
            for z1 in range(self.Nz):
                for z2 in range(z1, self.Nz):
                    Fcorr = tf.conj(psf_spect[z1])*psf_spect[z2]
                    #Fcorr = Fcorr_2dlist[z1][z2]
                    if z1 == z2:
                        # Difference between autocorrelation and target bandwidth
                        if self.ignore_dc:  
                            # Remove DC (assume no fftshift!)

                            Rmat_tf_diag.append(tf.reduce_sum(self.dc_mask * (tf.square((tf.abs(Fcorr) - self.target_F)/normsize))))
                        else:
                            Rmat_tf_diag.append(tf.reduce_sum(tf.square((tf.abs(Fcorr) - self.target_F)/normsize)))

                    else:
                        # Target is zero for cross correlation
                        if self.cross_corr_norm is 'l2':
                            Rmat_tf_off_diag.append( self.tau * tf.reduce_sum(tf.square(tf.abs(Fcorr)/normsize)))  #changed to one norm
                        elif self.cross_corr_norm is 'log_sum_exp':   
                            # Implementation of eq. 7 from http://users.cecs.anu.edu.au/~yuchao/files/SmoothApproximationforL-infinityNorm.pdf
                            ccorr = tf.abs(tf.ifft2d(Fcorr))
                            Rmat_tf_off_diag.append(self.tau * self.logsumexp_param * tf.reduce_logsumexp((ccorr)/self.logsumexp_param) )

                        elif self.cross_corr_norm is 'inf':
                            Rmat_tf_off_diag.append( self.tau * tf.reduce_max(tf.abs(tf.ifft2d(Fcorr))))
            
        
        elif self.loss_type is "psf_error":
            # List of l2 loss (squared) for each z plane
                       
            zstack = [zstack[n]*tf.reduce_sum(self.target_psf[n] * zstack[n])/tf.reduce_sum(zstack[n]**2) 
                      for n in range(self.Nz)]
#             Rmat_tf_diag = [0.333*(tf.reduce_sum(tf.abs(diff_tf(zstack[n] - self.target_psf[n],0)))
#                                              +tf.reduce_sum(tf.abs(diff_tf(zstack[n] - self.target_psf[n],1)))
#                                              +tf.reduce_sum(tf.abs(zstack[n] - self.target_psf[n])))
#                                                 for n in range(self.Nz)] 
            Rmat_tf_diag = [tf.reduce_sum(tf.abs(zstack[n] - self.target_psf[n])) for n in range(self.Nz)] 


        Rmat_tf = tf.concat([Rmat_tf_diag, Rmat_tf_off_diag],0)  
        return Rmat_tf #note this returns int data type!! vector not matrix. This is also my loss!
    
    def gen_correlation_2d(self,psf_spect):
        Fcorr=[[] for n in range(tf.shape(psf_spect)[0])]
        for z1 in range(tf.shape(psf_spect)[0]):
            [Fcorr[z1].append([]) for n in range(z1)]
            for z2 in range(z1,tf.shape(psf_spect)[0]):
                Fcorr[z1].append(tf_fftshift(tf.ifft2d(tf.conj(psf_spect[z1])*psf_spect[z2])))

        return Fcorr
    
    def gen_correlation_stack(self,psf_spect):
        Fcorr=[]
        for z1 in range(tf.shape(psf_spect)[0]):
            for z2 in range(z1,tf.shape(psf_spect)[0]):
                Fcorr.append(tf_fftshift(tf.ifft2d(tf.conj(psf_spect[z1])*psf_spect[z2])))

        return Fcorr
        
    def gen_psf_stack(self, T, aper, prop_pad, zplanes=0):
        zstack = []
        
        if np.size(zplanes) == 1:
            zplanes = tf.constant(1./(np.linspace(1/self.zmin_virtual, 1./self.zmax_virtual, self.Nz)),
                                  tf.float64)#mm or dioptres
            
        #if np.size(zplanes_opt) == 0:
        #    zplanes = self.defocus_list
        #else:
        #    zplanes = zplanes_opt   
        for z in range(0, tf.size(zplanes)):
            
           # zstack.append(gen_psf_ag_tf(T,self,zplanes[z],'angle',0., prop_pad,Grin=self.Grin[z]))
            zstack.append(gen_psf_ag_tf(T,self,zplanes[z],'angle',0., prop_pad,Grin=0,normalize=self.psf_norm))

        return zstack
    
    def gen_stack_spectrum(self, zstack):
                #Padding for fft
        psf_pad=[]
#         Rmat = np.zeros((self.Nz,self.Nz))

        for z1 in range(tf.shape(zstack)[0]):
            psf_pad.append(pad_frac_tf(zstack[z1],self.corr_pad_frac)) #how much to pad? 

        psf_spect=[]
        
        #Getting spectrum

        for z1 in range(tf.shape(zstack)[0]):
            psf_spect.append(tf.fft2d(tf.complex(tf_fftshift(psf_pad[z1]),tf.constant(0.,dtype = tf.float64))))

        return psf_spect