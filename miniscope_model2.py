import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import scipy.special as scsp
from miniscope_utils_tf2 import *
import scipy.io
import cv2


class Model(tf.keras.Model):
    def __init__(self,target_res=0.005,lenslet_CA=0.2,zsampling = 'uniform_random', cross_corr_norm = 'log_sum_exp', aberrations = True,GrinAber=[]):   #'log_sum_exp'
        super(Model, self).__init__()
        target_option = 'airy'
        self.samples = (1400,1400)  #Grid for PSF simulation
        
        self.lam=510e-6
        #file=scipy.io.loadmat('GrinAberrations.mat')
        #GrinAber=file['GrinAberrations']
        #self.Grin=[]
        #for i in range(len(GrinAber)):
        #    Grinpad=pad_frac_tf(GrinAber[i,:,:]*self.lam, padfrac=0.5)
        #    Grinresize=cv2.resize(Grinpad.numpy(),(self.samples[1],self.samples[1]))
        #    self.Grin.append(Grinresize)
        # min and max lenslet focal lengths in mm
        self.fmin = 4.50980392#6.41176471#5.4254902
        self.fmax = 10.0937255#15 #14.1529412
        self.ior = 1.51 #1.51
        self.lam=510e-6
        self.fmax_test=self.fmin#15.6862745
        self.fmin_test=self.fmax#5.33529412
        # Min and max lenslet radii
        self.Rmin = self.fmin*(self.ior-1.)
        self.Rmax = self.fmax*(self.ior-1.)
        #self.Rmin = 3.2
        #self.Rmax = 12
        # Convert to curvatures
        self.cmin = 1/self.Rmax
        self.cmax = 1/self.Rmin
        self.xgrng = 1.1*np.array((-1.8, 1.8)).astype('float32')    #Range, in mm, of grid of the whole plane (not just grin)
        self.ygrng = 1.1*np.array((-1.8, 1.8)).astype('float32')
        self.lenslet_CA=lenslet_CA
        self.lenslet_CA2=0.150
        self.t = 10.    #Distance to sensor from mask in mm

        #Compute depth range of virtual image that mask sees (this is assuming an objective is doing some magnification)

        self.zmin_virtual = 1./(1./self.t - 1./self.fmin_test)
        self.zmax_virtual = 1./(1./self.t - 1./self.fmax_test)
        #self.zmin_virtual =  13.263#13.263
        #self.zmax_virtual = -19.85#-19.85
        self.CA = .9; #semi clear aperature of GRIN
        self.Nlenslets=36
        #self.mean_lenslet_CA=tfe.Variable(tf.ones(self.Nlenslets)*lenslet_CA,name='lenslet_CA',dtype=tf.float32,constraint=lambda t: tf.clip_by_value(t,0.16, lenslet_CA))
        self.mean_lenslet_CA = tf.constant(tf.ones(self.Nlenslets)*lenslet_CA,name='lenslet_CA',dtype=tf.float32) #average lenslest semi clear aperture in mm. 
        #Getting number of lenslets and z planes needed as well as defocus list
        self.ps = (self.xgrng[1] - self.xgrng[0])/self.samples[0]
        #self.Nlenslets=np.int(np.floor((self.CA**2)/(self.mean_lenslet_CA**2)))
        
        self.Nz = 5
        self.zsampling = zsampling
        self.grid_z_planes=20
        
        self.numzern = 4
        self.numzern2 = 7
        #self.defocus_grid=  1./(np.linspace(1/self.zmin_virtual, 1./self.zmax_virtual, self.grid_z_planes)) #mm or dioptres

        #if self.zsampling is 'fixed':
        #    self.defocus_list = 1./(np.linspace(1/self.zmin_virtual, 1./self.zmax_virtual, self.Nz)) #mm or dioptres
            
        self.min_offset= 0# -10e-3
        self.max_offset= 5e-3
        #self.lenslet_offset=tfe.Variable(tf.zeros(self.Nlenslets),name='offset', dtype = tf.float32)
        #self.lenslet_offset=tfe.Variable(tf.zeros(self.Nlenslets),name='offset', dtype = tf.float32,constraint=lambda t: tf.clip_by_value(t,self.min_offset, self.max_offset))
        self.lenslet_offset=tf.zeros(self.Nlenslets)
        #initializing the x and y positions
        [xpos,ypos, rlist]=poissonsampling_circular(self,position_mode='initial36',radius_mode='random')
        
        self.min_r= tf.constant(self.Rmin*tf.ones(self.Nlenslets),dtype=tf.float32)
        self.max_r= tf.constant(self.Rmax*tf.ones(self.Nlenslets),dtype=tf.float32)
        min_rr=tf.maximum(self.min_r,rlist-0.1)
        max_rr=tf.minimum(self.max_r,rlist+0.1)
        self.max_rr= tf.constant(max_rr,dtype=tf.float32)
        self.min_rr= tf.constant(min_rr,dtype=tf.float32)
        self.rlist=tf.constant(rlist,dtype = tf.float32)
        #self.rlist = tfe.Variable(rlist,name='rlist', dtype = tf.float32,constraint=lambda t: tf.clip_by_value(t,self.min_rr, self.max_rr))
        #self.xpos = tfe.Variable(xpos, name='xpos', dtype = tf.float32, constraint=lambda t: tf.clip_by_value(t,-self.CA, self.CA))
        #self.ypos = tfe.Variable(ypos, name='ypos', dtype = tf.float32, constraint=lambda t: tf.clip_by_value(t,-self.CA, self.CA))
        #self.xpos = tfe.Variable(xpos, name='xpos', dtype = tf.float32,constraint=lambda t: tf.clip_by_value(t,xpos-0.05, xpos+0.05))
        #self.ypos = tfe.Variable(ypos, name='ypos', dtype = tf.float32,constraint=lambda t: tf.clip_by_value(t,ypos-0.05, ypos+0.05))
        self.xpos = tfe.Variable(xpos, name='xpos', dtype = tf.float32)
        self.ypos = tfe.Variable(ypos, name='ypos', dtype = tf.float32)
        #self.xpos = tf.constant(xpos, name='xpos', dtype = tf.float32)
        #self.ypos = tf.constant(ypos, name='ypos', dtype = tf.float32)
        #parameters for making the lenslet surface
        self.yg = tf.constant(np.linspace(self.ygrng[0], self.ygrng[1], self.samples[0]),dtype=tf.float32)
        self.xg=tf.constant(np.linspace(self.xgrng[0], self.xgrng[1], self.samples[1]),dtype=tf.float32)
        self.px=tf.constant(self.xg[1] - self.xg[0],tf.float32)
        self.py=tf.constant(self.yg[1] - self.yg[0],tf.float32)
        self.xgm, self.ygm = tf.meshgrid(self.xg,self.yg)

        
        # Normalized coordinates
        self.xnorm =  self.xgm/np.max(self.xgm)
        self.ynorm =  self.ygm/np.max(self.ygm)

        #PSF generation parameters
        self.lam=tf.constant(510.*10.**(-6.),dtype=tf.float32)
        self.k = np.pi*2./self.lam
        
        fx = tf.constant(np.linspace(-1./(2.*self.ps),1./(2.*self.ps),self.samples[1]),dtype=tf.float32)
        fy = tf.constant(np.linspace(-1./(2.*self.ps),1./(2.*self.ps),self.samples[0]),dtype=tf.float32)
        self.Fx,self.Fy = tf.meshgrid(fx,fy)
        self.field_list = tf.constant(np.array((0., 0.)).astype('float32'))
        M=5.6
        self.corr_pad_frac = 0
        self.target_res = target_res*M# micron  
        
        
        
        airy_option= 'airy'
        #         sig = 2*self.target_res/(2.355) * 1e-3
        #         real_target = tf.exp(-(tf.square(self.xgm) + tf.square(self.ygm))/(2*tf.square(sig)))
        #         real_target = pad_frac_tf(real_target / tf.reduce_max(real_target), self.corr_pad_frac)
        #         self.target_F = tf.abs(tf.fft2d(tf.complex(tf_fftshift(real_target), 0.)))

        D=1.22*self.lam/(tf.sin(2*tf.atan(self.target_res/(2*self.t))))  #0.514 for half_max, 1.22 for first zero. 
        x_airy=self.k*D*tf.sin(tf.atan(self.xgm/self.t))/2
        y_airy=self.k*D*tf.sin(tf.atan(self.ygm/self.t))/2
        #Xa, Ya = tf.meshgrid(x_airy, y_airy)
        Ra = tf.sqrt(tf.square(x_airy) + tf.square(y_airy))

        #         self.target_airy=((2*scsp.j1(x_airy)/x_airy)**2) *((2*scsp.j1(y_airy)/y_airy)**2)
        target_airy = (2*scsp.j1(Ra)/Ra)**2
        if airy_option=='airy':
            self.target_airy = target_airy/tf.sqrt(tf.reduce_sum(tf.square(target_airy)))
        elif airy_option=='airy_autocorrelation':
            self.target_airy = target_airy/tf.reduce_max(target_airy)
        self.target_airy_pad = pad_frac_tf(self.target_airy, self.corr_pad_frac)

        sig_aprox=0.42*self.lam*self.t/D
        self.airy_aprox_target = tf.exp(-(tf.square(self.xgm) + tf.square(self.ygm))/(2*tf.square(sig_aprox)))

        self.airy_aprox_pad = pad_frac_tf(self.airy_aprox_target / tf.reduce_max(self.airy_aprox_target), self.corr_pad_frac)

        if target_option=='airy':
            self.target_F = tf.square(tf.abs(tf.fft2d(tf.complex(tf_fftshift(self.target_airy_pad), 0.))))
        elif target_option=='airy_autocorrelation':
            self.airy_spect=tf.fft2d(tf.complex(tf_fftshift(self.target_airy_pad),tf.constant(0.,dtype = tf.float32)))
            self.acorr_F = tf.conj(self.airy_spect)*self.airy_spect
            self.target_F = tf.abs(tf.ifft2d(self.acorr_F))
        elif target_option=='airy_autocorrelation_frequency':
            self.airy_spect=tf.fft2d(tf.complex(tf_fftshift(self.target_airy_pad),tf.constant(0.,dtype = tf.float32)))
            self.target_F = tf.conj(self.airy_spect)*self.airy_spect
        elif target_option=='gaussian':
            self.target_F = tf.abs(tf.fft2d(tf.complex(tf_fftshift(self.airy_aprox_pad), 0.)))
        
        # Set regularization. Problem will be treated as l1 of spectral error at each depth + tau * l_inf(cross_corr)
        self.cross_corr_norm = cross_corr_norm
        self.logsumexp_param = tf.constant(1e-3, tf.float32)   #lower is better l-infinity approximation, higher is worse but smoother
        self.tau = tf.constant(5000.,tf.float32)    #Extra weight for cross correlation terms
        
        
        self.ignore_dc = True   #Ignore DC in autocorrelations when computing frequency domain loss
        dc_mask = np.ones_like(self.target_F.numpy())  #Mask for DC. Set to zero anywhere we want to ignore loss computation of power spectrum
        dc_mask[:3,:3] = 0
        dc_mask[-2:,:1] = 0
        dc_mask[:1,-2:] = 0
        dc_mask[-2:,-2:]= 0
        dc_mask = dc_mask * self.target_F.numpy()>(.001*np.max(self.target_F.numpy()))
        self.dc_mask = tf.constant(dc_mask,tf.float32)
        
        # Use Zernike aberrations with random initialization 
        if aberrations == True:
            zern_init = []
            for i in range(self.Nlenslets):
                #zern_list.append([0,0,0,0,0,0,0,0,0,0])

                aberrations = np.zeros(self.numzern)
                #aberrations = np.random.uniform(low=0, high = .02, size = (5))
                #aberrations[0]  = np.random.uniform(low=0, high = .02)
                #aberrations[1]  = np.random.uniform(low=0, high = .02)
                zern_init.append(aberrations)
            #zern_init2=np.zeros(self.numzern2)
            self.zernlist = tfe.Variable(zern_init, dtype='float32', name='zernlist')
            #,constraint=lambda t: tf.clip_by_value(t,-0.05, 0.05)
            #self.zernlist = tf.constant(zern_init, dtype='float32', name='zernlist')
            #self.zernlist=tf.Variable(zern_init, dtype='float32', name='zernlist')
        else:
            self.zernlist = 0
        
        
    def call(self, defocus_list):

        if np.size(defocus_list) == 1:
            defocus_list = 1./(np.linspace(1/self.zmin_virtual, 1./self.zmax_virtual, self.Nz)) #mm or dioptres
            
            #if self.zsampling is "uniform_random":
            #    self.defocus_list = 1/np.sort(np.random.uniform(low = 1/self.zmin_virtual, high = 1./self.zmax_virtual, size = (self.Nz,)))
            #
            #elif self.zsampling is "random_grid":
            #    testint = random.sample(range(0,model.grid_z_planes), model.Nz)
            #    self.defocus_list = self.defocus_grid[testint]
        #else:
        #    self.defocus_list = grid_opt
            
        T, aper,_ = make_lenslet_tf_zern(self) 
        #T,aper=make_lenslet_tf(self) #offset added
        # Get psf stack
        zstack = self.gen_psf_stack(T, aper, 0, defocus_list)
        
        psf_spect = self.gen_stack_spectrum(zstack)

        normsize=tf.to_float(tf.shape(psf_spect)[0]*tf.shape(psf_spect)[1])
        
        Rmat_tf_diag = []
        Rmat_tf_off_diag = []
        #calculating Xcorr
        
        
        # This now computes diagonals and off-diagonals separately then concatenates them. this makes is easier to "find" the diagonals/off diagonals for separate treatment  later.
        for z1 in range(self.Nz):
            for z2 in range(z1, self.Nz):
                Fcorr = tf.conj(psf_spect[z1])*psf_spect[z2]
                if z1 == z2:
                    # Difference between autocorrelation and target bandwidth
                    if self.ignore_dc:  
                        # Remove DC (assume no fftshift!)

                        Rmat_tf_diag.append(tf.reduce_sum(self.dc_mask * (tf.square((tf.abs(Fcorr) - tf.abs(self.target_F))/normsize))))
                        #Rmat_tf_diag.append(tf.reduce_sum( tf.linalg.inv(((tf.abs(Fcorr)))+0.000001)))

                    else:
                        acorr = tf.abs(tf.ifft2d(Fcorr))
                        Rmat_tf_diag.append(tf.reduce_sum(tf.square((acorr - self.target_F)))) #maybe get rid of negatives. 
                    
                else:
                    # Target is zero for cross correlation
                    if self.cross_corr_norm is 'l2':
                        ccorr = tf.abs(tf.ifft2d(Fcorr))
                        Rmat_tf_off_diag.append( self.tau * tf.sqrt(tf.reduce_sum(tf.square(ccorr))))  #changed to one norm
                    elif self.cross_corr_norm is 'log_sum_exp':   
                        # Implementation of eq. 7 from http://users.cecs.anu.edu.au/~yuchao/files/SmoothApproximationforL-infinityNorm.pdf
                        ccorr = tf.abs(tf.ifft2d(Fcorr))
                        Rmat_tf_off_diag.append(self.tau * self.logsumexp_param * tf.reduce_logsumexp(tf.square(ccorr)/self.logsumexp_param) )
                        
                    elif self.cross_corr_norm is 'inf':
                        Rmat_tf_off_diag.append( self.tau * tf.reduce_max(tf.abs(tf.ifft2d(Fcorr))))
        Rmat_tf = tf.concat([Rmat_tf_diag, Rmat_tf_off_diag],0)
                        
                        

        #Rmat=tf.reshape(Rmat_tf,(self.Nz,self.Nz))
            
        return Rmat_tf #note this returns int data type!! vector not matrix. This is also my loss!
    
    def gen_correlation_stack(self,psf_spect):
        Fcorr=[]
        for z1 in range(self.Nz):
            for z2 in range(self.Nz):
                Fcorr.append(tf_fftshift(tf.ifft2d(tf.conj(psf_spect[z1])*psf_spect[z2])))

        return Fcorr
        
    def gen_psf_stack(self, T, aper, prop_pad, zplanes=0):
        zstack = []
        
        if np.size(zplanes) == 1:
            zplanes = 1./(np.linspace(1/self.zmin_virtual, 1./self.zmax_virtual, self.Nz)) #mm or dioptres
            
        #if np.size(zplanes_opt) == 0:
        #    zplanes = self.defocus_list
        #else:
        #    zplanes = zplanes_opt   
        for z in range(0, len(zplanes)):
            
           # zstack.append(gen_psf_ag_tf(T,self,zplanes[z],'angle',0., prop_pad,Grin=self.Grin[z]))
            zstack.append(gen_psf_ag_tf(T,self,zplanes[z],'angle',0., prop_pad,Grin=0))

        return zstack
    
    def gen_stack_spectrum(self, zstack):
                #Padding for fft
        psf_pad=[]
#         Rmat = np.zeros((self.Nz,self.Nz))

        for z1 in range(self.Nz):
            psf_pad.append(pad_frac_tf(zstack[z1],self.corr_pad_frac)) #how much to pad? 

        psf_spect=[]
        
        #Getting spectrum

        for z1 in range(self.Nz):
            psf_spect.append(tf.fft2d(tf.complex(tf_fftshift(psf_pad[z1]),tf.constant(0.,dtype = tf.float32))))

        return psf_spect