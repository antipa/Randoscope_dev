import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
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

def poissonsampling_circular(model,factor=1.2,position_mode='random',radius_mode='random'):
    EAperR=model.CA-model.lenslet_CA
    xrng=[-(model.CA),model.CA] 
    yrng=xrng
    Rrng=[model.Rmin,model.Rmax]
    rlist = np.random.permutation(1/(np.linspace(1/model.Rmax, 1/model.Rmin, model.Nlenslets)))
    #rlist=np.random.permutation(np.array([ 6.75,  7.14,  8,  9, 9.53, 11.11, 13.5, 17.5,
     #  22.22,6.75,  7.14,  8,  9, 9.53, 11.11, 13.5, 17.5,
     #  22.22,6.75,  7.14,  8,  9, 9.53, 11.11, 13.5, 17.5,
     #  22.22,6.75,  7.14,  8,  9, 9.53, 11.11, 13.5, 17.5,
      # 22.22 ]))
    #rlist=rlist/2
    xpos = (xrng[1]-xrng[0])*np.random.rand(model.Nlenslets) + xrng[0]
    ypos = (yrng[1]-yrng[0])*np.random.rand(model.Nlenslets) + yrng[0]
    lens_curv = 1/rlist
    if position_mode=='random':
        while (np.sum((xpos**2+ypos**2)>EAperR)!=0):
            xpos = (xrng[1]-xrng[0])*np.random.rand(model.Nlenslets) + xrng[0]
            ypos = (yrng[1]-yrng[0])*np.random.rand(model.Nlenslets) + yrng[0]
        # poission disc sampling (kindof)
        step=model.lenslet_CA
        #check if points are too close and have them walk!
        i=0
        j=0
        while i<model.Nlenslets:
            while j<model.Nlenslets:
                if i!=j:
                    d=np.sqrt((xpos[i]-xpos[j])**2+(ypos[i]-ypos[j])**2)
                    if d<factor*model.lenslet_CA or xpos[j]**2+ypos[j]**2>EAperR**2 or xpos[i]**2+ypos[i]**2>EAperR**2:
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
    
    elif position_mode=='initial_nano':
        xpos=np.array([ 0.18578115,  0.74248238,  0.35555949,  0.10187365, -0.22157956,
       -0.5502,  0.29999206,  0.68421244,  0.11299734,  0.3125789 ,
        0.69618861, -0.28096646,  0.45888084, -0.27032854,  0.07697116,
       -0.5625567 , -0.58, -0.22702032,  0.07976543, -0.70372301,
       -0.34797056, -0.1152197 ,  0.41478314, -0.49736261,  0.39453689,
       -0.09447947, -0.08091086, -0.40003641,  0.04180626,  0.49347817,
        0.25394288,  0.02494959,  0.53552421,  0.28558491,  0.61067326,
       -0.72650046,-0.32])
        ypos=np.array([-0.44942685,  0.0187904 , -0.08720638,  0.7031471 ,  0.11111811,
        0.2784 ,  0.16797938, -0.22899178, -0.07601691,  0.67472988,
        0.23505869,  0.65642641,  0.52935778,  0.36883709,  0.1762651 ,
        0.01505985, -0.4, -0.69954257,  0.41217862, -0.2266678 ,
       -0.06487182,  0.51443979, -0.30910161,  0.51766252, -0.53589769,
       -0.11244721, -0.46289246, -0.56147136, -0.29425527,  0.2757895 ,
       -0.69100594, -0.64317962,  0.01663745,  0.40540164, -0.42807524,
        0.168447,-0.3  ])
    elif position_mode=='initial_nano_new':
        xpos=np.array([ 0.17578115,  0.7324824 ,  0.34555948,  0.09187365, -0.21157956,
                -0.5402    ,  0.28999206,  0.69421244,  0.10299734,  0.303706  ,
                 0.68618864, -0.2841991 ,  0.46888083, -0.27588776,  0.0767772 ,
                -0.5525567 , -0.57      , -0.22591965,  0.08791482, -0.693723  ,
                -0.33931562, -0.1052197 ,  0.40478313, -0.5051086 ,  0.4045369 ,
                -0.08447947, -0.09091086, -0.3900364 ,  0.05180626,  0.48347816,
                 0.24394289,  0.02650596,  0.5255242 ,  0.2755849 ,  0.62067324,
                -0.7252318 , -0.32522455])
        ypos=np.array([-0.43942684,  0.0087904 , -0.07720638,  0.7131471 ,  0.12111811,
                 0.2684    ,  0.15797938, -0.23899178, -0.06601691,  0.6847299 ,
                 0.24505869,  0.66086996,  0.53915256,  0.3686992 ,  0.18616977,
                 0.02505985, -0.41      , -0.7095426 ,  0.40217862, -0.2166678 ,
                -0.05666433,  0.5044398 , -0.29910162,  0.5276625 , -0.54589766,
                -0.10244721, -0.45289245, -0.57147133, -0.28425527,  0.2657895 ,
                -0.70100594, -0.6331796 ,  0.00764467,  0.40803543, -0.42228377,
                 0.17431541, -0.30963156])
    elif position_mode=='initial':
        if model.Nlenslets==36:
            xpos=np.array([ 0.4131344 ,  0.07026076, -0.36032027, -0.2950179 ,  0.6399725 ,
       -0.5207495 ,  0.06639417,  0.04873332, -0.38358852, -0.04776596,
       -0.5954222 ,  0.4216355 , -0.67778796, -0.1588442 ,  0.23223598,
       -0.2688033 ,  0.43465754,  0.2339625 , -0.1746701 ,  0.28348893,
        0.4805356 , -0.6730768 , -0.46448663,  0.0775789 ,  0.6725761 ,
       -0.26156783,  0.5729606 , -0.00260094,  0.4098965 , -0.20852783,
        0.23989761, -0.49635968, -0.11971747, -0.10505962,  0.03983941,
        0.29504523])
            ypos=np.array([-0.56124103, -0.26532307, -0.3102669 ,  0.64471316,  0.1481013 ,
        0.11668031, -0.4548191 ,  0.19978055,  0.44213313, -0.68001646,
       -0.37753585,  0.28509444, -0.08660459, -0.31030378,  0.004179  ,
        0.15873532, -0.03099893, -0.65588766, -0.51921844,  0.43212706,
       -0.38039157,  0.26715028, -0.54251754,  0.42332056, -0.11268332,
       -0.68524253,  0.39293897,  0.6608805 , -0.2135221 , -0.05516782,
        0.661144  , -0.12468256,  0.2905781 ,  0.4744946 , -0.08732768,
       -0.3712923 ])
        else:
            xpos=np.array([ 0.44763693,  0.32069656,  0.17007685,  0.55496037, -0.42755082,
            0.5855074 ,  0.49686506, -0.01771884, -0.6458085 , -0.40372765,
            0.3708608 , -0.25112858,  0.17999306,  0.01989856, -0.09181889,
            0.35329962,  0.11995778, -0.6139183 , -0.19938703, -0.41797048,
            0.71391004,  0.577655  ,  0.11036615, -0.40891695, -0.19601497,
           -0.15551926,  0.00748745, -0.28095615,  0.24532567, -0.6450773 ,
            0.3864855 ])
            ypos=np.array([ 5.5220222e-01, -4.2501873e-01, -7.0745277e-01,  3.7471771e-01,
           -1.0441011e-01,  6.9850333e-02, -2.0334278e-01,  5.0833023e-01,
           -1.8442065e-01,  3.7859333e-01,  2.1210821e-01,  3.5482556e-02,
           -1.7035936e-01, -3.6299065e-01, -5.8999467e-01, -6.1155687e-04,
            2.5758258e-01,  1.3219318e-01, -2.4258418e-01, -4.7758019e-01,
           -1.9103232e-01, -4.4472083e-01,  6.6134125e-01,  6.0649139e-01,
            2.8008059e-01,  6.9886565e-01,  6.3269727e-02, -6.4446259e-01,
            5.1393324e-01,  3.6000755e-01, -6.0905510e-01])
    elif position_mode=='initial_ballbearings':
        xpos=np.array([-0.36,  0.43, -0.17,  0.08,  0.07, -0.67,  0.57,  0.04,  0.41,
         0.05, -0.38,  0.67, -0.52,  0.28, -0.11,  0.3 , -0.05, -0.16,
         0.  , -0.21,  0.64,  0.48, -0.46, -0.12,  0.07, -0.6 ,  0.41,
        -0.5 , -0.3 ,  0.42,  0.23, -0.27, -0.68,  0.23, -0.26,  0.24])
                                      
        ypos=np.array([-0.31, -0.03, -0.52,  0.42, -0.27,  0.27,  0.39, -0.09, -0.56,
        0.2 ,  0.44, -0.11,  0.12,  0.43,  0.47, -0.37, -0.68, -0.31,
        0.66, -0.06,  0.15, -0.38, -0.54,  0.29, -0.45, -0.38, -0.21,
       -0.12,  0.64,  0.29,  0.  ,  0.16, -0.09, -0.66, -0.69,  0.66])
    if radius_mode=='initial':
       
        if model.Nlenslets==36:
            rlist=np.array([ 6.75 ,  8.75 , 11.11 ,  3.57 ,  4.5  ,  5.555,  4.   ,  6.75 ,
        6.75 ,  4.765,  4.   ,  3.57 ,  3.375,  4.765,  3.57 ,  3.57 ,
       11.11 ,  3.375, 11.11 ,  5.555,  4.5  ,  8.75 ,  4.5  , 11.11 ,
        6.75 ,  3.375,  8.75 ,  4.765,  4.   ,  4.765,  3.375,  4.   ,
        4.5  ,  5.555,  8.75 ,  5.555])
        elif model.Nlenslets==31:
             rlist=np.array([ 3.9643483,  4.093112 ,  5.783875 ,  9.854543 ,  3.16725  ,
        7.981981 ,  5.5300603,  7.0845137,  8.521749 ,  4.5350084,
        6.062109 ,  4.886726 , 12.875    ,  4.7043023,  3.5212533,
        3.248906 ,  4.377476 ,  9.139813 ,  5.0838685,  5.297586 ,
        7.5065174,  3.6224742,  3.7296867, 10.690537 ,  6.707433 ,
       11.681522 ,  3.4255354,  3.843439 ,  3.3348835,  6.368465 ,
        4.2305207])
    elif radius_mode=='initial_nano':
        rlist=np.array([4.000442 , 4.440296 , 3.7237842, 3.5596673, 3.2063835, 5.4982405,
       4.2089086, 3.0839548, 5.3172565, 3.1439776, 6.364798 , 3.4829168,
       2.767    , 4.3215075, 4.839368 , 4.988825 , 2.8152227, 5.1478076,
       3.409406 , 6.9092703, 4.6986055, 3.2713168, 5.691979 , 4.5658   ,
       4.1020284, 2.970532 , 6.625868 , 7.218    , 3.8116512, 5.8998694,
       2.865156 , 3.6398768, 3.9037654, 2.9168925, 3.3389342, 3.026181 ,
       6.123522 ])
    elif radius_mode=='initial_nano_new':
        
        
        rlist=np.array([5.691093 , 6.5581717, 6.9837832, 6.7642894, 6.364244 , 3.27     ,
        3.787973 , 3.5388365, 3.659169 , 3.4261663, 4.408728 , 3.597997 ,
        3.7224572, 4.911805 , 4.597064 , 3.3724794, 7.218    , 4.697398 ,
        4.235216 , 5.544482 , 3.48159  , 6.181456 , 6.008875 , 4.074845 ,
        5.4052353, 3.8558364, 3.3204494, 4.3202305, 5.1467204, 4.8022094,
        5.8456683, 3.9261758, 3.999129 , 4.500927 , 5.0265193, 5.272811 ,
        4.153483 ])
#         rlist=np.array([3.9261758, 4.8022094, 3.999129 , 4.597064 , 4.074845 , 3.48159  ,
#         7.218    , 6.364244 , 3.5388365, 3.787973 , 4.697398 , 4.408728 ,
#         3.27     , 6.7642894, 6.9837832, 3.4261663, 6.181456 , 5.1467204,
#         5.272811 , 4.153483 , 4.500927 , 3.8558364, 6.008875 , 5.691093 ,
#         3.7224572, 5.8456683, 4.911805 , 3.659169 , 4.3202305, 3.597997 ,
#         4.235216 , 3.3724794, 5.4052353, 5.0265193, 5.544482 , 3.3204494,
#         6.5581717])
        
        
        
#         rlist=[5.467153 , 5.8581676, 7.4584723, 6.835998 , 5.1250706, 8.638365 ,
#        3.3456757, 4.969596 , 3.093    , 4.099132 , 3.4154296, 3.4881542,
#        4.823276 , 8.205667 , 3.9990652, 4.431818 , 6.562163 , 4.5550475,
#        6.075427 , 9.656803 , 3.278714 , 3.9037673, 3.812906 , 3.6433077,
#        4.685326 , 6.3094215, 3.564043 , 4.3150806, 4.204335 , 5.6559105,
#        5.290588 , 3.21438  , 7.8142486, 3.726178 , 3.152522 , 7.133682 ,
#        9.119237 ]
    elif radius_mode=='initial_ballbearings':
        rlist=np.array([11.11, 11.11, 11.11, 11.11,  8.75,  8.75,  8.75,  8.75,  6.75,
        6.75,  6.75,  6.75,  5.56,  5.56,  5.56,  5.56,  4.77,  4.77,
        4.77,  4.77,  4.5 ,  4.5 ,  4.5 ,  4.5 ,  4.  ,  4.  ,  4.  ,
        4.  ,  3.57,  3.57,  3.57,  3.57,  3.38,  3.38,  3.38,  3.38])
    elif radius_mode=='in_out':
        rlista=np.sort(rlist)
        posFromCenter=np.sqrt(xpos**2+ypos**2)
        Index = np.argsort(posFromCenter)
        rlist[Index]=rlista
    elif radius_mode=='out_in':
        rlista=np.sort(rlist)
        rlista[::-1].sort()
        posFromCenter=np.sqrt(xpos**2+ypos**2)
        Index = np.argsort(posFromCenter)
        rlist[Index]=rlista
    elif radius_mode=='half_in_out':
        rlistt=np.sort(rlist)
        rlista=rlistt[0:len(rlistt)//2]
        rlistd=rlistt[len(rlistt)//2:len(rlistt)]
        rliste = np.empty((len(rlista)+len(rlistd)))
        rliste[::2] = rlistd
        rliste[1::2] = rlista
        posFromCenter=np.sqrt(xpos**2+ypos**2)
        Index = np.argsort(posFromCenter)
#         Indexd= np.argsort(posFromCenter)
#         Indexd[::-1].sort()
#         Index = np.empty((len(Indexa)+len(Indexd)),dtype=int)
#         Index[::2] = Indexd
#         Index[1::2] = Indexa
#         Index=Index[0:len(rlist)]
        rlist[Index]=rliste 
    elif radius_mode=='half_in_out_quad':
        rlistt=np.sort(rlist)
        rlista=rlistt[0:len(rlistt)//2]
        rlistd=rlistt[len(rlistt)//2:len(rlistt)]
        rliste = np.empty((len(rlista)+len(rlistd)))
        rliste[::2] = rlistd
        rliste[1::2] = rlista
        posFromCenter=np.sqrt((xpos-1.8)**2+(ypos-1.8)**2)
        Index = np.argsort(posFromCenter)
#         Indexd= np.argsort(posFromCenter)
#         Indexd[::-1].sort()
#         Index = np.empty((len(Indexa)+len(Indexd)))
#         Index[::2] = Indexd
#         Index[1::2] = Indexa
        rlist[Index]=rliste 
    return xpos, ypos, rlist






def make_lenslet_tf_zern(model):
    T = tf.zeros([model.samples[0],model.samples[1]])
    aper1 = tf.sqrt(model.xgm**2+model.ygm**2) <= model.CA
    aper=tf.to_float(aper1)
    if np.shape(model.zernlist) != (): 
        T_orig = tf.zeros([model.samples[0],model.samples[1]])
    else:
        T_orig = []
    
    for n in range(model.Nlenslets):
        sph1 = model.lenslet_offset[n]+tf.real(tf.sqrt(tf.square(model.rlist[n]) - tf.square((model.xgm- model.xpos[n]))- tf.square((model.ygm-model.ypos[n])))
                                          )-tf.real(tf.sqrt(tf.square(model.rlist[n])-tf.square(model.mean_lenslet_CA[n])))
        
        if np.shape(model.zernlist) != ():  # Including Zernike aberrations 
            # change to normalize by CA
            Z = zernikecartesian(model.zernlist[n],  model.xnorm - model.xpos[n]/np.max(model.xgm)  ,model.ynorm - model.ypos[n]/np.max(model.xgm))
            
            #r = 1. # Not sure about this
            #Z[(model.xnorm  - model.xpos[n]/np.max(model.xgm) )**2+(model.ynorm - model.ypos[n]/np.max(model.xgm))**2>r**2]=0 # Crop to be a circle
            sph2 = sph1 + Z
            
            
            T_orig = tf.maximum(T_orig,sph1)
            T = tf.maximum(T,sph2)
        else:
            T = tf.maximum(T,sph1)
    #Z2 = (zernikecartesian2(model.zernlist2,  model.xnorm  ,model.ynorm ))
    #T=T+Z2*aper    
        
    return T, aper, T_orig


def zernikecartesian(coefficient,x,y):
    """
    ------------------------------------------------
    __zernikecartesian__(coefficient,x,y):

    Return combined aberration

    Zernike Polynomials Caculation in Cartesian coordinates

    coefficient: Zernike Polynomials Coefficient from input
    x: x in Cartesian coordinates
    y: y in Cartesian coordinates
    ------------------------------------------------
    """
    Z = coefficient
    #Z = [0]+coefficient
    r = tf.sqrt(tf.square(x) + tf.square(y))
    #Z1  =  Z[0]
    Z2  =  Z[0]  * 2.*x
    Z3  =  Z[1]  * 2.*y
    #Z4  =  Z[1]  * tf.sqrt(3.)*(2.*tf.square(r)-1.)
    #Z5  =  Z[0]  * 2.*tf.sqrt(6.)*x*y
    #Z6  =  Z[1]  * tf.sqrt(6.)*(x**2-y**2)
    #Z7  =  Z[2]  * tf.sqrt(8.)*y*(3*r**2-2)
    #Z8  =  Z[2]  * tf.sqrt(8.)*x*(3*r**2-2)
    #Z9  =  Z[6]  * tf.sqrt(8.)*y*(3*x**2-y**2)
    #Z10 =  Z[7] * tf.sqrt(8.)*x*(x**2-3*y**2)
    #Z11 =  Z[11] * __sqrt__(5)*(6*r**4-6*r**2+1)
    #Z12 =  Z[4] * tf.sqrt(10.)*(x**2-y**2)*(4*r**2-3)
    #Z13 =  Z[5] * 2*tf.sqrt(10.)*x*y*(4*r**2-3)
    #Z14 =  Z[14] * __sqrt__(10)*(r**4-8*x**2*y**2)
    #Z15 =  Z[15] * 4*__sqrt__(10)*x*y*(x**2-y**2)
    #Z16 =  Z[16] * __sqrt__(12)*x*(10*r**4-12*r**2+3)
    #Z17 =  Z[17] * __sqrt__(12)*y*(10*r**4-12*r**2+3)
    #Z18 =  Z[18] * __sqrt__(12)*x*(x**2-3*y**2)*(5*r**2-4)
    #Z19 =  Z[19] * __sqrt__(12)*y*(3*x**2-y**2)*(5*r**2-4)
    #Z20 =  Z[20] * __sqrt__(12)*x*(16*x**4-20*x**2*r**2+5*r**4)
    #Z21 =  Z[21] * __sqrt__(12)*y*(16*y**4-20*y**2*r**2+5*r**4)
    #Z22 =  Z[22] * __sqrt__(7)*(20*r**6-30*r**4+12*r**2-1)
    #Z23 =  Z[6] * 2*tf.sqrt(14.)*x*y*(15*r**4-20*r**2+6)
    #Z24 =  Z[7] * tf.sqrt(14.)*(x**2-y**2)*(15*r**4-20*r**2+6)
    #Z25 =  Z[25] * 4*__sqrt__(14)*x*y*(x**2-y**2)*(6*r**2-5)
    #Z26 =  Z[26] * __sqrt__(14)*(8*x**4-8*x**2*r**2+r**4)*(6*r**2-5)
    #Z27 =  Z[27] * __sqrt__(14)*x*y*(32*x**4-32*x**2*r**2+6*r**4)
    #Z28 =  Z[28] * __sqrt__(14)*(32*x**6-48*x**4*r**2+18*x**2*r**4-r**6)
    #Z29 =  Z[29] * 4*y*(35*r**6-60*r**4+30*r**2-4)
    #Z30 =  Z[30] * 4*x*(35*r**6-60*r**4+30*r**2-4)
    #Z31 =  Z[31] * 4*y*(3*x**2-y**2)*(21*r**4-30*r**2+10)
    #Z32 =  Z[32] * 4*x*(x**2-3*y**2)*(21*r**4-30*r**2+10)
    #Z33 =  Z[33] * 4*(7*r**2-6)*(4*x**2*y*(x**2-y**2)+y*(r**4-8*x**2*y**2))
    #Z34 =  Z[34] * (4*(7*r**2-6)*(x*(r**4-8*x**2*y**2)-4*x*y**2*(x**2-y**2)))
    #Z35 =  Z[35] * (8*x**2*y*(3*r**4-16*x**2*y**2)+4*y*(x**2-y**2)*(r**4-16*x**2*y**2))
    #Z36 =  Z[36] * (4*x*(x**2-y**2)*(r**4-16*x**2*y**2)-8*x*y**2*(3*r**4-16*x**2*y**2))
    #Z37 =  Z[37] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1)
    ZW = Z2+Z3#+Z7+Z8#+Z12+Z13+Z23+Z24
    #ZW =     Z1 + Z2 +  Z3+  Z4+  Z5+  Z6#+  #Z7+  Z8+  Z9+ Z10
    #+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+ \
            #Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+ \
            #Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
    return ZW
def zernikecartesian2(coefficient,x,y):
    """
    ------------------------------------------------
    __zernikecartesian__(coefficient,x,y):

    Return combined aberration

    Zernike Polynomials Caculation in Cartesian coordinates

    coefficient: Zernike Polynomials Coefficient from input
    x: x in Cartesian coordinates
    y: y in Cartesian coordinates
    ------------------------------------------------
    """
    Z = coefficient
    #Z = [0]+coefficient
    r = tf.sqrt(tf.square(x) + tf.square(y))
    #Z1  =  Z[0]
    Z2  =  Z[0]  * 2.*x
    Z3  =  Z[1]  * 2.*y
    Z4  =  Z[2]  * tf.sqrt(3.)*(2.*tf.square(r)-1.)
    Z5  =  Z[3]  * 2.*tf.sqrt(6.)*x*y
    Z6  =  Z[4]  * tf.sqrt(6.)*(x**2-y**2)
    Z7  =  Z[5]  * tf.sqrt(8.)*y*(3*r**2-2)
    Z8  =  Z[6]  * tf.sqrt(8.)*x*(3*r**2-2)
#    Z9  =  Z[4]  * tf.sqrt(8.)*y*(3*x**2-y**2)
#    Z10 =  Z[5] * tf.sqrt(8.)*x*(x**2-3*y**2)
#     Z11 =  Z[9] * tf.sqrt(5.)*(6*r**4-6*r**2+1)
#     Z12 =  Z[10] * tf.sqrt(10.)*(x**2-y**2)*(4*r**2-3)
#     Z13 =  Z[11] * 2*tf.sqrt(10.)*x*y*(4*r**2-3)
#     Z14 =  Z[12] * tf.sqrt(10.)*(r**4-8*x**2*y**2)
#     Z15 =  Z[13] * 4*tf.sqrt(10.)*x*y*(x**2-y**2)
#     Z16 =  Z[14] * tf.sqrt(12.)*x*(10*r**4-12*r**2+3)
#     Z17 =  Z[15] * tf.sqrt(12.)*y*(10*r**4-12*r**2+3)
#     Z18 =  Z[16] * tf.sqrt(12.)*x*(x**2-3*y**2)*(5*r**2-4)
#     Z19 =  Z[17] * tf.sqrt(12.)*y*(3*x**2-y**2)*(5*r**2-4)
#     Z20 =  Z[18] * tf.sqrt(12.)*x*(16*x**4-20*x**2*r**2+5*r**4)
#     Z21 =  Z[19] * tf.sqrt(12.)*y*(16*y**4-20*y**2*r**2+5*r**4)
#     Z22 =  Z[20] * tf.sqrt(7.)*(20*r**6-30*r**4+12*r**2-1)
#     Z23 =  Z[21] * 2*tf.sqrt(14.)*x*y*(15*r**4-20*r**2+6)
#     Z24 =  Z[22] * tf.sqrt(14.)*(x**2-y**2)*(15*r**4-20*r**2+6)
#     Z25 =  Z[23] * 4*tf.sqrt(14.)*x*y*(x**2-y**2)*(6*r**2-5)
#     Z26 =  Z[24] * tf.sqrt(14.)*(8*x**4-8*x**2*r**2+r**4)*(6*r**2-5)
#     Z27 =  Z[25] * tf.sqrt(14.)*x*y*(32*x**4-32*x**2*r**2+6*r**4)
#     Z28 =  Z[26] * tf.sqrt(14.)*(32*x**6-48*x**4*r**2+18*x**2*r**4-r**6)
#     Z29 =  Z[27] * 4*y*(35*r**6-60*r**4+30*r**2-4)
#     Z30 =  Z[28] * 4*x*(35*r**6-60*r**4+30*r**2-4)
#     Z31 =  Z[29] * 4*y*(3*x**2-y**2)*(21*r**4-30*r**2+10)
#     Z32 =  Z[30] * 4*x*(x**2-3*y**2)*(21*r**4-30*r**2+10)
#     Z33 =  Z[31] * 4*(7*r**2-6)*(4*x**2*y*(x**2-y**2)+y*(r**4-8*x**2*y**2))
#     Z34 =  Z[32] * (4*(7*r**2-6)*(x*(r**4-8*x**2*y**2)-4*x*y**2*(x**2-y**2)))
#     Z35 =  Z[33] * (8*x**2*y*(3*r**4-16*x**2*y**2)+4*y*(x**2-y**2)*(r**4-16*x**2*y**2))
#     Z36 =  Z[34] * (4*x*(x**2-y**2)*(r**4-16*x**2*y**2)-8*x*y**2*(3*r**4-16*x**2*y**2))
#     Z37 =  Z[35] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1)
    #ZW = Z5+Z
    ZW =  Z2+Z3+Z4+ Z5+  Z6+  Z7+  Z8 #+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18#+ Z19+Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28#+ Z29+Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
    return ZW
def make_lenslet_tf(model):
        T = tf.zeros([model.samples[0],model.samples[1]])
        for n in range(model.Nlenslets):
            sph1 = model.lenslet_offset[n]+tf.real(tf.sqrt(tf.square(model.rlist[n]) - tf.square((model.xgm-
                                                                          model.xpos[n])) - tf.square((model.ygm-model.ypos[n]))))-tf.real(tf.sqrt(tf.square(model.rlist[n]
                                                                          )-tf.square(model.mean_lenslet_CA)))
            T = tf.maximum(T,sph1)
        aper = tf.sqrt(model.xgm**2+model.ygm**2) <= model.CA
        return T,aper
    
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
        Fx, Fy = tf.meshgrid(tf.lin_space(tf.reduce_min(model.Fx), tf.reduce_max(model.Fx), U.shape[0]), tf.lin_space(tf.reduce_min(model.Fy), tf.reduce_max(model.Fy), U.shape[1]))
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

def gen_psf_ag_tf(T,model,z_dis, obj_def, pupil_phase=0, prop_pad = 0,Grin=0):  #check negatives in exponents
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
    amp = tf.to_float(tf.sqrt(tf.square(model.xgm) + tf.square(model.ygm)) <= model.CA)
    U_prop = propagate_field_freq_tf(model, tf.complex(tf.real(U_out)*amp,tf.imag(U_out)*amp), prop_pad)    
    psf = tf.square(tf.abs(U_prop))
    return(psf/tf.sqrt(tf.reduce_sum(tf.square(psf)))) #DO WE NEED TO DO THIS????
    


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
    return tf.reduce_sum(tf.square(Rmat)), Rmat

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

def project_to_aper_keras(model):
    lenslets_dist = tf.sqrt(tf.square(model.xpos) + tf.square(model.ypos))
   # print(lenslets_dist)
    dist_new = tf.minimum(lenslets_dist, model.CA-model.mean_lenslet_CA-tf.sqrt(2*model.rlist*model.lenslet_offset-tf.square(model.lenslet_offset)))
    
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
        scipy.io.savemat('/media/hongdata/Kristina/MiniscopeData/best_init.mat', dict_best)
        scipy.io.savemat('/media/hongdata/Kristina/MiniscopeData/worst_init.mat', dict_worst)

       
    
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
