from __future__ import print_function
import os
import h5py
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.interpolate import interp1d, interp2d,interpn,RegularGridInterpolator,griddata
from numpy import cos, sin, pi
from tvtk.api import tvtk
#from dyn_to_seis.seis1d import prem
from seis1d import prem
#from mayavi.scripts import mayavi2

prem = prem()
DIR = os.path.dirname(os.path.abspath(__file__))
lookup_tables=DIR+'/data/lookup_tables.hdf5'
adiabats_dir=DIR+'/data/LargeSetPyroliteAdiabats'

def gen_adiabat_interpolator(adiabat_dir,plot=False,debug=False):
    '''
    used in conversion from potential temperature to absolute
    temperature.  
    '''

    p = []
    pot_T = []
    abs_T = []

    ads = glob.glob(adiabat_dir+'/*.out')
    print(ads)

    for ad in ads:
        pot_temp_str = ad.split('_')[-1].split('K')[0]
        pot_temp = int(pot_temp_str)
        f = np.loadtxt(ad)
        temp = f[:,0]
        pressure = f[:,1]
        for i in range(0,len(temp)):
            p.append(pressure[i])
            pot_T.append(pot_temp)
            abs_T.append(temp[i])

    #first interpolate irregular scatter data onto uniform mesh
    x_i = np.arange(0,2550,50) #adiabats above 2500 are nonsense
    y_i = np.linspace(0,1350000,1000)
    z_i = griddata((pot_T,p),abs_T, (x_i[None,:],y_i[:,None]), method='linear')

    if debug:
        print('min z_i:',np.min(z_i))
        print('max z_i:',np.max(z_i))

    if plot:
       #plt.scatter(pot_T,p,c=abs_T,cmap='viridis',edgecolor='none')
       #plt.show()
       fig = plt.figure(figsize=[6.5,4])
       plt.contourf(x_i,y_i,z_i)
       plt.colorbar(label='absolute T (K)')
       plt.xlabel('potential T (K)')
       plt.ylabel('pressure (bar)')
       plt.xlim([300,3000])
       plt.ylim([0,1350000])
       plt.tight_layout()
       plt.show()

    #next make an interpolating function
    rgi = RegularGridInterpolator(points=(x_i,y_i),values=z_i.T,
            bounds_error=False,fill_value=None)
    #rgi takes a tuple of (potential temp, pressure) and returns absolute temp

    return rgi

def cart2polar(x,y):
   '''
   Given x,y in cartesian coordinates, returns r, theta
   '''

   r = np.sqrt(x**2+y**2)
   theta = np.arctan2(y,x)

   return r,theta

def polar2cart(radius,theta,theta_deg=True):
   '''
   takes polar coordinates and returns cartesian.theta should be in degree
   '''
   if theta_deg == True:
       theta = np.radians(theta)

   x = radius*np.cos(theta)
   y = radius*np.sin(theta)
   return(x,y)


def rotation_matrix(n,phi):

  """ compute rotation matrix
  input: rotation angle phi [deg] and rotation vector n normalised to 1
  return: rotation matrix
  """

  phi=np.pi*phi/180.0

  A=np.array([ (n[0]*n[0],n[0]*n[1],n[0]*n[2]), (n[1]*n[0],n[1]*n[1],n[1]*n[2]), (n[2]*n[0],n[2]*n[1],n[2]*n[2])])
  B=np.eye(3)
  C=np.array([ (0.0,-n[2],n[1]), (n[2],0.0,-n[0]), (-n[1],n[0],0.0)])

  R=(1.0-np.cos(phi))*A+np.cos(phi)*B+np.sin(phi)*C

  return np.matrix(R)

def rotate_coordinates(n,phi,colat,lon):

  """ rotate colat and lon
  input: rotation angle phi [deg] and rotation vector n normalised to 1, original colatitude and longitude [deg]
  return: colat_new [deg], lon_new [deg]
  """

  # convert to radians

  colat=np.pi*colat/180.0
  lon=np.pi*lon/180.0

  # rotation matrix

  R=rotation_matrix(n,phi)

  # original position vector

  #x=np.matrix([[np.cos(lon)*np.sin(colat)], [np.sin(lon)*np.sin(colat)], [np.cos(colat)]])
  x=np.array([[np.cos(lon)*np.sin(colat)], [np.sin(lon)*np.sin(colat)], [np.cos(colat)]])

  # rotated position vector

  #y=R*x
  y=R.dot(x)

  # compute rotated colatitude and longitude

  colat_new=np.arccos(y[2])
  lon_new=np.arctan2(y[1],y[0])

  return float(180.0*colat_new/np.pi), float(180.0*lon_new/np.pi)

class model_1d(object):

    def __init__(self,depthmin=0.0,depthmax=2891.0,ddepth=1.0,f=0.18):
        self.depth = np.arange(depthmin,depthmax+ddepth,ddepth)
        self.T = np.zeros(len(self.depth))
        self.f = np.zeros(len(self.depth))
        self.npts_depth = len(self.depth)
        self.vp = np.zeros(len(self.depth))
        self.vs = np.zeros(len(self.depth))
        self.rho = np.zeros(len(self.depth))
        self.rho_rel = np.zeros(len(self.depth))
        self.dvo_rel = np.zeros(len(self.depth))
        self.dvo_rel = np.zeros(len(self.depth))
        self.rho_abs = np.zeros(len(self.depth))
        self.dvo_abs = np.zeros(len(self.depth))
        self.dvo_abs = np.zeros(len(self.depth))
        self.drho = np.zeros(len(self.depth))
        self.ddepth = ddepth
        self.depthmin = depthmin
        self.depthmax = depthmax

        #get 1d pressure profile from PREM
        self.P = prem.get_p(self.depth)
        self.P /= 1e5 #convert to bar?

        #by default, use pyrolite composition
        self.f[:] = 0.18

    def read_adiabat(self,adiabat_file):
        adiabat = np.loadtxt(adiabat_file)
        T_adiabat = adiabat[:,0]
        P_adiabat = adiabat[:,1]
        interp_T = interp1d(P_adiabat,T_adiabat,fill_value='extrapolate')
        T_int = interp_T(self.P)
        self.T = T_int[:]

    def change_temp(self,dep1,dep2,delta_T):
        '''
        args-
        dep1: depth at the top of the layer
        dep2: depth at the bottom of the layer
        delta_T: temperature perturbation within the layer
        '''
        i_1 = int(dep1 / self.ddepth) + int(self.depthmin/self.ddepth)
        i_2 = int(dep2 / self.ddepth) + int(self.depthmin/self.ddepth)
        self.T[i_1:i_2] += delta_T

    def change_comp(self,dep1,dep2,f):
        '''
        args-
        dep1: depth at the top of the layer
        dep2: depth at the bottom of the layer
        f: basalt fraction of the layer
        '''
        i_1 = int(dep1 / self.ddepth) + int(self.depthmin/self.ddepth)
        i_2 = int(dep2 / self.ddepth) + int(self.depthmin/self.ddepth)
        self.f[i_1:i_2] = f

    def velocity_conversion(self,composition,lookup_tables=lookup_tables):
        '''
        performs the velocity conversion.
        args:

           lookup_tables: str
                          path to h5py lookup tables
           composition: str
                        one of either 'pyrolite', 'harzburgite', morb', or, 'mixture'
                        'mixture' uses a mechanical mixture of morb and harzburgite
        '''

        tables = h5py.File(lookup_tables,'r')
        #read harzburgite lookuptable
        harz_vp_table = tables['harzburgite']['vp'][:]
        harz_vs_table = tables['harzburgite']['vs'][:]
        harz_rho_table = tables['harzburgite']['rho'][:]
        #read morb lookuptable
        morb_vp_table = tables['morb']['vp'][:]
        morb_vs_table = tables['morb']['vs'][:]
        morb_rho_table = tables['morb']['rho'][:]
        #read topology
        P0 = tables['pyrolite']['P0'].value
        T0 = tables['pyrolite']['T0'].value
        nP = tables['pyrolite']['nP'].value
        nT = tables['pyrolite']['nT'].value
        dP = tables['pyrolite']['dP'].value
        dT = tables['pyrolite']['dT'].value
        P_table_axis = np.arange(P0,((nP-1)*dP+P0+dP),dP)
        T_table_axis = np.arange(T0,((nT-1)*dT+T0+dT),dT)
        #create interpolators
        harz_vp_interpolator  = interp2d(P_table_axis,T_table_axis, harz_vp_table)
        harz_vs_interpolator  = interp2d(P_table_axis,T_table_axis, harz_vs_table)
        harz_rho_interpolator = interp2d(P_table_axis,T_table_axis, harz_rho_table)
        morb_vp_interpolator  = interp2d(P_table_axis,T_table_axis, morb_vp_table)
        morb_vs_interpolator  = interp2d(P_table_axis,T_table_axis, morb_vs_table)
        morb_rho_interpolator = interp2d(P_table_axis,T_table_axis, morb_rho_table)

        for i in range(0,len(self.depth)):
            P_here = self.P[i]
            T_here = self.T[i]
            f_here = self.f[i]
            vp_harz = harz_vp_interpolator(P_here,T_here)
            vs_harz = harz_vs_interpolator(P_here,T_here)
            rho_harz = harz_rho_interpolator(P_here,T_here)
            vp_morb = morb_vp_interpolator(P_here,T_here)
            vs_morb = morb_vs_interpolator(P_here,T_here)
            rho_morb = morb_rho_interpolator(P_here,T_here)
            self.vp[i] = f_here*vp_morb + (1-f_here)*vp_harz
            self.vs[i] = f_here*vs_morb + (1-f_here)*vs_harz
            self.rho[i] = f_here*rho_morb + (1-f_here)*rho_harz

    def plot(self):
        fig,axes=plt.subplots(1,3,sharey=True)
        axes[0].plot(self.T,self.depth)
        axes[0].set_ylim([self.depthmax,self.depthmin])
        axes[0].set_ylabel('depth (km)')
        axes[0].set_xlabel('T')
        axes[1].plot(self.P,self.depth)
        axes[1].set_ylim([self.depthmax,self.depthmin])
        axes[1].set_xlabel('P')
        axes[2].plot(self.f,self.depth)
        axes[2].set_ylim([self.depthmax,self.depthmin])
        axes[2].set_xlabel('f (%)')
        plt.tight_layout()
        plt.show()

class model_2d(object):
    def init(self):
        self.depth = np.zeros(1)
        self.theta = np.zeros(1)
        self.T = np.zeros(1)
        self.f = np.zeros(1)
        self.x = np.zeros(1)
        self.y = np.zeros(1)
        self.P = np.zeros(1)
        self.dtheta = np.zeros(1)
        self.ddepth = np.zeros(1)
        self.theta_axis = np.zeros(1)
        self.depth_axis = np.zeros(1)
        self.T_array = np.zeros((1,1))
        self.f_array = np.zeros((1,1))
        self.npts_theta = 0
        self.npts_depth = 0
        self.fmt = 'na'

    def read(self,path_to_file,fmt='pvk',**kwargs):
        '''
        reads in temperature and composition data from a cylindrical geodynamic model

        args:

          path_to_file: str
             path to input file

          theta_max_deg: float (optional)
             specifies maximum polar angle in degrees. default = 180 degrees

          npts_rad: int (optional)
             number of points in radius. default = 

          npts_theta: int (optional).
             number of points in theta. default = 

          fmt: str (optional)
             format of the input file. options: only 'pvk' right now
             if 'pvk':
                x(km), y(km), co-lat(degrees), depth(km), Tpot(C), conc. of basalt tracers
             if 'plume':
                x(km), y(km), T (C), dT(C)

        kwargs:

           bf_scaling: float (optional, default = 8.0)
              value corresponding to 100% basalt concentration in dynamic simulations.
              composition field will be normalized so that 1.0 corresponds to 100% basalt.

           plume_f = float (default = 0.18 (i.e., pyrolite))
              define a constant composition for the plume (use if fmt='plume')

           debug = if True, writes additional output diagnostics (default = False)
        '''
        bf_scaling = kwargs.get('bf_scaling',8.0)
        plume_f = kwargs.get('plume_f',0.18)
        debug = kwargs.get('debug',False)

        model = np.loadtxt(path_to_file)
        if fmt=='pvk':
           x = model[:,0]
           y = model[:,1]
           theta = model[:,2]
           depth = model[:,3]
           Tpot = model[:,4]
           f = model[:,5]
           self.fmt='pvk'
        elif fmt=='pvk_full':
           x = model[:,0]
           y = model[:,1]
           Tpot = model[:,2]
           f = model[:,3]
           self.fmt='pvk_full'
        elif fmt=='plume':
           x = model[:,0]
           y = model[:,1]
           T = model[:,2]
           self.fmt='plume'

        self.x = x
        self.y = y

        if fmt=='pvk':
           self.T = Tpot
           self.f = f
           self.theta = theta
           self.depth = depth

        elif fmt=='pvk_full':
           self.T = Tpot
           self.f = f
           radius, theta_deg = cart2polar(self.x,self.y)
           self.depth = 6371.0 - radius
           self.theta = 90.0 - np.rad2deg(theta_deg)
           for i in range(0,len(self.theta)):
               if self.theta[i] > 360.0:
                   self.theta[i] -= 360.0
               elif self.theta[i] < 0.0:
                   self.theta[i] += 360.0

        elif fmt=='plume':
           self.T = T
           self.f = np.zeros(len(self.T))
           self.f[:] = plume_f
           radius, lat_rad = cart2polar(self.x,self.y)
           self.depth = 6371.0 - radius
           self.theta = 90.0 - np.rad2deg(lat_rad)


        #remove 'outliers' that are over 100% basalt
        for i in range(0,len(self.f)):
            if self.f[i] > bf_scaling:
                self.f[i] = bf_scaling

        #normalize basalt composition field
        #self.f /= bf_scaling #Not sure if this is done correctly!

        #normalize basalt composition field
        # C = 0 is pure harzburgite, C = 1 is pyrolite C = 8? is pure basalt
        pyro_c = 0.125
        for i in range(0,len(self.f)):
           self.f[i] *= pyro_c
           #self.f[i] /= pyro_c

           if self.f[i] >= 1.0:
              self.f[i] = 1.0

        #reshape scatter data into an array
        if fmt=='pvk':
           dtheta = np.max((np.diff(self.theta)))
           ddepth = np.max(np.abs(np.diff(self.depth)))
           self.dtheta = dtheta
           self.ddepth = ddepth
           self.theta_axis = np.arange(np.min(self.theta),np.max(self.theta)+dtheta,dtheta)
           self.depth_axis = np.arange(np.min(self.depth),np.max(self.depth)+ddepth,ddepth)
           npts_theta = len(self.theta_axis)
           npts_depth = len(self.depth_axis)

        elif fmt=='pvk_full':
           #this is the format use in EPSL2008/F_98 (gridTandC)
           npts_theta = 360
           npts_depth = 100
           dtheta = 1.0
           ddepth = 28.5
           self.dtheta = dtheta
           self.ddepth = ddepth
           self.theta_axis = np.arange(np.min(self.theta),np.max(self.theta)+dtheta,dtheta)
           #self.depth_axis = np.arange(np.min(self.depth),np.max(self.depth)+ddepth,ddepth)
           self.depth_axis = np.linspace(np.min(self.depth),np.max(self.depth),npts_depth)

        elif fmt=='plume':
           npts_theta = 201
           npts_depth = 288
           dtheta = 10.0/npts_theta
           ddepth = 2878.0/npts_depth
           self.dtheta = dtheta
           self.ddepth = ddepth
           self.theta_axis = np.linspace(0,10,npts_theta)
           self.depth_axis = np.linspace(0,2878,npts_depth)

        self.npts_theta = npts_theta
        self.npts_depth = npts_depth

        if debug:
           print(ddepth, dtheta)
           print(self.T.shape, npts_depth, npts_theta, npts_depth*npts_theta)

        if fmt == 'pvk':
           self.T_array = np.reshape(self.T,(npts_depth,npts_theta))
           self.f_array = np.reshape(self.f,(npts_depth,npts_theta))

        elif fmt == 'pvk_full':
           self.T_array = np.reshape(self.T,(npts_depth,npts_theta),order='F')
           self.f_array = np.reshape(self.f,(npts_depth,npts_theta),order='F')

        #get 1d pressure profile from PREM
        self.P = prem.get_p(self.depth_axis)
        self.P /= 1e5 #convert to bar?

    def add_adiabat(self,adiabat_file,cmb_temp=3000,surf_temp=273,scaleT=True):
        '''
        adds an adiabat to the temperature profile
        '''
        adiabat = np.loadtxt(adiabat_file)
        T_adiabat = adiabat[:,0]
        P_adiabat = adiabat[:,1]
        interp_T = interp1d(P_adiabat,T_adiabat,fill_value='extrapolate')
        T_int = interp_T(self.P[::-1])
        self.T_array = self.T_array + T_int[:,None]

        #scale temperature field
        if scaleT:
           deltaT_mantle = cmb_temp - surf_temp
           self.T_array -= np.min(self.T_array)
           self.T_array /= np.max(self.T_array)
           self.T_array *= deltaT_mantle
           self.T_array += surf_temp
           self.T = np.ravel(self.T_array)

    def potT_to_absT(self,adiabats_dir=adiabats_dir,debug=False):
        tp2ta = gen_adiabat_interpolator(adiabats_dir,debug=False)
        for i in range(0,len(self.T)):
            if self.T[i] < 350:
                self.T[i] = 350.0

            P_here = prem.get_p(self.depth[i])
            P_here /= 1.0e5
            T_here = tp2ta((self.T[i],P_here))

            if debug:
               if np.isnan(T_here):
                  print('Tpot:',self.T[i],'depth:',self.depth[i],'P:',P_here)

            self.T[i] = T_here

        if self.fmt == 'pvk':
           self.T_array = np.reshape(self.T,(self.npts_depth,self.npts_theta))
        elif self.fmt =='pvk_full':
           cut_in_half = True
           if cut_in_half:
              self.T_array = np.reshape(self.T,(self.npts_depth,(int(self.npts_theta/2))),order='F')
           else:
              self.T_array = np.reshape(self.T,(self.npts_depth,self.npts_theta),order='F')

    def potT_to_absT_OLD(self,adiabat_dir):
        p = []
        pot_T = []
        abs_T = []

        ads = glob.glob(adiabat_dir+'./*.out')

        for ad in ads:
            pot_temp_str = ad.split('_')[1].split('K')[0]
            pot_temp = int(pot_temp_str)
            f = np.loadtxt(ad)
            temp = f[:,0]
            pressure = f[:,1]
            for i in range(0,len(temp)):
                p.append(pressure[i])
                pot_T.append(pot_temp)
                abs_T.append(temp[i])

        plot = False
        if plot:
           plt.scatter(pot_T,p,c=abs_T,cmap='viridis',edgecolor='none')
           plt.show()

        #first interpolate irregular scatter data onto uniform mesh
        x_i = np.arange(0,2550,50) #adiabats above 2500 are nonsense
        #x_i = np.arange(0,1850,50) #adiabats above 2500 are nonsense
        y_i = np.linspace(0,1350000,1000)
        z_i = griddata((pot_T,p),abs_T, (x_i[None,:],y_i[:,None]), method='linear')

        #next make an interpolating function
        rgi = RegularGridInterpolator(points=(x_i,y_i),values=z_i.T,
                bounds_error=False,fill_value=None)
        #rgi takes a tuple of (potential temp, pressure) and returns absolute temp
        #for i in range(0,len(self.T_array.flatten())):
        #    new_T = rgi((self.T_array.flatten()[i],
        #                 self.P_array.flatten()[i]))
        #    self.T_array.flatten[i] = new_T

        for i in range(0,len(self.depth_axis)):
            for j in range(0,len(self.theta_axis)):
                if self.T_array[i,j] < 300:
                    self.T_array[i,j] = 300
                new_T = rgi((self.T_array[i,j],self.P[::-1][i]))
                self.T_array[i,j] = new_T

    def velocity_conversion(self,composition,basalt_fraction=0.18,
            #lookup_tables='/geo/home/romaguir/utils/lookup_tables/lookup_tables.hdf5',
            mode='global',**kwargs):
        '''
        performs the velocity conversion.

        args:

           lookup_tables: str
                          path to h5py lookup tables

           composition: str
                        'dynamic' to use f from dynamic simulations or 
                        'constant' to set f with argument 'basalt_fraction'

           mode: str
                        current options: 'global', and 'rel1d'
                        if 'global', the percent deviations are taken
                        relative to the average. 
                        if 'rel1d' the deviations are computed relative to a reference
                        profile, defined by the user (see kwargs)

        kwargs:
           ref_theta_range: tuple
                            defines the theta range of what we consider to be 
                            'average' mantle.  default is the entire model.
                            The entire model may not be desirable for the half
                            cylinder models if certain depths are heavily biased
                            by the presence strong dynamical structure
           ref_f: float
                  (if mode is 'rel1d') composition of the reference profile
                  by default the value is 0.18 (i.e., pyrolite)
           ref_T: numpy array
                  (if mode is 'red1d') Temperature of reference profile
        '''

        #get kwargs
        ref_theta_range = kwargs.get('ref_theta_range','full')
        ref_f = kwargs.get('ref_f',0.18)
        #ref_T = kwargs.get('ref_T',0)
        ref_T = self.T_array[:,-1]

        if composition == 'constant':
           self.f[:] = basalt_fraction
           self.f_array[:,:] = basalt_fraction

        tables = h5py.File(lookup_tables,'r')
        #read harzburgite lookuptable
        harz_vp_table = tables['harzburgite']['vp'][:]
        harz_vs_table = tables['harzburgite']['vs'][:]
        harz_rho_table = tables['harzburgite']['rho'][:]
        #read morb lookuptable
        morb_vp_table = tables['morb']['vp'][:]
        morb_vs_table = tables['morb']['vs'][:]
        morb_rho_table = tables['morb']['rho'][:]
        #read topology
        P0 = tables['pyrolite']['P0'].value
        T0 = tables['pyrolite']['T0'].value
        nP = tables['pyrolite']['nP'].value
        nT = tables['pyrolite']['nT'].value
        dP = tables['pyrolite']['dP'].value
        dT = tables['pyrolite']['dT'].value
        P_table_axis = np.arange(P0,((nP-1)*dP+P0+dP),dP)
        T_table_axis = np.arange(T0,((nT-1)*dT+T0+dT),dT)
        #create interpolators
        harz_vp_interpolator  = interp2d(P_table_axis,T_table_axis, harz_vp_table)
        harz_vs_interpolator  = interp2d(P_table_axis,T_table_axis, harz_vs_table)
        harz_rho_interpolator = interp2d(P_table_axis,T_table_axis, harz_rho_table)
        morb_vp_interpolator  = interp2d(P_table_axis,T_table_axis, morb_vp_table)
        morb_vs_interpolator  = interp2d(P_table_axis,T_table_axis, morb_vs_table)
        morb_rho_interpolator = interp2d(P_table_axis,T_table_axis, morb_rho_table)
        #point by point velocity conversion
        self.vp_array = np.zeros(self.T_array.shape)
        self.vs_array = np.zeros(self.T_array.shape)
        self.rho_array = np.zeros(self.T_array.shape)
        self.dvp_array = np.zeros(self.T_array.shape)
        self.dvs_array = np.zeros(self.T_array.shape)
        self.drho_array = np.zeros(self.T_array.shape)
        self.deltaT_array = np.zeros(self.T_array.shape)
        for i in range(0,len(self.depth_axis)):
            for j in range(0,len(self.theta_axis)):
                P_here = self.P[::-1][i]
                T_here = self.T_array[i,j]
                f_here = self.f_array[i,j]
                print(f_here)
                vp_harz = harz_vp_interpolator(P_here,T_here)
                vs_harz = harz_vs_interpolator(P_here,T_here)
                rho_harz = harz_rho_interpolator(P_here,T_here)
                vp_morb = morb_vp_interpolator(P_here,T_here)
                vs_morb = morb_vs_interpolator(P_here,T_here)
                rho_morb = morb_rho_interpolator(P_here,T_here)
                self.vp_array[i,j] = f_here*vp_morb + (1-f_here)*vp_harz
                self.vs_array[i,j] = f_here*vs_morb + (1-f_here)*vs_harz
                self.rho_array[i,j] = f_here*rho_morb + (1-f_here)*rho_harz

        if mode=='global':

            if ref_theta_range == 'full':
                vp_avg = np.average(self.vp_array,axis=1)
                vs_avg = np.average(self.vs_array,axis=1)
                rho_avg = np.average(self.rho_array,axis=1)
                T_avg = np.average(self.T_array,axis=1)
            else:
                istart = ((ref_theta_range[0]-np.min(self.theta))/self.dtheta)
                iend = ((ref_theta_range[1]-np.min(self.theta))/self.dtheta)
                vp_avg = np.average(self.vp_array[:,istart:iend],axis=1)
                vs_avg = np.average(self.vs_array[:,istart:iend],axis=1)
                rho_avg = np.average(self.rho_array[:,istart:iend],axis=1)
                plt.plot(vs_avg)
                plt.show()


            #calculate the percent deviation of vp, vs, and rho
            for i in range(0,len(self.depth_axis)):
                for j in range(0,len(self.theta_axis)):
                    self.dvp_array[i,j] = ((self.vp_array[i,j] - vp_avg[i])/vp_avg[i]) * 100.0
                    self.dvs_array[i,j] = ((self.vs_array[i,j] - vs_avg[i])/vs_avg[i]) * 100.0
                    self.drho_array[i,j] = ((self.rho_array[i,j] - rho_avg[i])/rho_avg[i]) * 100.0
                    self.deltaT_array[i,j] = self.T_array[i,j] - T_avg[i]
        elif mode=='rel1d':
            vp_ref1d = np.zeros(len(self.depth_axis))
            vs_ref1d = np.zeros(len(self.depth_axis))
            rho_ref1d = np.zeros(len(self.depth_axis))

            for i in range(0,len(self.depth_axis)):
                P_here = self.P[::-1][i]
                T_here = ref_T[i]
                vp_harz = harz_vp_interpolator(P_here,T_here)
                vs_harz = harz_vs_interpolator(P_here,T_here)
                rho_harz = harz_rho_interpolator(P_here,T_here)
                vp_morb = morb_vp_interpolator(P_here,T_here)
                vs_morb = morb_vs_interpolator(P_here,T_here)
                rho_morb = morb_rho_interpolator(P_here,T_here)
                vp_ref1d[i] = ref_f*vp_morb + (1-ref_f)*vp_harz
                vs_ref1d[i] = ref_f*vs_morb + (1-ref_f)*vs_harz
                rho_ref1d[i] = ref_f*rho_morb + (1-ref_f)*rho_harz

            #calculate the percent deviation of vp, vs, and rho
            for i in range(0,len(self.depth_axis)):
                for j in range(0,len(self.theta_axis)):
                    self.dvp_array[i,j] = ((self.vp_array[i,j] - vp_ref1d[i])/vp_ref1d[i]) * 100.0
                    self.dvs_array[i,j] = ((self.vs_array[i,j] - vs_ref1d[i])/vs_ref1d[i]) * 100.0
                    self.drho_array[i,j] = ((self.rho_array[i,j] - rho_ref1d[i])/rho_ref1d[i]) * 100.0

        if self.fmt == 'pvk' or self.fmt =='plume':
           self.vp = np.ravel(self.vp_array) 
           self.vs = np.ravel(self.vs_array) 
           self.rho = np.ravel(self.rho_array) 
           self.dvp = np.ravel(self.dvp_array) 
           self.dvs = np.ravel(self.dvs_array) 
           self.drho = np.ravel(self.drho_array) 
        elif self.fmt == 'pvk_full':
           self.vp = np.ravel(self.vp_array,order='F') 
           self.vs = np.ravel(self.vs_array,order='F') 
           self.rho = np.ravel(self.rho_array,order='F') 
           self.dvp = np.ravel(self.dvp_array,order='F') 
           self.dvs = np.ravel(self.dvs_array,order='F') 
           self.drho = np.ravel(self.drho_array,order='F') 
           self.deltaT = np.ravel(self.deltaT_array,order='F') 
        else:
           raise ValueError('format', self.fmt, 'not recognized')

    def twoD_to_threeD(self,npts_z,npts_lat,npts_lon,var='dvs',debug=True,**kwargs):
        '''
        extrapolates 2d half-circle into 3D earth.

        args:
            npts_z: desired number of depth points in 3D model
            npts_lat: desired number of lat points in 3D model
            npts_lon: desired number of lon points in 3D model
            var: variable to extrapolate to 3D.

        **kwargs:
            z_mantle: mantle thickness in km (default 2885)

        returns:
           An instance of the model_3d class
        '''
        z_mantle = kwargs.get('z_mantle',2885.0)
        if var=='dvs':
           vals = self.dvs_array
        elif var=='dvp':
           vals = self.dvp_array

        dz = z_mantle/npts_z
        dlat = 180.0/npts_lat
        dlon = 360.0/npts_lon

        #first interpolate 2D grid to new grid (if a different resolution is required)
        if debug:
           print('function twoD_to_threeD: STARTING 2D INTERPOLATION')
           print('self.depth_axis (len{}):'.format(len(self.depth_axis)), self.depth_axis)
           print('self.theta_axis (len{}):'.format(len(self.theta_axis)), self.theta_axis)
           print('shape of self.depth_axis:', self.depth_axis.shape)
           print('shape of self.theta_aixs:', self.theta_axis.shape)
           print('shape of vals:', vals.shape)
        grid_interp = RegularGridInterpolator((self.depth_axis,self.theta_axis),vals, bounds_error=False)
        if debug:
           print('FINISHED 2D INTERPOLATION')

        x_i = np.linspace(0,z_mantle,npts_z)
        y_i = np.linspace(0,180.0,npts_lat)

        xx,yy = np.meshgrid(x_i,y_i,indexing='ij')
        field = grid_interp((xx,yy))
        field = np.fliplr(field)
        print('shape of interpolated field = ',field.shape)

        mod_3d = model_3d(radmin=(6371-z_mantle),radmax=6371.0,npts_rad=npts_z,
                          latmin=-90.0,latmax=90.0,npts_lat=npts_lat,
                          lonmin=0.0,lonmax=360.0,npts_lon=npts_lon)

        print('shape of 3d model data ', mod_3d.data.shape)
        for i in range(0,len(mod_3d.lon)-1):
            mod_3d.data[:,:,i] = field

        return mod_3d

    def plot_TandC(self,type='scatter'):
        if type=='scatter':
           fig,axes = plt.subplots(1,2,sharey=True,figsize=(12,8),facecolor='grey',frameon=False)
           ax0 = axes[0].scatter(self.x,self.y,c=self.T,cmap='hot',edgecolor='none')
           cb0 = plt.colorbar(ax0,ax=axes[0],orientation='vertical',label='T (C)')
           axes[0].axis('off')
           ax1 = axes[1].scatter(self.x,self.y,c=self.f,cmap='viridis',edgecolor='none')
           cb1 = plt.colorbar(ax1,ax=axes[1],orientation='vertical',label='basalt fraction')
           axes[1].axis('off')
        elif type=='imshow':
           extent = (np.min(self.theta_axis),np.max(self.theta_axis),
                     np.max(self.depth_axis),np.min(self.depth_axis))
           fig,axes = plt.subplots(2,sharex=True,figsize=(15,10),facecolor='grey',frameon=False)
           ax0 = axes[0].imshow(np.flipud(self.T_array),cmap='hot',extent=extent,aspect='auto')
           axes[0].set_ylabel('depth (km)')
           cb0 = plt.colorbar(ax0,ax=axes[0],orientation='vertical',label='T (C)')
           ax1 = axes[1].imshow(np.flipud(self.f_array),cmap='viridis',extent=extent,aspect='auto')
           cb1 = plt.colorbar(ax1,ax=axes[1],orientation='vertical',label='basalt fraction')
           axes[1].set_ylabel('depth (km)')
           axes[1].set_xlabel('theta (deg)')
        plt.show()

    def plot_field(self,field,type='scatter'):
        '''
        plot a 2d model field.

        args:

           field: str
                  variable to plot ['vp','vs','rho','dvp','dvs',or 'drho']
        '''
        if type == 'scatter':
            if field=='vp': 
                plt.scatter(self.x,self.y,c=self.vp,edgecolor='none',facecolor='grey')
                plt.axis('off')
                plt.colorbar(label='$V_p$ (km/s)')
            elif field=='vs': 
                plt.scatter(self.x,self.y,c=self.vs,edgecolor='none',facecolor='grey')
                plt.colorbar(label='$V_s$ (km/s)')
                plt.axis('off')
            elif field=='rho': 
                plt.scatter(self.x,self.y,c=self.rho,edgecolor='none',facecolor='grey')
                plt.colorbar(label='$\rho$ (g/cm$^3$)')
                plt.axis('off')
            elif field=='dvp': 
                plt.scatter(self.x,self.y,c=self.dvp,edgecolor='none',facecolor='grey',cmap='seismic_r',vmin=-5,vmax=5)
                plt.colorbar(label='$\delta V_p$ (%)')
                plt.axis('off')
            elif field=='dvs': 
                plt.scatter(self.x,self.y,c=self.dvs,edgecolor='none',facecolor='grey',cmap='seismic_r',vmin=-5,vmax=5)
                plt.colorbar(label='$\delta V_s$ (%)')
                plt.axis('off')
            elif field=='drho': 
                plt.scatter(self.x,self.y,c=self.drho,edgecolor='none',facecolor='grey',cmap='seismic_r',vmin=-5,vmax=5)
                plt.colorbar(label='$\delta \rho$ (%)')
                plt.axis('off')
            plt.show()
        if type == 'imshow':
            if field=='vp': 
                plt.imshow(np.flipud(self.vp_array),aspect='auto',cmap='seismic_r')
                plt.axis('off')
                plt.colorbar(label='$V_p$ (km/s)')
            if field=='vs': 
                plt.imshow(np.flipud(self.vs_array),aspect='auto',cmap='seismic_r')
                plt.axis('off')
                plt.colorbar(label='$V_s$ (km/s)')
            if field=='dvp': 
                plt.imshow(np.flipud(self.dvp_array),aspect='auto',cmap='seismic_r')
                plt.axis('off')
                plt.colorbar(label='$dV_p/V_p$ ((%))',cmap='seismic_r')
            if field=='dvs': 
                plt.imshow(np.flipud(self.dvs_array),aspect='auto',cmap='seismic_r')
                plt.axis('off')
                plt.colorbar(label='$dV_s/V_s$ ((%))')
            plt.show()

    def plot_radial_average(self,var='T'):
        if var=='T':
            plt.plot(np.average(self.T_array,axis=1),self.depth_axis[::-1])
            plt.gca().invert_yaxis()
            plt.xlabel('T (C)')
            plt.ylabel('depth (km)')
            plt.show()
        elif var=='f':
            plt.plot(np.average(self.f_array,axis=1),self.depth_axis[::-1])
            plt.gca().invert_yaxis()
            plt.xlabel('basalt fraction')
            plt.ylabel('depth (km)')
            plt.show()
        elif var=='vp':
            plt.plot(np.average(self.vp_array,axis=1),self.depth_axis[::-1])
            plt.gca().invert_yaxis()
            plt.xlabel('basalt fraction')
            plt.ylabel('depth (km)')
            plt.show()
        elif var=='vs':
            plt.plot(np.average(self.vs_array,axis=1),self.depth_axis[::-1])
            plt.gca().invert_yaxis()
            plt.xlabel('basalt fraction')
            plt.ylabel('depth (km)')
            plt.show()

    def scaleT(self,scale_factor):
        '''
        scales temperatures by scale_factor
        '''
        self.T *= scale_factor

    def write_2d_output(self,filename,fmt='pvk'):
        if fmt == 'pvk':
            #np.savetxt(filename,np.c_[self.x,self.y,self.theta,self.depth,self.T,self.f,self.dvp,self.dvs,self.drho,self.deltaT], fmt='%3f')
            np.savetxt(filename,np.c_[self.x,self.y,self.theta,self.depth,self.T,self.f,self.dvp,self.dvs,self.drho], fmt='%3f')

# 3D_model_class
class model_3d(object):
   '''
   a class for dealing with 3d global data

   args--------------------------------------------------------------------------
   latmin: minimum latitude of model (default = -10.0)
   latmax: maximum latitude of model (default = 10.0)
   lonmin: minimum longitude of model (default = -10.0)
   lonmax: maximum longitude of model (default = 10.0)
   radmin: minimum radius of model (default = 3494.0 km)
   radmax: maximum radius of model (default = 6371.0 km)
   dlat: latitude spacing (default = 1.0)
   dlon: longitude spacing (default = 1.0)
   drad: radius spacing (default = 20.0)

   kwargs------------------------------------------------------------------------
   region: select preset region
   '''

   def __init__(self,npts_rad,npts_lat,npts_lon,
                radmin=3494.0,radmax=6371.0,
                latmin=-10.0,latmax=10.0,
                lonmin= -10.0,lonmax=10.0,**kwargs):

      #check if using a preset region--------------------------------------------
      region = kwargs.get('region','None')
      if region == 'North_America_rfs':
         lonmin=232
         lonmax=294
         latmin=24
         latmax=52
         radmin=5000   
   
      #define model range--------------------------------------------------------
      self.npts_rad = npts_rad
      self.npts_lat = npts_lat
      self.npts_lon = npts_lon
      drad        = (radmax-radmin)/npts_rad
      dlat        = (latmax-latmin)/npts_lat
      dlon        = (lonmax-lonmin)/npts_lon
      #self.rad    = np.arange(radmin,radmax+drad,drad)
      #self.lon    = np.arange(lonmin,lonmax+dlon,dlon)
      #self.lat    = np.arange(latmin,latmax+dlat,dlat)
      self.rad    = np.linspace(radmin,radmax,npts_rad)
      self.lon    = np.linspace(lonmin,lonmax,npts_lon)
      self.lat    = np.linspace(latmin,latmax,npts_lat)
      self.colat  = 90.0 - self.lat

      #data is given on grid points defined by self.lat, self.lon and self.rad
      self.data = np.zeros((len(self.rad),len(self.colat),len(self.lon)))
      self.dlat = dlat
      self.dlon = dlon
      self.drad = drad

   def rotate_3d(self,destination):
      '''
      solid body rotation of the mantle with respect to the surface.
      rotates the north pole to the specified destination.

      args:
          destination: (lat,lon)
      '''
      lat = destination[0]
      lon = destination[1]
      s1 = [6371, 180, 0]
      s2 = [6371, 90-lat, lon]
      rotation_vector = find_rotation_vector(s1,s2)
      s1 = [6371, 0, 0]
      s2 = [6371, 90-lat, lon]
      rotation_angle = find_rotation_angle(s1,s2)
      #note that s1 is defined to be either at the south pole when finding
      #the rotation vector and the north pole when finding the angle...
      #this has to do with how the cross product is defined in find_rotation_vector

      pts_new = []
      R=rotation_matrix(n=rotation_vector,phi=rotation_angle)
      interp =  RegularGridInterpolator(points = (self.rad,self.lat,self.lon),
                        values = self.data,bounds_error=False,fill_value = 0.0)

      print('patience... rotation and interpolation may take a couple minutes for large models')
      colat_rad = np.rad2deg(self.colat)
      lon_rad = np.rad2deg(self.lon)
      for i in range(0,len(self.rad)):
          for j in range(0,len(self.colat)):
              for k in range(0,len(self.lon)):
                  pt = rotate_coordinates(n=rotation_vector,
                                          phi=rotation_angle,
                                          colat=self.colat[j],
                                          lon=self.lon[k])

                  lat_new = 90-pt[0]
                  lon_new = pt[1]
                  if lon_new < 0:
                      lon_new += 360.0

                  pts_new.append([self.rad[i],lat_new,lon_new])

      data_new = interp(pts_new)
      data_new = data_new.reshape(self.data.shape,order='C')
      self.data = data_new
      

   def plot_3d(self,**kwargs):
      '''
      plots a 3d earth model using mayavi
      
      kwargs:
              grid_x: Draw the x plane (default False)
              grid_y: Draw the y plane (default False)
              grid_z: Draw the z plane (default False)
              earth:  Draw the earth outline and coastlines (default True)
              plot_quakes: Draw earthquakes in earthquake_list (default False)
              earthquake_list: a list of earthquake coordinates specified by 
                               (lat,lon,depth)

      '''
      import mayavi
      from mayavi import mlab
      from tvtk.api import tvtk
      from mayavi.scripts import mayavi2
      from mayavi.sources.vtk_data_source import VTKDataSource
      from mayavi.sources.builtin_surface import BuiltinSurface
      from mayavi.modules.api import Outline, GridPlane, ScalarCutPlane

      draw_gridx = kwargs.get('grid_x',False)
      draw_gridy = kwargs.get('grid_y',False)
      draw_gridz = kwargs.get('grid_z',False)
      draw_earth = kwargs.get('earth',True)
      draw_quakes = kwargs.get('draw_quakes',False)
      earthquake_list = kwargs.get('earthquake_list','none')

      dims = (len(self.rad),len(self.lon),len(self.colat))
      pts = generate(phi=np.radians(self.lon),theta=np.radians(self.colat),rad=self.rad)
      sgrid = tvtk.StructuredGrid(dimensions=dims)
      sgrid.points = pts
      s = np.zeros(len(pts))

      #Map data onto the grid
      print('Patience... mapping data to structured grid')
      for i in range(0,len(self.colat)):
         for j in range(0,len(self.lon)):
            for k in range(0,len(self.rad)):
            
               s[count] = self.data[k,i,j]
               sgrid.point_data.scalars = s
               sgrid.point_data.scalars.name = 'scalars'
               count += 1
  
      #use vtk dataset
      src = VTKDataSource(data=sgrid)

      #set figure defaults
      mlab.figure(bgcolor=(0,0,0))

      #outline
      mlab.pipeline.structured_grid_outline(src,opacity=0.3)

      #show grid planes
      if draw_gridx == True:
         gx = mlab.pipeline.grid_plane(src,color=(1,1,1),opacity=0.25)
         gx.grid_plane.axis='x'
      if draw_gridy == True:
         gy = mlab.pipeline.grid_plane(src,color=(1,1,1),opacity=0.25)
         gy.grid_plane.axis='y'
      if draw_gridz == True:
         gz = mlab.pipeline.grid_plane(src,color=(1,1,1),opacity=0.25)
         gz.grid_plane.axis='z'

      #cutplane
      mlab.pipeline.scalar_cut_plane(src,plane_orientation='y_axes',
                                     colormap='jet',view_controls=False)

      #draw earth and coastlines
      if draw_earth == True:
         coastline_src = BuiltinSurface(source='earth',name='Continents')      
         coastline_src.data_source.radius = 6371.0
         coastline_src.data_source.on_ratio = 1
         mlab.pipeline.surface(coastline_src)

      #plot earthquakes
      if draw_quakes == True:
         print("Sorry, this doesn't work yet")
         lat_pts=earthquake_list[:,0]
         lon_pts=earthquake_list[:,1]
         rad_pts=6371.0-earthquake_list[:,2]
         theta_pts = np.radians(lat_pts)
         phi_pts   = np.radians(lon_pts)
       
         #convert point to cartesian
         x_pts = rad_pts*np.cos(phi_pts)*np.sin(theta_pts)
         y_pts = rad_pts*np.sin(phi_pts)*np.sin(theta_pts)
         z_pts = rad_pts*np.cos(theta_pts)
         eq_pts = mlab.points3d(x_pts,y_pts,z_pts,
                                scale_mode='none',
                                scale_factor=100.0,
                                color=(1,1,0))

      mlab.show()

   def find_index(self,lat,lon,depth):
      '''
      provided a latitude, longitude, and depth, finds the corresponding
      model index
      '''
      rad       = 6371.0-depth
      lat_min   = self.lat[0]
      lon_min   = self.lon[0]
      rad_min   = self.rad[0]

      lat_i = int((lat-lat_min)/self.dlat)
      lon_i = int((lon-lon_min)/self.dlon)
      rad_i = int((rad-rad_min)/self.drad)

      if lat_i >= len(self.lat): 
         print("latitude ", lat, " is outside the model")
      if lon_i >= len(self.lon): 
         print("longitude ", lon, " is outside the model")
      if rad_i >= len(self.rad): 
         print("depth", depth, " is outside the model")

      return rad_i, lat_i, lon_i

   def map_rf(self,pierce_dict,**kwargs):
      '''
      takes a seispy receiver function, cycles through the pierce
      dictionary and maps the values to the 3d model. alternatively
      takes just the pierce point dictionary instead of the receiver
      function object
  
      **kwargs-------------------------------------------------------------------
      rf: a seispy receiver function object, complete with pierce point dictionary
      pierce_dict: dictionary containing geographic pierce point information, and
                   receiver function amplitude
      '''
      #from seis_tools.models.models_3d import find_index
      #rf = kwargs.get('rf','None')
      #pierce_dict = kwargs.get('pierce_dict','None')

      #if rf != 'None':
      #   pts = rf.pierce
      #elif pierce_dict != 'None':
      #   pts = pierce_dict
      pts = pierce_dict     

      for i in pts:
         lat   = i['lat']
         lon   = i['lon']
         depth = i['depth']
         amp   = i['amplitude']

         ind = self.find_index(lat,lon,depth)
         print(lat,lon,depth,ind)
 
         #map the value
         #self.data[ind] += amp
         self.data[ind] = 1

   def probe_data(self,rad,lat,lon,**kwargs):
      '''
      returns the value of the field at the point specified. 
   
      params:
      lat
      lon
      depth
      '''
      return interpn(points = (self.rad,self.lat,self.lon),
                        values = self.data,
                        xi = (rad,lat,lon),
                        bounds_error=False,
                        fill_value = 0.0)

   def save(self,format,filename):
      '''
      save an instance of the 3d model class
      '''
      if format == 'pickle':
         pickle.dump(self,file(filename,'w'))

   def write_specfem_ppm(self,**kwargs):
       '''
       writes a 3d model to ppm format for specfem
       '''
       fname = kwargs.get('fname','model.txt')
       f = open(fname,'w')
       f.write('#lon(deg), lat(deg), depth(km), Vs-perturbation wrt PREM(%), Vs-PREM(km/s) \n')

       #loop through model and write points (lat = inner, lon = middle, rad = outer)
       for i in range(0,len(self.rad)):
           depth = 6371.0 - self.rad[::-1][i]
           prem_vs = 5.0 #prem.get_vs(depth)
           for j in range(0,len(self.lon)):
               lon = self.lon[j]
               for k in range(0,len(self.lat)):
                   lat = self.lat[k]
                   dv = self.data[(len(self.rad)-(i+1)),j,k]
                   f.write('{} {} {} {} {}'.format(lon,lat,depth,dv,prem_vs)+'\n')

   def write_specfem_heterogen(self,**kwargs):
       '''
       writes 3d model to a 'heterogen.txt' to be used in specfem.
       '''
       fname = kwargs.get('fname','heterogen.txt')
       f = open(fname,'w')
       for i in range(0,len(self.rad)):
           for j in range(0,len(self.lat)):
               for k in range(0,len(self.lon)):
                   f.write('{}'.format(self.data[i,j,k])+'\n')

def write_specfem_ppm(dvs_model3d,dvp_model3d,drho_model3d,**kwargs):
    '''
    writes a 3d model to ppm format for specfem
    '''
    fname = kwargs.get('fname','model.txt')
    f = open(fname,'w')
    f.write('#lon(deg), lat(deg), depth(km), dvs(%), dvp(%), drho(%) \n')

    #loop through model and write points (lat = inner, lon = middle, rad = outer)
    for i in range(0,len(dvs_model3d.rad)):
       depth = 6371.0 - dvs_model3d.rad[::-1][i]
       for j in range(0,len(dvs_model3d.lon)):
          lon = dvs_model3d.lon[j]
          for k in range(0,len(dvs_model3d.lat)):
             lat = dvs_model3d.lat[k]
             dvs = dvs_model3d.data[(len(dvs_model3d.rad)-(i+1)),j,k]
             dvp = dvp_model3d.data[(len(dvp_model3d.rad)-(i+1)),j,k]
             drho = drho_model3d.data[(len(drho_model3d.rad)-(i+1)),j,k]
             f.write('{} {} {} {} {} {}'.format(lon,lat,depth,dvs,dvp,drho)+'\n')

def write_s40_filter_inputs(model_3d,**kwargs):
   '''
   Takes in a 3d model and writes out the input files for the S40RTS tomographic filter.
   The data in the 3d model is given as point data, while this writes out data for 
   layers.  Therefore, the total number of layers is one less than the number of 
   radial grid points.  The value in each layer is taken as the average of the 
   closest two radial values.
   
   params:
   model_3d: instance of the model_3d class

   kwargs:
   model_name : model_name (string)
   save_dir : save directory (string)

   Each output file is of the form-----------------------------------------------

   layer depth_start
   layer depth_end
   lat, lon, val
   .
   .   
   .
   
   lat, lon, val
   ------------------------------------------------------------------------------
   '''
   model_name  = kwargs.get('model_name','none')
   save_dir    = kwargs.get('save_dir','./')

   if os.path.exists(save_dir) == False:
      os.makedirs(save_dir)

   type = kwargs.get('type','point')

   #initializations
   lat = model_3d.lat
   lon = model_3d.lon
   depth = 6371.0 - model_3d.rad
   lon_min = min(model_3d.lon)
   lon_max = max(model_3d.lon)
   lat_min = min(model_3d.lat)
   lat_max = max(model_3d.lat)

   #check for nans
   model_3d.data = np.nan_to_num(model_3d.data)
    
   for i in range(0,len(depth)-1):
      count = len(depth) - i - 1

      #open file and write header
      out_name = str(model_name)+'.'+str(count)+'.dat' 
      print('out_name', out_name)
      output   = open(save_dir+'/'+out_name,'w')

      output.write('{}'.format(6371.0-model_3d.rad[i+1])+'\n')
      output.write('{}'.format(6371.0-model_3d.rad[i])+'\n')

      for j in range(0,len(lat)-1):
         for k in range(0,len(lon)-1):
             
            #v1 = model_3d.data[i,j,k]
            #v2 = model_3d.data[i-1,j,k]
            v1 = model_3d.data[i,j,k]
            v2 = model_3d.data[i+1,j,k]
            value = (v1+v2)/2.0

            line = '{} {} {}'.format(lat[j],lon[k],value)
            output.write(line+'\n')

def rotate_about_axis(tr,lon_0=60.0,lat_0=0.0,degrees=0):
   '''
   Rotates the source and receiver of a trace object around an
   arbitrary axis.
   '''

   alpha = np.radians(degrees)
   lon_s = tr.stats.sac['evlo']
   lon_r = tr.stats.sac['stlo']
   colat_s = 90.0-tr.stats.sac['evla']
   colat_r = 90.0-tr.stats.sac['stla']
   colat_0 = 90.0-lat_0

   x_s = lon_s - lon_0
   y_s = colat_0 - colat_s
   x_r = lon_r - lon_0
   y_r = colat_0 - colat_r


   #rotate receiver
   tr.stats.sac['stla'] = 90.0-colat_0+x_r*np.sin(alpha) + y_r*np.cos(alpha)
   tr.stats.sac['stlo'] = lon_0+x_r*np.cos(alpha) - y_r*np.sin(alpha)

   #rotate source
   tr.stats.sac['evla'] = 90.0-colat_0+x_s*np.sin(alpha) + y_s*np.cos(alpha)
   tr.stats.sac['evlo'] = lon_0+x_s*np.cos(alpha) - y_s*np.sin(alpha)

def cartesian_to_spherical(x,degrees=True,normalize=False):
   '''
   Coverts a cartesian vector in R3 to spherical coordinates
   '''
   r = np.linalg.norm(x)
   theta = np.arccos(x[2]/r)
   phi = np.arctan2(x[1],x[0])

   if degrees:
      theta = np.degrees(theta)
      phi = np.degrees(phi)

   s = [r,theta,phi]

   if normalize:
      s /= np.linalg.norm(s)

   return s

def spherical_to_cartesian(s,degrees=True,normalize=False):
   '''
   Takes a vector in spherical coordinates and converts it to cartesian.
   Assumes the input vector is given as [radius,colat,lon] 
   '''

   if degrees:
      s[1] = np.radians(s[1])  
      s[2] = np.radians(s[2])

   x1 = s[0]*np.sin(s[1])*np.cos(s[2])
   x2 = s[0]*np.sin(s[1])*np.sin(s[2])
   x3 = s[0]*np.cos(s[1])

   x = [x1,x2,x3]

   if normalize:
      x /= np.linalg.norm(x)
   return x

def find_rotation_vector(s1,s2):
   '''
   Takes two vectors in spherical coordinates, and returns the cross product,
   normalized to one.
   '''

   x1 = spherical_to_cartesian(s1,degrees=True,normalize=True) 
   x2 = spherical_to_cartesian(s2,degrees=True,normalize=True)
   n = np.cross(x1,x2)
   n /= np.linalg.norm(n)
   return n

def find_rotation_angle(s1,s2,degrees=True):
   '''
   Finds the angle between two vectors in spherical coordinates
   
   params:
   s1,s2: vectors in spherical coordinates

   returns
   '''
   x1 = spherical_to_cartesian(s1,degrees=True,normalize=True) 
   x2 = spherical_to_cartesian(s2,degrees=True,normalize=True)
   if degrees:
      return np.degrees(np.arccos(np.clip(np.dot(x1,x2),-1.0,1.0)))
   else:
      return np.arccos(np.clip(np.dot(x1,x2),-1.0,1.0))

def km_per_deg_lon(latitude):
   '''
   returns how many km there are per degree of longitude, at a given latitude

   args:
       latitude: latitude in degrees
   '''
   latitude = np.radians(latitude)
   km_per_deg = (2*np.pi*6371.0*np.cos(latitude))/360.0
   return km_per_deg

def generate(phi=None, theta=None, rad=None):
    '''
    Generates a structured grid for a spherical section.
    Used for Mayavi visualization of Earth models.
    '''
    # Default values for the spherical section
    #if rad is None: rad = np.linspace(3490.0,6371.0,20)
    if rad is None: rad = np.linspace(1.0,2.0,50)
    if phi is None: phi = np.linspace(0,2*pi,50)
    if theta is None: theta = np.linspace(0.0,pi,50)

    # Find the x values and y values for each plane.
    #x_plane = (cos(phi)*theta[:,None]).ravel()
    #y_plane = (sin(phi)*theta[:,None]).ravel()
    #print "len x_plane = ", len(x_plane)

    # Allocate an array for all the points.  We'll have len(x_plane)
    # points on each plane, and we have a plane for each z value, so
    # we need len(x_plane)*len(z) points.
    len_points =  len(theta)*len(phi)*len(rad)
    points = np.empty([len_points,3])

    # Loop through the points and fill them with the
    # correct x,y,z values.
    count = 0

    print('rad',rad)
    print('theta',theta)
    print('phi',phi)

    for i in range(0,len(theta)):
       for j in range(0,len(phi)):
          for k in range(0,len(rad)):
             x = rad[k]*cos(phi[j])*sin(theta[i])
             y = rad[k]*sin(phi[j])*sin(theta[i])
             z = rad[k]*cos(theta[i])
             points[count,:] = x,y,z
             count += 1

    return points


def add_basaltic_core(m,f_core,rad,mindep,maxdep):
    '''
    adds a basaltic core to a plume model

    m: an instance of the 'model_2d' class
    f_core: basalt fraction within the plume stem
    rad: radius of basaltic cylinder (km)
    mindep: minimum depth of basaltic cylinder (km)
    maxdep: maximum depth of basaltic cylinder (km)
    '''
    for i in range(0,len(m.x)):
        if m.depth[i] > mindep and m.depth[i] < maxdep and m.x[i] < rad:
            m.f[i] = f_core

    m.f_array = np.reshape(m.f,(m.npts_depth,m.npts_theta))

def write_gmt_psxy_files(m,fname,field):
   '''
   writes scatter data to be plotted with psxy
   format is x(km), radius (km), value
   '''

   f = open(fname,'w')

   if field == 'T':
       vals = m.T
   elif field =='f':
       vals = m.f
   elif field =='vp':
       vals = m.vp
   elif field =='vs':
       vals = m.vs
   elif field =='rho':
       vals = m.rho
   elif field =='dvp':
       vals = m.dvp
   elif field =='dvs':
       vals = m.dvs
   elif field =='drho':
       vals = m.drho

   for i in range(0,len(m.theta)):
       f.write('{} {} {}'.format(m.x[i],6371.0-m.depth[i],vals[i])+'\n')

def full_cyl_to_half_cyl(m):
    n = deepcopy(m)
    n.x = n.x[0:int(len(n.x)/2)]
    n.y = n.y[0:int(len(n.y)/2)]
    n.T = n.T[0:int(len(n.T)/2)]
    n.f = n.f[0:int(len(n.f)/2)]
    n.theta = n.theta[0:int(len(n.depth)/2)]
    n.depth = n.depth[0:int(len(n.depth)/2)]

    i_stop = int(n.npts_theta / 2)

    n.theta_axis = n.theta_axis[0:i_stop]
    n.T_array = n.T_array[:,0:i_stop]
    n.f_array = n.f_array[:,0:i_stop]

    return n

def rotate_full_cyl(m,degrees):
    n = deepcopy(m)
    n.theta += degrees

    #th = np.deg2rad(degrees)
    #R = np.array([[np.cos(th),-1.0*np.sin(th)],[np.sin(th),np.cos(th)]])
    #pts = np.vstack((n.x,n.y))
    #rotated = R.dot(pts)
    #n.x = rotated[0,:]
    #n.y = rotated[1,:]

    i_roll = (int(degrees/n.dtheta) * n.npts_depth) * -1
    
    #rotate T and C vectors
    n.T = np.roll(m.T,i_roll)
    n.f = np.roll(m.f,i_roll)
    #try to rotate velocity and density fields
    try:
        n.vp = np.roll(m.vp,i_roll)
        n.vs = np.roll(m.vs,i_roll)
        n.rho = np.roll(m.rho,i_roll)
        n.dvp = np.roll(m.dvp,i_roll)
        n.dvs = np.roll(m.dvs,i_roll)
        n.drho = np.roll(m.drho,i_roll)
    except AttributeError:
        print('Vs, Vp, and Rho not found... please perform velocity conversion')

    for i in range(0,len(n.theta)):
        if n.theta[i] > 360.0:
            n.theta[i] = n.theta[i] - 360.0
        elif n.theta[i] < 0.0:
            n.theta[i] = n.theta[i] + 360.0

    i_roll = int(degrees/n.dtheta) * -1
    n.T_array = np.roll(m.T_array,i_roll,axis=1)
    n.f_array = np.roll(m.f_array,i_roll,axis=1)

    try:
       n.vp_array = np.roll(m.vp_array,i_roll,axis=1)
       n.vs_array = np.roll(m.vs_array,i_roll,axis=1)
       n.rho_array = np.roll(m.rho_array,i_roll,axis=1)
       n.dvp_array = np.roll(m.dvp_array,i_roll,axis=1)
       n.dvs_array = np.roll(m.dvs_array,i_roll,axis=1)
       n.drho_array = np.roll(m.drho_array,i_roll,axis=1)
    except AttributeError:
        print('Vs, Vp, and Rho not found... please perform velocity conversion')

    return n

def pile(wavelen,H,dv_in):
   '''
   wavelen = width of the base of the pile, in km
   H =  height of the pile from the CMB, in km
   dv_in = seismic velocity perturbation within the pile, in %
   '''
   m = model_2d()
   m.theta_axis = np.linspace(0.0,180.0,181)
   m.depth_axis = np.linspace(0.0,2891.0,100) 
   m.radius_axis = 6371.0 - m.depth_axis
   m.radius_axis = np.linspace(3480.0,6371.0,100)
   theta_array,depth_array = np.meshgrid(m.theta_axis,m.depth_axis)
   m.depth = np.ravel(depth_array)
   m.theta = np.ravel(theta_array)
   m.radius = 6371.0 - m.depth
   m.dvs_array = np.zeros(depth_array.shape)
   m.dvp_array = np.zeros(depth_array.shape)
   m.drho_array = np.zeros(depth_array.shape)

   rad_CMB = 3480.0

   k = 0
   for i in range(0,len(m.radius_axis)):
      km_per_deg = (2.0*np.pi*m.radius[i])/360.0
      for j in range(0,len(m.theta_axis)):
         d_in_km = m.theta_axis[j] * km_per_deg #distance from axis in km
         km_above_core = m.radius_axis[i] - rad_CMB 

         #horizontal distance between axis and ellipse
         dist_from_axis = np.sqrt(wavelen**2*(1.0-(km_above_core**2/H**2)))
 
         if d_in_km <= dist_from_axis and km_above_core <= H:

            m.dvs_array[i,j] = dv_in
            m.dvp_array[i,j] = dv_in
            m.drho_array[i,j] = dv_in

   return m
