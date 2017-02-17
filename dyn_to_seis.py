import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d, interp2d
from seis_tools.models import models_1d,models_3d
from scipy.misc import imresize

class model_2d(object):
    #TODO: write the following functions...
    #      rotate_cylinder
    def init():
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

    def read_2d(self,path_to_file,fmt='pvk',**kwargs):
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

        kwargs:

           bf_scaling: float (optional, default = 8.0)
              value corresponding to 100% basalt concentration in dynamic simulations.
              composition field will be normalized so that 1.0 corresponds to 100% basalt.
        '''
        bf_scaling = kwargs.get('bf_scaling',8.0)

        model = np.loadtxt(path_to_file)
        if fmt=='pvk':
           x = model[:,0]
           y = model[:,1]
           theta = model[:,2]
           depth = model[:,3]
           Tpot = model[:,4]
           f = model[:,5]

        self.x = x
        self.y = y
        self.theta = theta
        self.depth = depth
        self.T = Tpot
        self.f = f

        #remove 'outliers' that are over 100% basalt
        for i in range(0,len(self.f)):
            if self.f[i] > bf_scaling:
                self.f[i] = bf_scaling

        #normalize basalt composition field
        self.f /= bf_scaling

        #reshape scatter data into an array
        dtheta = np.max((np.diff(self.theta)))
        ddepth = np.max(np.abs(np.diff(self.depth)))
        self.dtheta = dtheta
        self.ddepth = ddepth
        self.theta_axis = np.arange(np.min(theta),np.max(theta)+dtheta,dtheta)
        self.depth_axis = np.arange(np.min(depth),np.max(depth)+ddepth,ddepth)
        npts_theta = len(self.theta_axis)
        npts_depth = len(self.depth_axis)
        self.T_array = np.reshape(self.T,(npts_depth,npts_theta))
        self.f_array = np.reshape(self.f,(npts_depth,npts_theta))

        #get 1d pressure profile from PREM
        prem = models_1d.prem()
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

    def velocity_conversion(self,lookup_tables,composition,**kwargs):
        '''
        performs the velocity conversion.

        args:

           lookup_tables: str
                          path to h5py lookup tables

           composition: str
                        one of either 'pyrolite', 'harzburgite', morb', or, 'mixture'
                        'mixture' uses a mechanical mixture of morb and harzburgite

        kwargs:
           ref_theta_range: tuple
                            defines the theta range of what we consider to be 
                            'average' mantle.  default is the entire model.
                            The entire model may not be desirable for the half
                            cylinder models if certain depths are heavily biased
                            by the presence strong dynamical structure
        '''

        ref_theta_range = kwargs.get('ref_theta_range','full')
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
        for i in range(0,len(self.depth_axis)):
            for j in range(0,len(self.theta_axis)):
                P_here = self.P[::-1][i]
                T_here = self.T_array[i,j]
                f_here = self.f_array[i,j]
                vp_harz = harz_vp_interpolator(P_here,T_here)
                vs_harz = harz_vs_interpolator(P_here,T_here)
                rho_harz = harz_rho_interpolator(P_here,T_here)
                vp_morb = morb_vp_interpolator(P_here,T_here)
                vs_morb = morb_vs_interpolator(P_here,T_here)
                rho_morb = morb_rho_interpolator(P_here,T_here)
                self.vp_array[i,j] = f_here*vp_morb + (1-f_here)*vp_harz
                self.vs_array[i,j] = f_here*vs_morb + (1-f_here)*vs_harz
                self.rho_array[i,j] = f_here*rho_morb + (1-f_here)*rho_harz

        if ref_theta_range == 'full':
            vp_avg = np.average(self.vp_array,axis=1)
            vs_avg = np.average(self.vs_array,axis=1)
            rho_avg = np.average(self.rho_array,axis=1)
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

        self.vp = np.ravel(self.vp_array) 
        self.vs = np.ravel(self.vs_array) 
        self.rho = np.ravel(self.rho_array) 
        self.dvp = np.ravel(self.dvp_array) 
        self.dvs = np.ravel(self.dvs_array) 
        self.drho = np.ravel(self.drho_array) 

    def twoD_to_threeD(self,dz,dlat,dlon,par='dvs'):
        #first interpolate 2D grid to new grid (if a different resolution is required)
        npts_rad   = abs(int(np.max(self.depth_axis)-np.min(self.depth_axis))/dz)
        npts_theta = abs(int(np.max(self.theta_axis)-np.min(self.theta_axis))/dlat)+1
        npts_phi = int(360.0/dlon)+1
        print npts_rad, npts_theta,npts_phi

        field = imresize(self.dvs_array,(npts_rad,npts_theta))
        print 'shape of field', field.shape
        mod_3d = models_3d.model_3d(drad=dz,latmin=-90,latmax=90+dlat,dlat=dlat,
                                    lonmin=0,lonmax=360+dlon,dlon=dlon)
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

    def write_2d_output(self,filename,fmt='pvk'):
        if fmt == 'pvk':
            np.savetxt(filename,np.c_[self.x,self.y,self.theta,self.depth,self.T,self.f,self.dvp,self.dvs,self.drho], fmt='%3f')
