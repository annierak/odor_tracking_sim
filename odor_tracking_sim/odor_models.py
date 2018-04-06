from __future__ import print_function

import scipy
import scipy.special
import matplotlib.pyplot as plt
import wind_models

from .utility import shift_and_rotate
from .utility import rotation_matrix
from .utility import distance

class FakeDiffusionOdorField(object):

    DefaultWindParam = {'evolving':False, 'angle': 0.0, 'speed': 1.0, 'wind_dt':None, 'dt':0.25}
    DefaultWindField = wind_models.WindField(param =DefaultWindParam)

    DefaultParam = {
            'wind_field'       : DefaultWindField,
            'diffusion_coeff'  : 0.001,
            'epsilon'          : 0.01}#,
#            'source_locations' : [(0,0),],
#            'source_strengths' : [ 1.0, ],
#            'epsilon'          : 0.01,
#            'trap_radius'      : 10.0,
#            }

    def __init__(self,traps,param={}):
        self.param = dict(self.DefaultParam)
        self.param.update(param)
        if  type(self.param['wind_field']) != wind_models.WindField:
            raise(ValueError, 'wind_field must of type wind_models.WindField')
        self.traps = traps

#    def check_if_in_trap(self,pos):
#        for trap_num, trap_loc in enumerate(self.param['source_locations']):
#            dist = distance(pos, trap_loc)
#            if dist <= self.param['trap_radius']:
#                return True, trap_num, trap_loc
#        return False, None, None


#    def is_in_trap(self,pos):
#        flag, trap_num, trap_loc = self.check_if_in_trap(pos)
#        return flag


    def value(self,t,x,y):
        """
        Returns odor concentration as a function of time and position.
        """
        # Extract parameters
        wind_field = self.param['wind_field']
        if wind_field.evolving:
            wind_speed,wind_angle = wind_field.value_polar(t,x,y)
        else:
            wind_angle = self.param['wind_field'].angle
            wind_speed = self.param['wind_field'].speed
        traps = self.traps
        source_locations = traps.param['source_locations']
        source_strengths = traps.param['source_strengths']
        dcoeff = self.param['diffusion_coeff']
        epsilon = self.param['epsilon']

        # Calculate odor value
        if type(x) == scipy.ndarray:
            if x.shape != y.shape:
                raise RuntimeError, 'shape of x and y must be the same'
            odor_value = scipy.zeros(x.shape)

            for src_loc, src_val in zip(source_locations,source_strengths):
                xx, yy = shift_and_rotate((x,y), src_loc, -wind_angle)
                tt = xx/wind_speed
                mask = tt >= 0
                term_0 = src_val*scipy.sqrt(4.0*scipy.pi*dcoeff*epsilon)
                term_1 = 1.0/scipy.sqrt(4.0*dcoeff*scipy.pi*(tt[mask]+epsilon))
                term_2 = scipy.exp((-yy[mask]**2)/(4.0*dcoeff*(tt[mask]+epsilon)))
                odor_value[mask] +=  term_0*term_1*term_2
        else:
            odor_value = 0.0
            for src_loc, src_val in zip(source_locations,source_strengths):
                xx, yy = shift_and_rotate((x,y), src_loc, -wind_angle)
                tt = xx/wind_speed
                if (tt < 0):
                    odor_value += 0.0
                else:
                    term_0 = src_val*scipy.sqrt(4.0*scipy.pi*dcoeff*epsilon)
                    term_1 = 1.0/scipy.sqrt(4.0*dcoeff*scipy.pi*(tt+epsilon))
                    term_2 = scipy.exp((-yy**2)/(4.0*dcoeff*(tt+epsilon)))
                    odor_value +=  term_0*term_1*term_2
        return odor_value

    def value_to_mesh(self,t,plot_param):
        xnum = plot_param['xnum']
        ynum = plot_param['ynum']
        xlim = plot_param['xlim']
        ylim = plot_param['ylim']
        x_values = scipy.linspace(xlim[0], xlim[1], xnum)
        y_values = scipy.linspace(ylim[0], ylim[1], ynum)
        x_mesh, y_mesh = scipy.meshgrid(x_values,y_values,indexing='xy')
        odor_value = self.value(t,x_mesh.flatten(), y_mesh.flatten())
        odor_value = scipy.reshape(odor_value,x_mesh.shape)
        odor_mesh = scipy.flipud(odor_value)
        return odor_mesh

    def plot(self, t,plot_param):
        xlim = plot_param['xlim']
        ylim = plot_param['ylim']
        cmap = plot_param['cmap']


        try:
            threshold = plot_param['threshold']
        except KeyError:
            threshold = None

        try:
            fignums = plot_param['fignums']
        except KeyError:
            fignums = (1,2)

        odor_mesh = self.value_to_mesh(t,plot_param)

        plt.figure(fignums[0])
        image = plt.imshow(odor_mesh, extent=(xlim[0],xlim[1],ylim[0],ylim[1]),cmap=cmap)
        for x,y in self.traps.param['source_locations']:
            #plt.plot([x],[y],'ok')
            s = scipy.linspace(0,2.0*scipy.pi,100)
            cx = x + self.traps.param['trap_radius']*scipy.cos(s)
            cy = y + self.traps.param['trap_radius']*scipy.sin(s)
            plt.plot(cx,cy,'k')
        plt.plot([0],[0],'ob')
        plt.grid('on')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.xlim(xlim)
        plt.ylim(ylim)
        if threshold is not None:
            plt.figure(fignums[1])
            odor_thresh = odor_value >= threshold
            plt.imshow(odor_thresh, extent=(xlim[0],xlim[1],ylim[0],ylim[1]),cmap=cmap)
            for x,y in self.traps.param['source_locations']:
                plt.plot([x],[y],'.k')

            plt.plot([0],[0],'ob')
            plt.grid('on')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title('Odor Concentration >= {0}'.format(threshold))
        return image
