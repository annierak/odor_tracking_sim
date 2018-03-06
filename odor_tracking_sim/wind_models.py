import scipy
import sys


class WindField(object):
    """
    Wind model specified by wind angle, speed, and time (optional)
    """

    DefaultParam = {'evolving':False, 'angle': 0.0, 'speed': 1.0, 'wind_dt':None, 'dt':0.25}

    def __init__(self,param={}):

        self.param = dict(self.DefaultParam)
        self.param.update(param)

        self.angle = param['angle']
        self.speed = param['speed']
        self.evolving = self.param['evolving']
        self.wind_dt = self.param['wind_dt']
        self.dt = self.param['dt']

    def value(self,t,x,y):
        if self.evolving:
            index = int(scipy.floor(t/self.wind_dt))
            vx = self.speed[index]*scipy.cos(self.angle[index])
            vy = self.speed[index]*scipy.sin(self.angle[index])
        else:
            vx = self.speed*scipy.cos(self.angle)
            vy = self.speed*scipy.sin(self.angle)
        if type(x) == scipy.ndarray:
            if x.shape != y.shape:
                raise(ValueError,'x.shape must equal y.shape')
            vx_array = scipy.full(x.shape,vx)
            vy_array = scipy.full(y.shape,vy)
            return vx_array, vy_array
        else:
            return vx, vy
    def value_polar(self,t,x,y):
        if self.evolving:
            index = int(scipy.floor(t/self.wind_dt))
            try:
                speed,angle = self.speed[index],self.angle[index]
            except IndexError:
                print('Out of wind data')
                sys.exit()
        else:
            speed,angle = self.speed,self.angle
        return speed,angle
#        if type(x) == scipy.ndarray:
#            if x.shape != y.shape:
#                raise(ValueError,'x.shape must equal y.shape')
#            speed_array = scipy.full(x.shape,speed)
#            angle_array = scipy.full(y.shape,angle)
#            return speed_array, angle_array
#        else:
#            return speed,angle
