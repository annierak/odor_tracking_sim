#Trap models detached from odor model.
class TrapModel(object):
    DefaultParam = {
            'source_locations' : [(0,0),],
            'source_strengths' : [ 1.0, ],
            'trap_radius'      : 10.0,
            }

    def __init__(self,param={}):
        self.param = dict(self.DefaultParam)
        self.param.update(param)

    def check_if_in_trap(self,pos):
        for trap_num, trap_loc in enumerate(self.param['source_locations']):
            dist = distance(pos, trap_loc)
            if dist <= self.param['trap_radius']:
                return True, trap_num, trap_loc
        return False, None, None

    def is_in_trap(self,pos):
        flag, trap_num, trap_loc = self.check_if_in_trap(pos)
        return flag
