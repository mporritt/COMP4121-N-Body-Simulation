"""
BarnesHutCube.py
by Michael Porritt
"""

import numpy as np

                
class BarnesHutCube():
    def __init__(self, cube_pos, size, mass, CoM, octants):
        
        self.cube_pos = cube_pos # Position of the most negative corner of the cube
        self.size = size         # Side length of the cube
        self.mass = mass         # Total contained mass
        self.CoM = CoM           # Centre of mass
        self.octants = octants   # List of sub-BHCubes / octants
    
    
    def get_acceleration(self, position, opening_angle, Fg, G):
        """ Get the Barnes-Hut net force from this cube acting at the given position with the given opening angle. """
        net_acc = np.zeros(3)
        
        if self.is_within_angle(position, opening_angle):
            net_acc = Fg(position, self.CoM, 1, self.mass, G)
            
        else:
            for o in self.octants:
                if o is not None:
                    net_acc = np.add( net_acc, o.get_acceleration(position, opening_angle, Fg, G) )
                
        return net_acc
        
    
    def is_within_angle(self, position, opening_angle):
        if self.is_leaf(): return True
        if self.contains(position): return False
        
        #cos_theta = self.get_angular_extent(position)
        #cos_theta_limit = np.cos(opening_angle * np.pi / 180)
        #return (cos_theta > cos_theta_limit)
        
        centre = np.add(self.cube_pos, self.size * np.ones(3))
        r = np.subtract(centre, position)
        dist_to_centre = np.sqrt(r.dot(r))
        
        extent = self.size * 3 ** 0.5
        
        angle_rads = opening_angle * np.pi / 180
        max_extent = np.tan(angle_rads / 2) * dist_to_centre * 2
        
        return extent < max_extent
    
    
    def is_leaf(self):
        return (self.octants.count(None) == len(self.octants))
    
        
    def contains(self, position):
        return all([ (position[i] >= self.cube_pos[i]) and 
                     (position[i] < self.cube_pos[i] + self.size) 
                      for i in range(3) ])
    
    
    def get_angular_extent(self, position):
        corners = self.get_corners()
        cos_theta_min = 1
        
        for i in range(8):
            for j in range(i+1, 8):
                r_1 = np.subtract(corners[i], position)
                r_2 = np.subtract(corners[j], position)
                r_1_mag = np.sqrt(r_1.dot(r_1))
                r_2_mag = np.sqrt(r_2.dot(r_2))
                
                if (r_1_mag==0 or r_2_mag==0):
                    continue
                
                cos_theta = r_1.dot(r_2) / (r_1_mag * r_2_mag)
                cos_theta_min = min( cos_theta_min, cos_theta )
        return cos_theta_min
    
    
    def get_angular_extent_quick(self, position):
        diagonal
    
    
    def get_corners(self):
        return [ 
                np.add(self.cube_pos, int_to_bool_array(i) * self.size) 
                for i in range(8)
               ]
    
    def get_num_bodies(self):
        if self.is_leaf(): return 1
        return sum([o.get_num_bodies() for o in self.octants if o is not None])
    
    def get_num_cubes(self):
        return 1 + sum([o.get_num_cubes() for o in self.octants if o is not None])
    
    
    @classmethod
    def construct(cls, cube_pos, size, positions, masses):
        if len(positions) == 0: return None
        
        # print(f"constructing BarnesHutCube : cube_pos={cube_pos}, size={size}, {len(positions)} bodies inside")
        
        mass = sum(masses)
        CoM = np.matmul(np.transpose(positions), masses) / mass
        octants = cls.get_octants(cube_pos, size, positions, masses)
        
        return cls(cube_pos, size, mass, CoM, octants)
        
    
    @classmethod
    def get_octants(cls, cube_pos, size, positions, masses):
        if len(positions) <= 1: 
            return [None]*8
        
        inds = [[] for i in range(8)]
        
        for i in range(len(positions)):
            octant_number = get_octant_number(cube_pos, size, positions[i])
            inds[octant_number].append(i)
        
        octants = [None]*8
        for n in range(8):
            octants[n] = cls.construct(np.add(cube_pos, int_to_bool_array(n) * size/2),
                                       size/2,
                                       [ positions[i] for i in inds[n] ], 
                                       [ masses[i] for i in inds[n] ]
                                      )
        return octants
            
        
        
def get_octant_number(cube_pos, size, position):
    octant_number = int(position[0] > (cube_pos[0] + size/2)) * 2 ** 0 + \
                    int(position[1] > (cube_pos[1] + size/2)) * 2 ** 1 + \
                    int(position[2] > (cube_pos[2] + size/2)) * 2 ** 2
    return octant_number

            
def int_to_bool_array(num):
    bool_array = [bool(num & (1<<n)) for n in range(3)]
    return np.array([int(b) for b in bool_array])
                
                
                