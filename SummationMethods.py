"""
SummationMethods.py
by Michael Porritt
"""

from abc import ABC, abstractmethod
import numpy as np

from BarnesHutCube import BarnesHutCube


BH_opening_angle = 30



def Fg(p1, p2, m1, m2, G):
    """ Calculate the force on body 1 due to body 2 """
    
    r = np.subtract(p2, p1)
    r_mag3 = r.dot(r) ** 1.5
    
    if r_mag3 < 1e-05: 
        return np.zeros(3) # Prevent self-reference and division going to infinity
    
    F_12 = G * m1 * m2 * r / r_mag3
    return F_12



class Summation(ABC):
    
    def __init__(self, masses, G):
        self.masses = masses    # List of masses of the bodies
        self.N = len(masses)
        self.G = G
        
        super().__init__()
    
    @abstractmethod
    def get_accelarations(self, positions):
        pass
    

    
class DirectSummation(Summation):
        
    def get_accelarations(self, positions):
        net_acc = np.zeros((self.N, 3))

        for i in range(self.N):
            for j in range(i+1, self.N):

                # Calculate the gravitational Force
                F_ij = Fg(positions[i], positions[j], self.masses[i], self.masses[j], self.G)

                # Add to the net force of each object
                net_acc[i] += F_ij / self.masses[i]
                net_acc[j] -= F_ij / self.masses[j]
            
        return net_acc
        
        

class BarnesHutSummation(Summation):
    
    def get_accelarations(self, positions):
        max_vals = [-float('inf')] * 3
        min_vals = [ float('inf')] * 3
        for p in positions:
            max_vals = [max(max_vals[i], p[i]) for i in range(3)]
            min_vals = [min(min_vals[i], p[i]) for i in range(3)]
        
        cube_pos = min_vals
        size = max(np.subtract(max_vals, min_vals))
        
        ### max_coord = max([max(abs(p)) for p in positions])
        
        root = BarnesHutCube.construct(
            cube_pos, 
            size, 
            positions, 
            self.masses
        )
        
        accelerations = np.array([
            root.get_acceleration(p, BH_opening_angle, Fg, self.G)
            for p in positions
        ])
        
        return accelerations