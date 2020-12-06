"""
Integration.py
by Michael Porritt
"""

from abc import ABC, abstractmethod
import numpy as np



class Integration(ABC):
    @abstractmethod
    def __init__(self, position_i, velocity_i, dt):
        pass
    
    @abstractmethod
    def integrate(self, net_acceleration):
        pass

    
class EulerIntegration(Integration):
    def __init__(self, position_i, velocity_i, dt):
        self.x = position_i
        self.v = velocity_i
        self.dt = dt
        
        
    def integrate(self, net_acceleration):
        self.v = self.v + net_acceleration * self.dt
        self.x = self.x + self.v * self.dt
        return self.x
    
    
class LeapfrogIntegration(Integration):
    def __init__(self, position_i, velocity_i, dt):
        self.x = position_i
        self.v_half = velocity_i
        self.first_iteration = True
        self.dt = dt
        
        
    def integrate(self, net_acceleration):
        if self.first_iteration:
            self.v_half = self.v_half + net_acceleration * (self.dt / 2)
            self.first_iteration = False
        else:
            self.v_half = self.v_half + net_acceleration * self.dt
        
        self.x = self.x + self.v_half * self.dt
        
        return self.x
        
    
# class RangeKuttaIntegration(Integration):