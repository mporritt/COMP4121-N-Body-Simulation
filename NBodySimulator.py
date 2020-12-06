"""
N_body_simulator.py
by Michael Porritt
"""

import numpy as np
import vpython as vp
import time
import scipy.constants

from SummationMethods import Summation
from IntegrationMethods import Integration


class NBodySimulator():
    
    def __init__(self, summationClass, integrationClass):
        assert issubclass(summationClass, Summation)
        assert issubclass(integrationClass, Integration)
        
        self.summationClass = summationClass
        self.integrationClass = integrationClass
        
        
    def run(self, masses, pos, velocity=None, p=None, time_max=1e9, dt=1e5, G=None, return_info=False):
        """ 
        Run N-body simulation using the provided methods to calculate forces and integrate.

        Parameters
        ==========
        masses :     Array of masses of the bodies.
        pos :      Array of vector starting positions (numpy arrays).
        velocity : Array of vector velocities (numpy arrays). This or p must be given.
        p :        Array of vector momenta (numpy arrays). This or velocity must be given.
        time_max : Time to iterate up to in seconds. Defaults to 10^9.
        dt :       Time increment per loop in seconds. Defaults to 10^5.
        """
        t0 = time.perf_counter()
        num_bodies = len(masses)
        
        if velocity is None:
            if not p: raise TypeError("Neither velocity nor momentum were given")
            velocity = [p[i] / masses[i] for i in range(num_bodies)]
        
        
        num_iterations = int(time_max // dt)
        positions = np.zeros((num_iterations, num_bodies, 3)) # An array of vector positions for each time-step
        positions[0] = pos
        
        if not G: G = scipy.constants.G
        summation_object = self.summationClass(masses, G)
        integration_object = self.integrationClass(positions[0], velocity, dt)
        
        for i in range(num_iterations - 1):
            # Get a list of acceleration vectors from the given force summation method
            net_acceleration = summation_object.get_accelarations(positions[i])
            
            # Integrate using the given acceleration
            positions[i+1] = integration_object.integrate(net_acceleration)
            
            
        
        time_tot = time.perf_counter() - t0
        print(f"N-body simulation took {time_tot} seconds.")
        
        if return_info:
            velocities = np.diff(positions, axis=0)
            positions = positions[:-1]
            E = energy(positions, velocities, masses, G)
            P = momentum(velocities, masses)
            return (positions, E, P, time_tot)
        
        return positions
        
        
    @staticmethod
    def animate(positions, masses=None, radius=None, max_fps=None, time=None):
        if not max_fps:
            if time: 
                max_fps = len(positions) / time
            else: 
                max_fps = 200
        
        radius_to_mass_ratio = 1e-20
        if not radius:
            radius = [m * radius_to_mass_ratio for m in masses]
        
        
        scene = vp.canvas()
        #scene.forward = vp.vector(0,-.3,-1)
        
        
        num_bodies = len(positions[0])
        spheres = []
        for i in range(num_bodies):
            sphere = vp.simple_sphere(
                    pos = vp.vector(positions[0][i][0], positions[0][i][1], positions[0][i][2]), 
                    radius = radius[i],
                    color = vp.color.yellow,
                    emissive = True,
                    #make_trail = True, trail_type = 'points', interval = 10, retain = 50,
                )
            spheres.append(sphere)
            
        for pos in positions:
            vp.rate(max_fps)
            
            for i in range(len(spheres)):
                spheres[i].pos = vp.vector(pos[i][0], pos[i][1], pos[i][2])
        return
    
    
    def run_preset(self, preset):
        
        if preset == "figure 8":
            masses = [1, 1, 1]
            radius = [0.05, 0.05, 0.05]
            pos = [np.array([-0.97000436, 0.24308753,0]), np.array([0.97000436, -0.24308753,0]), np.array([0,0,0])]
            velocity = [np.array([-0.93240737/2, -0.86473146/2,0]), 
                        np.array([-0.93240737/2, -0.86473146/2,0]), 
                        np.array([0.93240737, 0.86473146,0])]

            positions = self.run(masses, pos, velocity=velocity, time_max=50, dt=0.01, G=1)
            NBodySimulator.animate(positions, radius=radius, time=50)
            
        return
    
    
    
def energy(positions, velocities, masses, G):
    KE = np.zeros(len(positions))
    
    for i in range(len(KE)):
        for n in range(len(masses)):
            KE[i] += 0.5 * masses[n] * np.dot(velocities[i][n],velocities[i][n])
            
    GPE = np.zeros(len(positions))
    
    for i in range(len(GPE)):
        # to find the GPE of the system at each point, consider bringing each particle in from infinity one by one
        for n in range(len(masses)):
            for m in range(n):
                r = np.subtract(positions[i][n], positions[i][m])
                dist = r.dot(r) ** 0.5
                GPE[i] += - G * masses[n] * masses[m] / dist
    
    E = np.add(KE, GPE)
    return E
    
    
    
def momentum(velocities, masses):
    P = np.zeros((len(velocities), 3))
    
    for i in range(len(P)):
        for n in range(len(masses)):
            P[i] += masses[n] * velocities[i][n]
            
    return P
            
            
            
            
    
    