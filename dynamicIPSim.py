"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""

# File: dynamicIPSim.py


import numpy as np
from random import randint, uniform, choice
from operator import add
from stochastic.processes.continuous import BrownianMotion, FractionalBrownianMotion

import sys
sys.path.insert(0, 'C:/Users/Dina Ratsimandresy/Documents/GitHub/PSFToolbox-Python')
# from math_tools import create_grid

import pandas


def simulate_2d_randomwalk(nsteps=10000, stepsize=1):
    """
    code source: https://medium.com/@mlblogging.k/simulating-brownian-motion-and-stock-prices-using-python-17b6b4bd2a1
    """

    deltas = [ (0,-1*stepsize), (-1*stepsize,0), (0,1*stepsize), (1*stepsize,0) ]

    steps = [ list(choice(deltas)) for i in range(nsteps) ]
    steps = np.array(steps)
    steps = np.cumsum(steps,axis=0)
    y = list(steps[:,1])
    x = list(steps[:,0])

    return x, y

def simulate_2d_bm(m, nsteps=100, t=0.01):

    x0=np.empty((m,nsteps)) 
    y0=np.empty((m,nsteps)) 
    for k in range(m):
        x = np.cumsum([ np.random.randn()*np.sqrt(t) for i in range(nsteps) ])
        y = np.cumsum([ np.random.randn()*np.sqrt(t) for i in range(nsteps) ])

        x0[k,:]=x
        y0[k,:]=y
    return x0,y0

# Brownian motion using the stochastics package
def simulate_2d_bm_stochastics(m, nsteps=100, drift=1,scale=1,dt=1): 
    bm = BrownianMotion(drift, scale, dt)
    x0=np.empty((m,nsteps)) 
    y0=np.empty((m,nsteps)) 
    for k in range(m):
        x = bm.sample(nsteps-1)
        y = bm.sample(nsteps-1)

        x0[k,:]=x
        y0[k,:]=y
    return x0,y0

# Fractional Brownian motion using the stochastics package
def simulate_2d_fbm(m, nsteps, hurst=0.5, dt=1):
    fbm = FractionalBrownianMotion(hurst, dt)

    x0=np.empty((m,nsteps)) 
    y0=np.empty((m,nsteps)) 
    for k in range(m):
        x = fbm.sample(nsteps-1)
        y = fbm.sample(nsteps-1)

        x0[k,:]=x
        y0[k,:]=y

    return x0, y0

def compute_xy_pos(n, m, sxy, dt=0.01, method='Brownian', edgeex=0.1, drift=1, scale=1, hurst=0.5,rand_pos_particle_x=None,rand_pos_particle_y=None):
    """
    Function computes random spatial position given the input arguments
    Input :
    n : int , number of time points
    m : int , number of particles
    sxy : a list of two integers , size in x and y
    dt : float 
        time step
    method : string
        method chosen to use to compute the random position. 
        Options include: 'Brownian', 'Brownian_Stochastics' from the stochastics package, 'RandomWalk', 'FractionalBM' for FractionalBrownianMotion from the stochastics package
    edgeex : float between 0 and 1, 1 exclueded ie [0,1[
        percentage of the grid on which no particles are placed on 
    drift, scale : parameters of the 'Brownian_Stochastics' methods
        help : stochastic.processes.continuous.BrownianMotion 
    hurst : parameter of the fractional Brownian motion
        help : stochastic.processes.continuous.FractionalBrownianMotion

    """

     # random position in x and y of the particles, excluding 10% position at the edge
    if rand_pos_particle_x==None:
        rand_pos_particle_x = [randint(np.ceil(sxy[0]*edgeex), np.floor(sxy[0]*(1-edgeex))) for p in range(m)]
    if rand_pos_particle_y==None:
        rand_pos_particle_y = [randint(np.ceil(sxy[1]*edgeex), np.floor(sxy[1]*(1-edgeex))) for p in range(m)]

    match method:
        case 'Brownian':
            x0, y0 = simulate_2d_bm(m, n, dt)
        case 'Brownian_Stochastics':
            x0, y0 = simulate_2d_bm_stochastics(m, n, drift, scale, dt)
        case 'FractionalBM':
            x0, y0 = simulate_2d_fbm(m, n, hurst, dt)
        case 'RandomWalk':
            x0, y0 = simulate_2d_randomwalk(n, stepsize=1) # change code source to account for the value of m
        case _:
            return "Input method is not supported. Choose Brownian or RandowmWalk" # check if this will work eventually
    
    for nk in range(n):
        for mk in range(m):
            xpos = rand_pos_particle_x[mk]+x0[mk,nk]
            ypos = rand_pos_particle_y[mk]+y0[mk,nk]
            # overwrite the position
            x0[mk,nk] = xpos
            y0[mk,nk] = ypos

    return x0, y0

def compute_beads_params(m,particle_size_range = [1.5,5],bead_circularity=1,x0=None,y0=None):
    """
    Compute the parameters of the bead at the first time point 
    --------
    Input: 
    m: int 
        number of beads
    particle_size_range: list of two floats
        range of the particle size. First entry: lower bound. Second entry: upper bound. 
    bead_circularity: 1 or 0
        command to compute a circular bead (1) or non-circular bead (0). this affects the values of sigma_x and sigma_y
    x0: list or 1D array of length equal to m
        x-position of the beads at t=0s before the motion of the beads takes off
    y0: list or 1D array of length equal to m
        y-position of the beads at t=0s before the motion of the beads takes off
    --------
    Output: 
    A dataframe with 5 columns
    amp: amplitude of the bead intensity
    sigma_x: Gaussian width along x
    sigma_y: Gaussian width along y
    mu_x: x-position at t = 0s
    mu_y: y-position at t = 0s
    """

    sigma_x, sigma_y = sigma_xy(m, particle_size_range,bead_circularity) 
    amp = generate_amp(m, sigma_x, 5)
    bead_params = pandas.DataFrame({'amp':amp,'sigma_x':sigma_x,'sigma_y':sigma_y})
    if x0.all() != None:
        bead_params['x0_init'] = x0
    if y0.all() != None:
        bead_params['y0_init'] = y0

    return bead_params


def sigma_xy(m, particle_size_range = [1.5,5],bead_circularity=1):
    """
    Generate random values of the sigma for the Gaussian distribution for each particle. The sigma size is delimited by the particle size. 

    Input :
    m : int
        number of particles
    particle_size_range : a list of two floats
        range of the size of the particles in pixels
    
    Outputs : 
    sigma_x, sigma_y : width of the Gaussian distribution
    """
    sigma = [particle_size_range[p]/(2*np.sqrt(2*np.log(2))) for p in range(2)]
    sigma_x = [uniform(sigma[0],sigma[1]) for p in range(m)]
    if bead_circularity==1:
        sigma_y = sigma_x 
    else:
        sigma_y = [uniform(sigma[0],sigma[1]) for p in range(m)] # we firstly assume a circular particle
    return sigma_x, sigma_y

def generate_amp(m, sigma_x, factor=5):
    """
    Generate a range of peak intensity using the width of the particles. Lower peak value is attributed to smaller width.
    Input :
    m : int, number of particles
    sigma_x : width of the particles

    """
    amp = [sigma_x[p]*factor for p in range(m)]
    return amp

def decay_image(myimage, gamma=0.0009):
    """
    Add decay in the image through time
    I(t) = I(0)exp(-gamma*t)
    Input :
    myimage: 3D image of dimension XYZ
    gamma : float between 0 and 1, decay factor
    dt : time step
    Output:
    bleached_image
    """
    myimage = np.asarray(myimage)
    sz = myimage.shape
    d = np.expand_dims(np.asarray([np.exp(-gamma*t) for t in range(sz[2])]), axis=(1,2)) # decay factor
    d = np.transpose(d,(2,1,0))
    myimage_decay = myimage*d
    return myimage_decay

def generate_images(x0, y0, sxy, bead_params_all, gamma=0.0009, N=1000):
    """
    Use the brownian function defined in the function brownian to generate 2D Brownian motion, 


    Input :
    ---------
    x0, y0 : arays of float numbers with size equal to m x n, 
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    m : int
        Number of particles whose positions in time and images are to simulate. 
    n : int
        The number of step distance to take.
    sxy : list of two integers
        grid size in x and y directions on which the images of the particles to be placed on.
    bead_params_all: pandas dataframe of 3 or 5 columns
        ['amp']: first column, amplitude
        ['sigma_x']: width of the bead along x
        ['sigma_y']: width of the bead along y
        ['mu_x']: center position along the x-axis [optional, not necessarily required for the computation here]
        ['mu_y']: center position along the y-axis [optional, not necessarily required for the computation here]
    gamma : float between 0 and 1
        Decaying factor when the photobleaching in the data during the acquisition time is considered, 0 yields to no bleaching.
    N : int
        maximal number of photons.

    Outputs :
    -------
    noisyimg : 3D array of size sxy[0] x sxy[1] x n
        Similuted 3D image containing the motion of the m particles, a Poisson noise is added at the end of the simulation.

    """

    m,n = x0.shape # m: number of particles, n: number of time points
    
    sigma_x = bead_params_all['sigma_x']
    sigma_y = bead_params_all['sigma_y']

    # amplitude // bigger particle has higher peak intensity
    amp = bead_params_all['amp']

    destgrid = np.zeros((sxy[0],sxy[1],n)) 
    xx, yy = create_grid(sxy)

    grid_first = np.zeros((sxy[0],sxy[1]))

    for nk in range(n): #
        grid_k = grid_first+0

        for mk in range(m):
            # compute the Gaussian function
            grid_k = grid_k + gauss2D(xx,yy,[x0[mk,nk],y0[mk,nk]],[sigma_x[mk],sigma_y[mk]],amp[mk])
        destgrid[:,:,nk] = grid_k
    # decay in the intensity over time due to photobleaching
    decayed = decay_image(destgrid,gamma)
    # maximum number of photons acquired in the image
    decayed = decayed*N/np.max(decayed)  
    # add Poisson noise
    noisyimg = np.random.poisson(decayed)

    return noisyimg

def create_grid(sxy):
    """
    Input argument :
    sxy : list of two integers
        Grid size
    Output :
    xx, yy : arrays of size sxy
        Arrays of elements increasing from 0 to sxy-1 in both direction horizontal and vertical respectively

    To add: if the input sxy is only one integer but not a list
    """
    xx, yy = np.meshgrid(np.linspace(0,sxy[0]-1,sxy[0]),np.linspace(0,sxy[1]-1,sxy[1]))
    return xx, yy

def gauss2D(xx,yy, mu_xy, sigma_xy, amp):
    """
    Input arguments :
    sxy : a list of two integers
        Grid size on which the Gaussian distribution will be calculated
    mu_xy : a list of two floats
        Mean or center position in x and y of the Gaussian distribution
    sigma_xy : a list of two floats
        Width of the distribution in x and y
    amp : float
        Amplitude 
    
    Output :
    2D Gaussian distribution with the function below

    g(x, y; mu_x, mu_y, sigma_x, sigma_y) = add here

    """
    c = 1/(2*np.pi*sigma_xy[0]*sigma_xy[1]) # constant
    return amp*c*np.exp(-((xx-mu_xy[0])**2/(2*(sigma_xy[0])**2)  + (yy-mu_xy[1])**2/(2*(sigma_xy[1])**2)))


