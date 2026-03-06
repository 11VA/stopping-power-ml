"""Tools for computing the stopping distance of a projectile"""

from scipy.optimize import minimize_scalar
from scipy.integrate import RK45, LSODA, DOP853, Radau, BDF
from time import perf_counter, time
from copy import copy
import pandas as pd
import numpy as np
import tempfile
import keras
from ._stepper import _velocity_verlet, _rk4

import functools
print = functools.partial(print, flush=True)

def create_force_calculator_given_displacement(cell, model, featurizers, start_pos, traj_dir):
    """Create a function that computes the stopping force given displacement and current velocity
    
    :param start_pos: [float], starting point in conventional cell fractional coordinates
    :param traj_dir: [float], directional of travel in cartesian coordinate
    :return: (float, float)->float Takes displacement in distance units and velocity magnitude and computes force
    """
    
    # Get the trajectory direction as a unit vector
    traj_dir = np.divide(traj_dir, np.linalg.norm(traj_dir))
    
    # Convert the start point to Cartesian coordinates
    start_pos = cell.conv_strc.lattice.get_cartesian_coords(start_pos)
    print("start_pos", start_pos)
    
    # Make the function
    def output(disp, vel_mag, variance = np.array([0, 0, 0])):
        pos = start_pos + disp * traj_dir
        x = featurizers.featurize(pos + variance, vel_mag * traj_dir)
        return model.predict(np.array([x]), verbose = 0)[0].item()
    return output

def _make_ode_function(cell, model, featurizers, start_point, start_traj, proj_mass = 1837):
    """Make the function used to run the ODE

    Args:
        start_point ([float]*3): Starting point of the run
        start_traj ([float]*3): Starting direction
    """

    # Make the force calculator
    force_calc = create_force_calculator_given_displacement(cell, model, featurizers, start_point, start_traj)
    
    def output(t, y):
        # Get the velocity and displacement
        v, x = y

        # Compute the force
        ts = time()
        f = force_calc(x, v)
        return [-f / proj_mass, v]
    return output


def _in_house_stepper(cell, model, featurizers, start_point, start_velocity, proj_mass, stepper, stop_velocity_mag, dt, max_time, output, status):
    start_time = time()

    force_calc = create_force_calculator_given_displacement(cell, model, featurizers, start_point, start_velocity)
    v = np.linalg.norm(start_velocity)
    x = 0
    acc = 0

    t = 0

    i = 0
    states = [(0, v, 0, time() - start_time)]

    while v > stop_velocity_mag and t < max_time:
        wts = time()

        x, v, acc = stepper(x, v, acc, force_calc, proj_mass, dt)

        i += 1
        t += dt
        if output is not None and i % output == 0:
            states.append([t, v, x, time() - start_time])
            if status:
                _viewable(i, t, v, x, dt, time() - wts)
            
    stop_dist = x

    # Return the results
    if output is not None:
        return stop_dist, _output(i, t, v, x, time() - start_time, states)
    return stop_dist

def _scipy_stepper(cell, model, featurizers, start_point, start_velocity, proj_mass, stepper, stop_velocity_mag, max_time, rtol, atol, max_step, output, status):
    """Compute the stopping distance of a projectile
    
    Args:
        start_point ([float]*3): Starting point of the run. In fractional coordinates of conventional cell
        start_velocity ([float]*3): Starting velocity
        stop_velocity_mag (float): Velocity at which to stop the calculation
        max_time (float): Time at which to stop the solver (assuming an error)
        output (int): Number of timesteps between outputting status information
        status (bool): Whether to print status information to screen
    Returns:
        - (float) Stopping distance
        - (pd.DataFrame) Velocity as a function of position and time
    """
    start_time = time()

    # Make the force calculator
    fun = _make_ode_function(cell, model, featurizers, start_point, start_velocity, proj_mass)
    
    # Compute the initial velocity
    v_init = np.linalg.norm(start_velocity)
    
    # Create the ODE solvers
    stepper = stepper(fun, 0, [v_init, 0], max_time, rtol = rtol, atol = atol, max_step = max_step)
    
    # Iterate until velocity slows down enough
    i = 0
    states = [(0, v_init, 0, time() - start_time)]

    while stepper.y[0] > stop_velocity_mag and stepper.t < max_time:
        wts = time()

        stepper.step()
        i += 1
        if (output is not None) and (i % output == 0):
            states.append([stepper.t, *stepper.y, time() - start_time])
        if (status):
            stepper.t_old = 0 if i == 1 else stepper.t_old  
            _viewable(i, stepper.t, stepper.y[0], stepper.y[1], stepper.t - stepper.t_old, time() - wts)

    # Determine the point at which the velocity crosses the threshold
    #   ODE solvers give you an interpolator over the last timestep
    interp = stepper.dense_output()
    res = minimize_scalar(lambda x: np.abs(interp(x)[0] - stop_velocity_mag), bounds=(stepper.t_old, stepper.t))
    stop_dist = interp(res.x)[1]
            
    # Return the results
    if output is not None:
        return stop_dist, _output(i, stepper.t, stepper.y[0], stepper.y[1], time() - start_time, states)
    return stop_dist

def _output(step, t, v, x, ctime, states):
    states = pd.DataFrame(dict(zip(['time', 'velocity', 'displacement', 'sim_time'], np.transpose(states))))
    return states

def _viewable(i, t, v, x, dt, wtime):
    if (i%10 == 0):
        print(f'Step: {i} - Time: {t:0.8f} - Velocity: {v} - Position: {x} - time step: {dt} - wall time: {wtime:0.4f} sec ', end = "\r")

def compute_stopping_distance(cell, model, featurizers, start_point, start_velocity, *, proj_mass = 1837, max_step = 50, rtol = 1e-5, atol = 1e-7, stepper = 'rk45', stop_velocity_mag = 0.4, max_time = 3e5, output = None, status = True):
    if (np.linalg.norm(start_velocity) < stop_velocity_mag):
        print("starting velocity is smaller than stopping velocity")
        return 
    stepper = stepper.lower()
    scipy_steppers = {'rk45': RK45, 'lsoda': LSODA, 'dop853': DOP853, 'radau': Radau, 'bdf': BDF}
    in_house_steppers = {'rk4': _rk4, 'velocity_verlet': _velocity_verlet}
    if (stepper in scipy_steppers):
        return _scipy_stepper(cell, model, featurizers, start_point, start_velocity, proj_mass, scipy_steppers[stepper], stop_velocity_mag, max_time, rtol, atol, max_step, output, status)
    elif (stepper in in_house_steppers):
        return  _in_house_stepper(cell, model, featurizers, start_point, start_velocity, proj_mass, in_house_steppers[stepper], stop_velocity_mag, max_time, output, status)
    else:
        raise NameError(f"{stepper} is not supported")
