import numpy as np
import ibmos as ib

# List of default parameters
default_α = 30*np.pi/180

#def solver1():
#    s1 = ib.stretching(192, 0.04, 0.25, int(0.65/0.04), 16, 16, 0.04)
#    s2 = ib.stretching(96, 0.04, 0.25, int(0.65/0.04), 16, 16, 0.04)
#    x = np.concatenate([-s2[::-1], s1[1:]])
#
#    s = ib.stretching(22*3//2 + 1, 0.04, 0.25, int(0.65/0.04), 16, 16, 0.04)
#    y = np.concatenate([-s[::-1], s[1:]])
#    assert (len(y)-1)%3==0, "len(y)-1 must be divisible by 3"
#
#    solver = ib.Solver(x, y, periodic=True, iRe=1/40.0, Co=0.5)
#    solver.set_solids(ib.shapes.cylinder("cylinder", 0, 0, 0.5, solver.dxmin))
#    
#    return solver

def solver():
    s1 = ib.stretching(192, 0.04, 0.25, int(0.65/0.04), 16, 16, 0.04)
    s2 = ib.stretching(96, 0.04, 0.25, int(0.65/0.04), 16, 16, 0.04)
    x = np.concatenate([-s2[::-1], s1[1:]])

    s = ib.stretching(22*3//2 + 1, 0.04, 0.25, int(0.65/0.04), 16, 16, 0.04)
    y0 = np.concatenate([-s[::-1], s[1:]])
    Ly = y0[-1] - y0[0]
    
    y = np.r_[y0[:-1] - Ly, y0[:-1], y0 + Ly]
    assert (len(y)-1)%3==0, "len(y)-1 must be divisible by 3"

    solver = ib.Solver(x, y, periodic=True, iRe=1/40.0, Co=0.5)
    solver.set_solids(ib.shapes.cylinder("cyl[-]", 0, -Ly, 0.5, solver.dxmin),
                      ib.shapes.cylinder("cyl[0]", 0, 0, 0.5, solver.dxmin),
                      ib.shapes.cylinder("cyl[+]", 0, Ly, 0.5, solver.dxmin),)
    
    return solver

def boundary_conditions(solver, α = default_α):
    uBC, vBC = solver.zero_boundary_conditions()

    for k in range(2):
        uBC[k][:] = np.cos(α)
        vBC[k][:] = np.sin(α)

    sBC = [(np.zeros(solidk.l), np.zeros(solidk.l)) for solidk in solver.solids]
    
    return uBC, vBC, sBC
    
def initial_condition(solver,  α = default_α):
    x = solver.zero(); 
    x[:solver.fluid.u.size] = np.cos(α)
    x[solver.fluid.u.size:solver.fluid.u.size + solver.fluid.v.size] = np.sin(α)
    
    return x

def base_flow(solver_ = None, α = default_α, x0 = None, verbose = False):
    if not solver_:
        solver_ = solver()
        
    if not x0:
        x0 = initial_condition(solver_, α)
        
    bc = boundary_conditions(solver_, α)
    
    x = solver_.steady_state(x0, *bc, verbose = verbose, outflowEast = False)[0]
    
    return x, solver_
    
    
    