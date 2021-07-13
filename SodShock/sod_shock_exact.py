import sod
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve


class SodShockTube:
  """
  Exact solution to the Sod shock problem after time t
  Inspired by:
    - https://www.astro.uu.se/~hoefner/astro/teach/ch10.pdf
        --> calculation of expansion fan and determination of regions
    - https://en.wikipedia.org/wiki/Sod_shock_tube
        --> calculation of shock and contact regions
    - https://github.com/ibackus/sod-shocktube
        --> testing of the code
  """

  def __init__(self, x_0 = 0.5, gamma=1.4,
               rho_l = 1.0, rho_r = 0.125,
               p_l = 1.0, p_r = 0.1,
               u_l = 0.0, u_r = 0.0):

    """
    ------------domain is split in 5 parts------------
    |   l  |       e       |    2    |   1   |   r   |
    | left | expansion fan | contact | shock | right |

    x_0: float
      location of barrier between the two sections
    gamma: float
      heat capacity ratio
    rho: float
      mass density
      rho_l --> left regime
      rho_r --> right regime
    rho: float
      mass density
      rho_l --> left regime
      rho_r --> right regime
    P: float
      pressure
      P_l --> left regime
      P_r --> right regime
    u: float
      velocity
      u_l --> left regime
      u_r --> right regime
    """

    # x0
    self.x_0    = x_0

    # heat capacity constant (gas constant)
    self.gamma  = gamma

    # define initial conditions, these are also boundaries for all t
    # left side
    self.rho_l  = rho_l
    self.p_l    = p_l
    self.u_l    = u_l

    # right side
    self.rho_r  = rho_r
    self.p_r    = p_r
    self.u_r    = u_r

    # sound speed
    self.c_l    = np.sqrt(self.gamma*p_l/rho_l)
    self.c_r    = np.sqrt(self.gamma*p_r/rho_r)

    # constants
    self.Lambda = (gamma - 1)/(gamma + 1)
    self.beta = (gamma - 1)/(2*gamma)

  def shock_tube_function(self, p_mrk):
    """
    implicit relation, p_mrk = p_1 when u_1 - u_2 = 0
    """
    u_1 = (p_mrk - self.p_r)*np.sqrt((1 - self.Lambda)/(self.rho_r*(p_mrk
                                    + self.Lambda*self.p_r)))
    u_2 = (self.p_l**self.beta - p_mrk**self.beta)*np.sqrt((1 - self.Lambda**2)
              *self.p_l**(1/self.gamma)/(self.Lambda**2*self.rho_l))
    return u_1 - u_2


  def iterate(self, func, guess):
    """
    solve implicit equation iteratively
    """
    P = fsolve(func, guess)
    return P[0]

  def shock(self, p_mrk):
    """
    shock region (1)
    """

    # thermodynamical quantities
    p_1 = p_mrk
    rho_1 = self.rho_r*(p_1 + self.Lambda*self.p_r)/(self.p_r + self.Lambda*p_1)
    u_1 = (p_mrk - self.p_r)*np.sqrt((1 - self.Lambda)/(self.rho_r*(p_mrk
                                    + self.Lambda*self.p_r)))

    return p_1, rho_1, u_1

  def contact(self, p_1, u_1):
    """
    contact region (2)
    """
    u_2 = u_1
    p_2 = p_1
    rho_2 = self.rho_l*(p_2/self.p_l)**(1/self.gamma)

    return p_2, rho_2, u_2

  def expansion_fan(self, x, t):
    """
    expansion fan region (e)
    """
    u_e = 2/(self.gamma + 1)*(self.c_l + (x - self.x_0)/t)

    # sound speed, varies over region!
    c_e = self.c_l - (self.gamma - 1)*u_e/2

    p_e = self.p_l*(c_e/self.c_l)**(2*self.gamma/(self.gamma - 1))

    # find rho_e from sound speed!
    rho_e = self.gamma*p_e/c_e**2

    return p_e, rho_e, u_e

  def x_1(self, t):
    """
    interface between left (l) and expansion fan (e)
    """
    x_1 = self.x_0 - self.c_l*t
    return x_1

  def x_2(self, t, u_2, c_2):
    """
    interface between expansion fan (e) and contact (2)
    """
    x_2 = self.x_0 + (u_2 - c_2)*t
    return x_2

  def x_3(self, t, u_2):
    """
    interface between contact (2) and shock (1)
    """
    x_3 = self.x_0 + u_2*t
    return x_3

  def x_4(self, t, M_s):
    """
    interface between shock (1) and right (r)
    """
    x_4 = self.x_0 + M_s*t
    return x_4

  def make_array(self, q_l, q_e, q_2, q_1, q_r, N):
    """
    merge 5 regions into one array
    """
    q_array = np.concatenate([q_l*np.ones(N),
                              q_e,
                              q_2*np.ones(N),
                              q_1*np.ones(N),
                              q_r*np.ones(N)])
    return q_array


  def __call__(self, t, N=100):
    """
    calculate exact solution at time t
    """

    # Left and right are equal to initial conditions!
    p_l, rho_l, u_l = self.p_l, self.rho_l, self.u_l
    p_r, rho_r, u_r = self.p_r, self.rho_r, self.u_r

    p_guess = (p_l + p_r)/2 #self.u_l/self.c_l
    p_mrk = self.iterate(self.shock_tube_function, p_guess)

    p_1, rho_1, u_1 = self.shock(p_mrk)

    p_2, rho_2, u_2 = self.contact(p_1, u_1)

    # find boundaries before calculating expansion fan
    x_1 = self.x_1(t)

    c_2 = np.sqrt(self.gamma*p_2/rho_2)
    x_2 = self.x_2(t, u_2, c_2)

    x_3 = self.x_3(t, u_2)

    z = p_2/p_r - 1
    fact = np.sqrt(1 + (self.gamma + 1)/(2*self.gamma)*z)
    x_4 = self.x_4(t, self.c_r*fact)

    x_e = np.linspace(x_1, x_2, N)

    p_e, rho_e, u_e = self.expansion_fan(x_e, t)

    p_array = self.make_array(p_l, p_e, p_2, p_1, p_r, N)
    rho_array = self.make_array(rho_l, rho_e, rho_2, rho_1, rho_r, N)
    u_array = self.make_array(u_l, u_e, u_2, u_1, u_r, N)

    x_array = np.concatenate([np.linspace(0, x_1, N),
                              np.linspace(x_1, x_2, N),
                              np.linspace(x_2, x_3, N),
                              np.linspace(x_3, x_4, N),
                              np.linspace(x_4, 1, N)])

    return x_array, p_array, rho_array, u_array

def main():
  """
  Plot quantities
  """
  sod_shock = SodShockTube()
  x, p, rho, u = sod_shock(0.2, N=25)

  gamma = 1.4
  npts = 500

  # left_state and right_state set p, rho and u
  # geometry sets left boundary on 0., right boundary on 1 and initial
  # position of the shock xi on 0.5
  # t is the time evolution for which positions and states in tube should be calculated
  # gamma denotes specific heat
  # note that gamma and npts are default parameters (1.4 and 500) in solve function
  positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.),
                                         geometry=(0., 1., 0.5), t=0.2, gamma=gamma, npts=npts)

  # Finally, let's plot solutions
  p_mrk = values['p']
  rho_mrk = values['rho']
  u_mrk = values['u']

  plt.figure(1)
  plt.plot(x, rho, color='b', linewidth=3)
  plt.plot(values['x'], rho_mrk, linewidth=1.5, color='k', ls='--')
  plt.xlabel(r"$x$")
  plt.ylabel(r"Density")
  plt.savefig("figs/sod_shock_rho.png")
  plt.close()

  plt.figure(2)
  plt.plot(x, p, color='r', linewidth=3)
  plt.plot(values['x'], p_mrk, linewidth=1.5, color='k', ls='--')
  plt.xlabel(r"$x$")
  plt.ylabel(r"Pressure")
  plt.savefig("figs/sod_shock_pressure.png")
  plt.close()

  plt.figure(3)
  plt.plot(x, u, color='b', linewidth=3)
  plt.plot(values['x'], u_mrk, linewidth=1.5, color='k', ls='--')
  plt.xlabel(r"$x$")
  plt.ylabel(r"$U_x$")
  plt.savefig("figs/sod_shock_u_x.png")
  plt.close()

  return

def write_table():
  """
  write Sod data to file
  """
  # first load the data
  sod_shock = SodShockTube()
  N = 5
  x, p, rho, u = sod_shock(0.2, N=N)
  # check
  #plt.scatter(x, rho)
  #plt.show()
  with open("sod_data.txt", 'w', encoding = 'utf-8') as f:
    f.write("# x density pressure u_x\n")
    for i in range(5*N):
      f.write("{x:.4f} {rho:.4f} {p:.4f} {u:.4f}\n".format(x=x[i], rho=rho[i], p=p[i], u=u[i]))

  return

if __name__ == "__main__":
  main()
  write_table()
