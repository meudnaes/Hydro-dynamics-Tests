import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import fsolve, newton

class SedovBlastWave:

  def __init__(self, gamma, E, rho_0):
    self.gamma = gamma
    self.E = E
    self.rho_0 = rho_0
    self.beta = self.beta_func()

  def Z(self, U):
    r"""
    $Z(U(\xi))$
    """
    # gamma factor
    gamma = self.gamma
    gm1 = gamma*(gamma-1)/2

    return gm1 * U**2 * (1-U)/(gamma*U-1)

  def xi5(self, U):
    r"""
    fifth power of $\xi$, the dimensionless similarity variable

    $\xi = \beta \frac{E}{\rho_0}^{-1/5} r t^{-2/5}$
    """
    # gamma factor
    gamma = self.gamma
    gm1 = 1/2*(gamma+1)*U

    # dividends
    p1 = (gamma+1)/(7-gamma)*(5-(3*gamma-1)*U)
    p2 = (gamma+1)/(gamma-1)*(gamma*U-1)

    # exponents
    nu1 = -(13*gamma**2-7*gamma+12)/((2*gamma+1)*(3*gamma-1))
    nu2 = 5*(gamma-1)/(2*gamma+1)

    return gm1**(-2) * p1**nu1 * p2**nu2

  def U_implicit(self, U, xi):
    r"""
    Used to find $U$ implicitly
    """
    return self.xi5(U) - xi**(5)

  def d_lnxi_dU(self, U):
    r"""
    $\frac{d \ln(\xi)}{dU}$
    """
    # gamma factor
    gamma = self.gamma

    p1 = 5-(3*gamma-1)*U

    frac1 = gamma/p1
    frac2 = 2*(1-U)/(U*p1)
    frac3 = gamma*(1-U)/((gamma*U-1)*p1)

    return frac1 - frac2 + frac3

  def G(self, U):
    r"""
    Dimensionless variable to scale mass density, function of $\xi$

    $\rho = \rho_0 G(\xi)$
    """
    # gamma factor
    gamma = self.gamma

    p1 = (gamma+1)/(gamma-1)
    p2 = p1*(gamma*U-1)
    p3 = (gamma+1)/(7-gamma)*(5-(3*gamma-1)*U)
    p4 = p1*(1-U)

    nu3 = 3/(2*gamma+1)
    nu4 = (13*gamma**2-7*gamma+12)/((2*gamma+1)*(3*gamma-1)*(2-gamma))
    nu5 = -2/(2-gamma)

    return p1 * p2**nu3 * p3**nu4 * p4**nu5

  def beta_func(self):
    """
    the integrand of beta
    """
    # gamma-factor
    gamma = self.gamma
    gm1 = 1/(gamma*(gamma-1))

    def integrand(U):
      """
      The intrgrand in beta factor
      """
      return self.xi5(U) * self.G(U) * (1/2*U**2 + gm1*self.Z(U))*self.d_lnxi_dU(U)

    # solve integral
    I5 = quad(integrand, 1/gamma, 2/(gamma+1), points=[1/gamma])
    # mulitply with prefactor and take fifth root to get solution
    return (16*np.pi/25*I5[0])**(1/5)

  def xi(self, r, t):
    r"""
    Self-similar spatial free variable

    $\xi(r, t)$
    """
    return self.beta*(self.E/self.rho_0)**(-1/5)*r*t**(-2/5)

  def r(self, xi, t):
    r"""
    Radius

    $r(\xi, t)$
    """
    return self.beta * (self.E/self.rho_0)**(1/5)*xi*t**(2/5)

  def v(self, r, t, U):
    r"""
    Velocity

    $v(r, t, U(\xi))$
    """
    return 2/5*r/t*U

  def c_s2(self, r, t, U):
    r"""
    Sound speed

    $c_s(r, t, Z(\xi))$
    """
    return 4/25*self.Z(U)*r**2/t**2

  def rho(self, U, xi):
    r"""
    Mass density

    $\rho(G(\xi))$
    """
    return self.rho_0*self.G(U)

  def p(self, rho, c_s2):
    """
    Equation Of State (EOS) to solve pressure
    """
    return rho*c_s2/self.gamma

  def beta_test(self):
    """
    function for "sanity check" of the beta factor
    *** gamma = 1.4 --> beta = 0.968 ***
    *** gamma = 5/3 --> beta = 0.868 ***
    """
    # store old gamma
    gamma_original = self.gamma

    # gamma = 5/3, case for air
    gamma_air = 5/3
    self.gamma = gamma_air
    beta_air = self.beta_func()
    test_air = abs(beta_air - 0.868) > 5e-4

    # gamma = 1.4
    gamma_14 = 1.4
    self.gamma = gamma_14
    beta_14 = self.beta_func()
    test_14 = abs(beta_14 - 0.968) > 5e-4

    if test_air or test_14:
      print('For gamma=5/3: got beta={:.3f}, expected {:.3f}'.format(beta_air,
                                                                     0.868))
      print('For gamma=1.4: got beta={:.3f}, expected {:.3f}'.format(beta_14,
                                                                     0.968))
    else:
      print('All tests passed!')

    # reset gamma
    self.gamma = gamma_original
    return

  def __call__(self, t, N=100):
    """
    calculates system at time t and returns position, density, pressure,
    and velocity
    """
    xi_arr = np.linspace(0, 1, N)
    # solve U
    U = np.zeros(N)
    U0 = 1/self.gamma
    for i, xi in enumerate(xi_arr):
      U[i] = fsolve(self.U_implicit, U0, args=(xi))
      U0 = U[i]

    # solve rest of system
    r_arr = self.r(xi_arr, t)
    rho = self.rho(U, xi)
    v = self.v(r_arr, t, U)
    # sound speed squared
    c_s2 = self.c_s2(r_arr, t, U)
    p = self.p(rho, c_s2)

    return r_arr, rho, p, v

def main(N = 10):
  # Set up system
  gamma = 5/3
  E = 1
  rho_0 = 1

  # Making instance of system
  blast = SedovBlastWave(gamma, E, rho_0)

  # defining time t
  t = 1

  # quantities
  r_arr, rho, p, v = blast(t)

  r_arr /= r_arr[-1]
  v /= v[-1]
  rho /= rho[-1]
  p /= p[-1]

  plt.figure(1)
  plt.title("Normalised velocity")
  plt.ylabel("velocity")
  plt.xlabel("r")
  plt.plot(r_arr, v, color='g', label='t = 0.2')
  plt.legend()
  plt.savefig("figs/sedov_blast_v.png")

  plt.figure(2)
  plt.title("Normalised density")
  plt.ylabel("density")
  plt.xlabel("r")
  plt.plot(r_arr, rho, color='b', label='t = 0.2')
  plt.legend()
  plt.savefig("figs/sedov_blast_rho.png")

  plt.figure(3)
  plt.title("Normalised pressure")
  plt.ylabel("pressure")
  plt.xlabel("r")
  plt.plot(r_arr, p, color='r', label='t = 0.2')
  plt.legend()
  plt.savefig("figs/sedov_blast_p.png")


  # sanity check
  blast.beta_test()

  return

if __name__=="__main__":
  main(N=500)
