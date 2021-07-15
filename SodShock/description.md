<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true},
jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
TeX: {
extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
equationNumbers: {
autoNumber: "AMS"
}
}
});
</script>


# Analytical solution for Sod shock tube

*This description shows the equations used in* `sod_shock_exact.py`*, a script calculating the analytical solution of the Sod shock tube problem after a time* t. *Equations labelled here bear the same tags as in that script.*

The Sod shock tube contains two different regions separated by a thin barrier at $x = x_0$.
Each region contains the same gas, but with different pressure and density.

The initial conditions for the left and right domains are

$$
\begin{align*}
p_l &= 1\,, \quad p_r = 0.1 \\ 
\rho_l &= 1\,, \quad \rho_r = 0.125 \\ 
u_l &= 0\,, \quad u_r = 0 \\
\end{align*}
$$

where $p$ is pressure, $\rho$ is mass density, and $u$ is the gas velocity in the direction along the tube. At $t = 0$, the barrier is removed, and the two initial regions are allowed to mix.

When the barrier is removed, the tube is split into five domains.

- Left boundary (l): *pressure, density, and velocity are constant*
- Expansion fan (e): *pressure and density decreases*
- Contact area (2): *region where the initial regions meet*
- Shock (1): *shock front creates a discontinuity with the right boundary*
- Right boundary (r): *pressure, density, and velocity are constant*

The system is described with the one-dimensional Euler system of PDEs

$$
\frac{\partial}{\partial t} \begin{bmatrix}
\rho\\
\rho u\\
e
\end{bmatrix}
+
\frac{\partial}{\partial x} \begin{bmatrix}
\rho u\\
\rho u^2 + p\\
(e + p)u
\end{bmatrix} = 0\,,
$$
where $e$ is the internal energy.

We use the equation of state to close the system of equations. The EOS is given by
$$
p = \rho (\gamma - 1) e
$$
with $\gamma = 1.4$. We define the constants
$$
\begin{equation*}
\Gamma = \frac{\gamma - 1}{\gamma + 1}\,, \quad \beta = \frac{\gamma - 1}{2\gamma}\,.
\end{equation*}
$$

Another equation needed before diving into the solution, is the equation for sound speed $c$. Without further ado,
$$
c = \sqrt{\gamma \frac{p}{\rho}}\,.
$$

First, the *shock* region is described by the Rankine–Hugoniot relations. This relation is formulated as

$$
u_1 - u_2 = 0\,.
$$
We can express the velocity in terms of pressure,
$$
(p' - p_r) \sqrt{\frac{1 - \Gamma}{\rho_r (p' + \Gamma p_r)}} - (p_l^{\beta} - p'^{\beta}) \sqrt{\frac{(1 - \Gamma^2) p_l^{1/\gamma}}{\Gamma^2 \rho_l}} = 0\,,
$$
and the $p'$ that satisties this relation is in fact the pressure in the *shock* region. Furthermore,

$$
\begin{align}
p_1 &= p'\,, \nonumber \\
\rho_1 &= \rho_r\frac{p_1 + \Gamma p_r}{p_r + \Gamma p_1}\,, \tag{1} \\
u_1 &= (p_1 - p_r)\sqrt{\frac{1 - \Gamma}{\rho_r(p_1 + \Gamma p_r)}} \nonumber\,.
\end{align}
$$

Then, the physical quantities of the *contact* region are found by the pressure and velocity being continuous, and mass density follows from the adiabatic gas law. This gives

$$
\begin{align}
u_2 &= u_1\,, \nonumber\\
p_2 &= p_1\,, \tag{2} \\
\rho_2 &= \rho_l \left( \frac{p_2}{p_l}\right)^{1/\gamma}\,. \nonumber
\end{align}
$$

Expressions for *shock* and *contact* regions are inspired by Wikipedia's article on the [Sod shock tube](https://en.wikipedia.org/wiki/Sod_shock_tube).

The *expansion fan* is solved according to Höfner's [lecture notes](https://www.astro.uu.se/~hoefner/astro/teach/ch10.pdf) (chapter 10). We calculate the region accordingly

$$
\begin{align}
u_e &= \frac{2}{\gamma + 1} \left(c_l + \frac{x - x_0}{t}\right) \,, \nonumber \\
p_e &= p_l \left(\frac{c_e}{c_l}\right)^{2 \gamma/(\gamma - 1)} \,, \tag{e} \\
\rho_e &= \gamma \frac{p_e}{c_e^2} \,. \nonumber
\end{align}
$$
The sound speed in the expansion fan is not constant, and found by
$$c_e = c_l - (\gamma - 1)\frac{u_e}{2}\,.$$

$$
%\begin{align}
%\tag{ex}
%\end{align}
$$

Lastly, the boundaries of the different regions are calculated. There are four boundaries that need to be found. The boundaries between *left* boundary and *expansion fan* (le), *expansion fan* and *contact* region (e2), and between *contact* and *shock* region (21) are found using the same expressions as Höfner's lecture notes. The boundary between the *shock* region and the right boundary (1r) is inspired by Ibackus' [github repository](https://github.com/ibackus/sod-shocktube).

$$
\begin{align}
x_{le} &= x_0 - c_l t\,, \tag{le}\\
x_{e2} &= x_0 + (u_2 - c_2) t\,,\tag{e2}\\
x_{21} &= x_0 + u_2 t\,,\tag{21}\\
x_{1r} &= x_0 + w t\,.\tag{1r}\\
\end{align}
$$

The factor $w$ in equation (1r) is given by
$$
w = c_r \sqrt{1 + \frac{\gamma + 1}{2\gamma}\left( \frac{p_2}{p_r} - 1\right)}\,.
$$
