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

The Sod shock tube contains two different regions separated by a thin barrier.
Each region contains the same gas, but with different pressure and density.

The initial conditions for the left and right domains are

$$p_l = 1\,, \quad p_l = 0.1 $$

$$\rho_l = 1\,, \quad \rho_r = 0.125 $$

$$u_r = 0\,, \quad u_r = 0 $$

where $p$ is pressure, $\rho$ is mass density, and $u$ is the gas velocity in the direction along the tube. At $t = 0$, the barrier is removed, and the two initial regions are allowed to mix.

When the barrier is removed, the tube is split into five domains.

1. Left boundary: *pressure, density, and velocity are constant*
2. Expansion fan: *pressure and density decreases*
3. Contact area: *region where the initial regions meet*
4. Shock: *shock front creates a discontinuity with the right boundary*
5. Right boundary: *pressure, density, and velocity are constant*

The system is described with the one-dimensional Euler system of PDEs

$$
\begin{equation}
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
\end{equation}
$$
where $e$ is the internal energy.

We use the equation of state to close the system of equations. The EOS is given by
$$
\begin{equation}
p = \rho (\gamma - 1) e
\end{equation}
$$
with $\gamma = 1.4$.

First, the *shock* region is described by the Rankine–Hugoniot relations. Then, the physical quantities of the *contact* region are found by the pressure and velocity being continuous, and mass density follows from the adiabatic gas law. Expressions for *shock* and *contact* regions are inspired by Wikipedia's article on the [Sod shock tube](https://en.wikipedia.org/wiki/Sod_shock_tube).
The *expansion fan* is solved according to Höfner's [lecture notes](https://www.astro.uu.se/~hoefner/astro/teach/ch10.pdf) (chapter 10).

Lastly, the boundaries of the different regions are calculated. There are four boundaries that need to be found. The boundaries between *left* boundary and *expansion fan*, *expansion fan* and *contact* region, and between *contact* and *shock* region are found using the same expressions as Höfner's lecture notes. The boundary between the *shock* region and the right boundary is in inspired by Ibackus' [github repository](https://github.com/ibackus/sod-shocktube).
