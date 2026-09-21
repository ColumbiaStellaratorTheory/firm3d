.. _stellgap-design:

Incompressible continuum solver for Boozer fields
=================================================

For a given ``InterpolatedBoozerField``, FIRM3D can calculate incompressible
shear Alfvén continuum on selected flux surfaces. The algorithm follows
the no-sound-wave computational approach of the STELLGAP code described in [Spong2003]_.

Continuum equation
------------------

The continuum is computed in normalized Boozer coordinates :math:`(s,\theta,\zeta)`, with
:math:`s=\psi/\psi_0` and signed boundary toroidal flux :math:`2\pi\psi_0`.

On each flux surface, the continuum equation [Paul2025]_ is

.. math::

   B\nabla_\parallel\left(
       \frac{|\nabla\psi|^2}{B}\nabla_\parallel\Phi\right)
   + \frac{\omega^2|\nabla\psi|^2}{v_A^2}\Phi = 0,

where :math:`\nabla_\parallel=\mathbf b\cdot\nabla`,
:math:`\mathbf b=\mathbf B/B`, and
:math:`v_A^2=B^2/[\mu_0\rho(s)]`.
The mass density :math:`\rho(s)` is assumed constant on each flux surface.

In Boozer coordinates, the parallel gradient and the angular
field-line derivative are 

.. math::

   \nabla_\parallel=\frac{B}{G+\iota I}\partial_\parallel,
   \qquad
   \partial_\parallel=\partial_\zeta+\iota(s)\partial_\theta.

Here :math:`G(s)` and :math:`I(s)` are the toroidal and poloidal covariant
magnetic-field components, respectively. Both are flux functions with units
of :math:`\mathrm{T\,m}`. Substituting these relations and cancelling
:math:`\psi_0^2` gives the equation in normalized flux coordinates:

.. math::
   :label: stellgap-dimensional

   \partial_\parallel\left(|\nabla s|^2\partial_\parallel\Phi\right)
   + \omega^2\mu_0\rho(s)(G+\iota I)^2
       \frac{|\nabla s|^2}{B^4}\Phi = 0.

In SI units, :math:`\Phi` is the perturbed electrostatic potential in
:math:`\mathrm{V}`, :math:`\omega` is the angular frequency in
:math:`\mathrm{rad\,s^{-1}}`, :math:`B=|\mathbf B|` is the magnetic field
strength in :math:`\mathrm{T}`, :math:`\rho(s)` is the mass density in
:math:`\mathrm{kg\,m^{-3}}`, and :math:`\mu_0` is the vacuum permeability.

To normalize the field strength and frequency, use the volume-averaged
field strength as a fixed reference:

.. math::

   B_{\rm vol}=\frac{\int_{\mathcal V}B\,dV}{\int_{\mathcal V}dV},

where :math:`\mathcal V` is the plasma volume. Define the dimensionless
frequency and the positive reference angular frequency by

.. math::

   \widehat\omega=\frac{\omega}{\omega_A(s)},\qquad
   \omega_A(s)=\frac{B_{\rm vol}^2}
       {|G(s)+\iota(s)I(s)|\sqrt{\mu_0\rho(s)}}.

The continuum equation then becomes

.. math::
   :label: stellgap-normalized

   \partial_\parallel\left(|\nabla s|^2\partial_\parallel\Phi\right)
   + \widehat\omega^2
       \frac{|\nabla s|^2}{(B/B_{\rm vol})^4}\Phi = 0.

This eigenproblem depends only on the magnetic equilibrium, so
:math:`\widehat\omega` can be computed without specifying density.
Although :math:`B_{\rm vol}` is fixed, :math:`\omega_A(s)` is a local
frequency scale through :math:`G+\iota I` and :math:`\rho(s)`.
If density is supplied, the physical frequency is

.. math::

   f(s)=\frac{\widehat\omega(s)\,\omega_A(s)}{2\pi},

with :math:`f` in :math:`\mathrm{Hz}` when :math:`\omega_A` is in
:math:`\mathrm{rad\,s^{-1}}`.

.. _stellgap-preliminary-considerations:

Preliminary considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``get_covariant_metric()`` method of the ``BoozerMagneticField`` class, covariant
Boozer metric components :math:`g_{ij}` are calculated from covariant basis vectors,

.. math::

   g_{ij}=\frac{\partial\mathbf{R}}{\partial x_i}\cdot\frac{\partial\mathbf{R}}{\partial x_j},

where , :math:`x_i` and :math:`x_j` are Boozer coordinates,
:math:`x_i,\;x_j\in(s,\theta,\zeta)`. When doing so, cylindrical coordinates
:math:`R,Z,\phi` are treated as given functions of Boozer coordinates, so the components
are given by

.. math::

   g_{ij} = \frac{\partial R}{\partial x_i}\frac{\partial R}{\partial x_j} +
   \frac{\partial Z}{\partial x_i}\frac{\partial Z}{\partial x_j} +
   R^2 \frac{\partial \phi}{\partial x_i}\frac{\partial \phi}{\partial x_j}.

Consistent contravariant metric is then computed as inverse of :math:`g_{ij}`,

.. math::

   g^{ij}=(g_{ij})^{-1}.

While this approach is valid, care should be taken when using :math:`g^{ij}` in
expressions involving covariant filed components :math:`I` and :math:`G`, like in
continuum equation above. Notably, the issue can arise in Booz_xform of VMEC equilibrium
since, although :math:`\mathbf{R}(s,\theta,\zeta)`, :math:`\psi_0` and :math:`\iota`
**are provided simultaneously** with :math:`B`, :math:`I`, and :math:`G`, **both these
inputs fully determine magnetic field,**

.. math::
   :nowrap:

   \[
   \begin{aligned}
   \mathbf{B}&=\frac{\psi_0}{\frac{\partial \mathbf{R}}{\partial s}\cdot\left(\frac{\partial \mathbf{R}}{\partial \theta}\times\frac{\partial \mathbf{R}}{\partial \zeta}\right)}\left(\frac{\partial \mathbf{R}}{\partial \zeta}+\iota\frac{\partial\mathbf{R}}{\partial\theta}\right)\\
   &=G\nabla\zeta+I\nabla\theta+K\nabla s.
   \end{aligned}\tag{1}
   \]

In :math:`{\rm VMEC}\to{\rm Booz\_xform}` pipeline, the :math:`\mathbf{B}` field
computed in covariant basis from :math:`\mathbf{R}`, :math:`\psi_0` and :math:`\iota`
**is only approximately** equal to the :math:`\mathbf{B}` field computed in
contravariant basis from :math:`G` and :math:`I`. One consequence of that is that, for
example, the Boozer Jacobian relation

.. math::

   \frac{\partial \mathbf{R}}{\partial s}\cdot\left(\frac{\partial \mathbf{R}}{\partial \theta}\times\frac{\partial \mathbf{R}}{\partial \zeta}\right)=\frac{G+\iota I}{B^2}

holds only approximately. Next section shows the consequences of vialating this identiy
for continuum equation.

.. _stellgap-geometric-continuum:

Continuum equation from :math:`\mathbf{R}(s,\theta,\zeta)` and :math:`\iota`
----------------------------------------------------------------------------

Suppose the covariant matrix :math:`g_{ij}`,  calculated from
:math:`\mathbf{R}(s,\theta,\zeta)` as described above, we are given rotational transform
:math:`\iota(s)` and :math:`\psi_0`. This information fully specifies the magnetic field
data needed in continuum equation, as in the rest of this section. We avoid clutter with
notation :math:`\mathbf r_i=\partial\mathbf R/\partial x^i`, and introduce

.. math::

   J_g=\mathbf r_s\cdot(\mathbf r_\theta\times\mathbf r_\zeta),\qquad
   \Delta=\det[g_{ij}]=J_g^2>0.

Magnetic field :math:`\mathbf{B}_g` inferred from the geometric description of flux
surfaces :math:`\mathbf{R}(s,\theta,\zeta)` is

.. math::

   \mathbf B_g=\frac{\psi_0}{J_g}(\mathbf r_\zeta+\iota\mathbf r_\theta).

Expressed in terms of the (squared) field-line tangent length,

.. math::

   S=g_{\zeta\zeta}+2\iota g_{\theta\zeta}+\iota^2g_{\theta\theta},

the amplitude of magnetic field :math:`B_g` is given by

.. math::

   B_{\rm g}^2=\frac{\psi_0^2 S}{\Delta}.

With derivative along the field line :math:`\partial_\|=\partial_\zeta +
\iota(s)\partial_\theta,` gradient :math:`\nabla_{||}^g` along the :math:`\mathbf{B}_g`
field is

.. math::

   \nabla_{\|}^g=\frac{\sigma}{\sqrt{S}}\partial_\|,

where :math:`\sigma=\operatorname{sgn}(\psi_0/J_g)` tracks the orientation.

Lastly, from vector identity

.. math::

   |\mathbf a\times\mathbf b|^2
   =(\mathbf a\cdot\mathbf a)(\mathbf b\cdot\mathbf b)
   -(\mathbf a\cdot\mathbf b)^2,

we can express :math:`|\mathbf r_\theta\times\mathbf r_\zeta|^2=H` as

.. math::

   H=g_{\theta\theta}g_{\zeta\zeta}-g_{\theta\zeta}^2,

so

.. math::

   |\nabla s|^2=g^{ss}=\frac{H}{\Delta}.

We can now substitute these expressions into the continuum equation,

.. math::

   B_g\nabla_\parallel^g\left(\frac{g^{ss}}{B_g}\nabla_\parallel^g\Phi\right)
   +\omega^2\mu_0\rho(s)\frac{g^{ss}}{B_g^2}\Phi=0.

Using notation above,

.. math::

   B_g \nabla_\|^g=\frac{|\psi_0|\sqrt{S}}{|J_g|}\frac{\operatorname{sgn}(\psi_0)}{\operatorname{sgn}(J_g)\sqrt{S}}\partial_\|=\frac{\psi_0}{J_g}\partial_\|,

and

.. math::

   \frac{g^{ss}}{B_g}\nabla_\|=\frac{H}{\Delta}\frac{J_g}{\psi_0S}\partial_\|,

so the continuum equation becomes

.. math::

   \frac{1}{J_g}\partial_\|\left(\frac{HJ_g}{\Delta S}\partial_\|\Phi\right)+\omega^2\mu_0\rho\frac{H}{\psi_0^2S}\Phi=0.

Observe that, since :math:`J_g` is continuous and nonzero, it can not change sign on its
surface, so :math:`\partial_\| J_{\rm g}=\operatorname{sgn}(J_g)\partial_\||J_g|`, and
continuum equation can be simplified into

.. math::

   \partial_\|(\mathcal A \partial_\|\Phi)-\omega^2\mathcal{W}\Phi=0,

where

.. math::

   \mathcal{A}=\frac{H}{\sqrt{\Delta} S}\text{ and }\mathcal{W}=\frac{\mu_0\rho\Delta}{\psi_0^2}\mathcal{A}.

Notably, although all metric :math:`g_{ij}=\mathbf{r}_i\cdot\mathbf{r}_j` components are
needed for evaluating :math:`H` and

.. math::

   \Delta=g_{ss}H-g_{s\theta}^2g_{\zeta\zeta}
   +2g_{s\theta}g_{s\zeta}g_{\theta\zeta}
   -g_{s\zeta}^2g_{\theta\theta},

only :math:`\Delta` and :math:`H/S` values are needed to evaluate the continuum
equation.

We seek approximate solution to the continuum equation, expressed in a basis of
:math:`N` known linearly independent periodic functions,

.. math::

   \Phi_N(\theta,\zeta)=\sum_{k=1}^N c_k f_k(\theta,\zeta),

where :math:`c_j` are the constant amplitudes to be determined. We will use cosine basis

.. math::

   f_k=\cos(m_k\theta-n_k\zeta),

so

.. math::

   \partial_\| f_k=-(m_k\iota-n_k)\sin(m_k\theta-n_k \zeta).

This expansion will not, in general, satisfy continuum equation everywhere on the
surface. However, a set of expansion amplitudes :math:`c_i` exists that satisfies
Galerkin condition of zero residual against each basis function used as a test function:

.. math::

   \int f_k^*\left[\partial_\| (\mathcal{A}\partial_\|\Phi_N)-\omega^2\mathcal{W}\Phi_N\right]d\Omega=0,\quad k=1,..,N,

where :math:`\Omega` denotes integral over Boozer angles,:math:`d\Omega=d\theta d\zeta`.
Using each of the functions from the :math:`N` basis functions in :math:`\Phi_N`
expansions results in :math:`N` linearly independent equations for :math:`N` unknown
amplitudes :math:`c_k`.

The first integral can be expressed as

.. math::

   \begin{aligned}
   \int f_k^*\partial_\| (\mathcal{A}\partial_\|\Phi_N)d\Omega&=\int f_k^*\partial_\zeta\left(\mathcal{A}\partial_\| \Phi_N\right)d\Omega\\
   &\quad +\iota\int f_k^*\partial_\theta (\mathcal{A}\partial_\|\Phi_N)d\Omega.
   \end{aligned}

Integrating the first with respect to :math:`\zeta` gives

.. math::

   \begin{aligned}
   \int f_k^*\partial_\zeta\left(\mathcal{A}\partial_\| \Phi_N\right)d\Omega&=\int_0^{2\pi}[f_k^*\mathcal{A}\partial_\|\Phi_N]_{\zeta=0}^{\zeta=2\pi}d\theta\\
   &\quad -\iota \int(\partial_\zeta f_k)^*\mathcal{A}\partial_\|\Phi_Nd\Omega,
   \end{aligned}

where the boundary terms vanishes since integrand matches on the periodic boundary.
Likewise, boundary term vanishes in

.. math::

   \begin{aligned}
   \int f_k^*\partial_\theta (\mathcal{A}\partial_\|\Phi_N)d\Omega&=\int [f_k^*\mathcal{A}\partial_\|\Phi_N]_{\theta=0}^{\theta=2\pi}d\theta\\
   &\quad -\int (\partial_\theta f_k)^* \mathcal{A}\partial_\|\Phi_Nd\Omega.
   \end{aligned}

Combining remaining terms gives

.. math::

   \int f_k^*\partial_\| (\mathcal{A}\partial_\|\Phi_N)d\Omega=-\int (\partial_\| f_i)^* \mathcal{A} \partial _\| \Phi_N d\Omega,

so the equation is

.. math::

   \int \mathcal{A}(\partial_\| f_k)^* \partial_\|\Phi_Nd\Omega=\omega^2\int\mathcal{W}f_k^*\Phi_Nd\Omega.

Substituting :math:`\Phi_N=\sum_jc_jf_j` gives

.. math::

   \sum_{j=1}^{N}c_j\int\mathcal A(\partial_\| f_k)^*\partial_\|f_j\,d\Omega
   =\omega^2\sum_{j=1}^{N}c_j\int\mathcal W f_k^*f_j\,d\Omega.

These expressions define :math:`N\times N` stiffness :math:`K_{ij}` and mass
:math:`M_{ij}` matrices

.. math::

   \begin{aligned}
   K_{ij} &= \int\mathcal A(\partial_\| f_k)^*\partial_\|f_j\,d\Omega,\\
   M_{ij} &=\int\mathcal W f_k^*f_j\,d\Omega,
   \end{aligned}

where row index :math:`i` labels the test function, and the column index :math:`j`
multiplies the basis function multiplying :math:`c_j`. The :math:`N` linear equations
are then

.. math::

   \sum_{j=1}^{N}\mathsf K_{ij}c_j
   =\omega^2\sum_{j=1}^{N}\mathsf M_{ij}c_j,
   \qquad i=1,\ldots,N.

Observe that, since :math:`\mathcal A` and :math:`\mathcal W` are real, conjugating a
matrix entry and interchanging its indices gives

.. math::

   \mathsf K_{ji}^*
   =\int_\Omega\mathcal A(Df_j)(Df_i)^*\,d\Omega
   =\mathsf K_{ij},\qquad
   \mathsf M_{ji}^*=\mathsf M_{ij}.

Thus both matrices are Hermitian: :math:`\mathsf K^\dagger=\mathsf K` and :math:`\mathsf
M^\dagger=\mathsf M`, where dagger denotes conjugate transpose. For a real basis, they
are real symmetric matrices. Note that, for any function :math:`\Phi_c=\sum_jc_jf_j`,
expanding the mass quadratic form gives

.. math::

   \begin{aligned}
   \mathbf c^\dagger\mathsf M\mathbf c
   &=\sum_{i,j}c_i^*c_j\int\mathcal W f_i^*f_j\,d\Omega\\
   &=\int\mathcal W
   \left(\sum_i c_if_i\right)^*
   \left(\sum_j c_jf_j\right)d\Omega\\
   &=\int\mathcal W|\Phi_c|^2\,d\Omega.
   \end{aligned}

Likewise,

.. math::

   \begin{aligned}
   \mathbf c^\dagger\mathsf K\mathbf c
   &=\int\mathcal A
   \left(\sum_i c_i\partial_\|f_i\right)^*
   \left(\sum_j c_j\partial_\|f_j\right)d\Omega\\
   &=\int\mathcal A|\partial_\|\Phi_c|^2\,d\Omega.
   \end{aligned}

Note that, since :math:`\mathcal{A}>0` and :math:`\mathcal{W}>0` are both positive, from
:math:`|\Phi_c|^2>0` follows that the mass matrix :math:`M_{ij}` is positive definite,
while, since :math:`|\partial_\|\Phi_c|^2>0`, the stiffness matrix :math:`K_{ij}` is
positive semi-definite. It follows that the eigenvalues :math:`\omega^2` of continuum
equation are non-negative,

.. math::

   \omega^2=\frac{\mathbf c^\dagger\mathsf K\mathbf c}
   {\mathbf c^\dagger\mathsf M\mathbf c}
   =\frac{\int\mathcal A|D\Phi_N|^2\,d\Omega}
   {\int\mathcal W|\Phi_N|^2\,d\Omega}\ge0,

as expected for ideal MHD model. To evaluate integrals over the flux surface
numerically, we sum integrand over the integration points
:math:`x_p=(\theta_p,\zeta_p)`, :math:`p=1,\ldots,N_q`, with positive weights
:math:`w_p`. Introducing :math:`N_q\times N` evaluation matrices

.. math::

   F_{pj}=f_j(x_p),\qquad(F_{\partial_\|})_{pj}={\partial_\|}f_j(x_p),

and letting :math:`\mathcal A_p=\mathcal A(x_p)`, :math:`\mathcal W_p=\mathcal W(x_p)`,
the quadrature approximation of the stiffness and mass matrices can be written as

.. math::

   \begin{aligned}
   \mathsf K^{(q)}&=F_{\partial_\|}^\dagger\operatorname{diag}(w_p\mathcal A_p)F_{\partial_\|},
   \\
   \mathsf M^{(q)}&=F^\dagger\operatorname{diag}(w_p\mathcal W_p)F.
   \end{aligned}

Because :math:`(F\mathbf c)_p=\Phi_c(x_p)` and :math:`(F_D\mathbf
c)_p=\partial_\|\Phi_c(x_p)`,

.. math::

   \begin{aligned}
   \mathbf c^\dagger\mathsf M^{(q)}\mathbf c
   &=\sum_p w_p\mathcal W_p|\Phi_c(x_p)|^2,\\
   \mathbf c^\dagger\mathsf K^{(q)}\mathbf c
   &=\sum_p w_p\mathcal A_p|D\Phi_c(x_p)|^2.
   \end{aligned}

This shows that finite sums preserve Herminian symmetry and nonnegativity of the
:math:`K` and :math:`M` matrices. The quadrature mass matrix :math:`M^{(q)}` is strictly
positive if :math:`F\mathbf c=0` implies :math:`\mathbf c=0`, meaning :math:`F` has full
column rank. This means that integration points must distinguish all basis function all
basis functions :math:`f_j`. To avoid this aliasing issue, we will  use a uniform grid
with Fourier modes strictly below the Nyquist limit.

Another important resolution consideration is truncated Fourier approximation of
:math:`\mathcal{A}` and :math:`\mathcal{W}` matrices, because such truncated
approximation can reach negative values. For example,

.. math::

   \begin{aligned}
   \mathcal W&=(1+\cos\theta)^2+0.1
   \\
   &=1.6+2\cos\theta+0.5\cos2\theta\ge0.1,
   \end{aligned}

while dropping the second harmonic leaves  :math:`\widetilde{\mathcal
W}=1.6+2\cos\theta`, whose minimum is :math:`-0.4`. On the three-function basis
:math:`(1,\cos\theta,\cos2\theta)`, normalized exact integrals give

.. math::

   \begin{aligned}
   M=\begin{pmatrix}1.6&1&0.25\\1&0.925&0.5\\0.25&0.5&0.8\end{pmatrix},\quad
   \widetilde M=\begin{pmatrix}1.6&1&0\\1&0.8&0.5\\0&0.5&0.8\end{pmatrix}.
   \end{aligned}

Observe that the first matrix is positive definite, while the second one is indefinite.
However, it is not necessary to preserve harmonics that never enter the selected basis
product. Defining cosine moments :math:`\mathcal W_c(k)=\langle\mathcal
W\cos\alpha_k\rangle` and :math:`\mathcal A_c(k)=\langle\mathcal A\cos\alpha_k\rangle`,
with :math:`\alpha_k=m\theta-n\zeta`, the products of cosines or sines give

.. math::

   \begin{aligned}
   M_{ij}&=\tfrac12[\mathcal W_c(k_i-k_j)+\mathcal W_c(k_i+k_j)],\\
   K_{ij}&=\tfrac12\kappa_i\kappa_j[\mathcal A_c(k_i-k_j)-\mathcal A_c(k_i+k_j)],
   \\
   \kappa_i&=m_i\iota-n_i.
   \end{aligned}

Therefore, if every required sum and difference moment is retained, Fourier lookup will
produce the same matrix as the quadrature.

References
----------

.. [Spong2003] D. A. Spong, R. Sanchez, and A. Weller,
   *Shear Alfvén continua in stellarators*,
   Physics of Plasmas **10**, 3217--3224 (2003).
   https://doi.org/10.1063/1.1590316
   In particular, Sec. II and Eqs. (1)--(8).

.. [Paul2025] E. J. Paul, A. Hyder, E. Rodríguez, R. Jorge, and A. Knyazev,
   *The shear Alfvén continuum of quasisymmetric stellarators*,
   Journal of Plasma Physics **91**, E101 (2025).
   https://doi.org/10.1017/S0022377825100524
   In particular, Secs. 2--3 and 5.1.

.. [Schwab1993] C. Schwab,
   *Ideal magnetohydrodynamics: Global mode analysis of three-dimensional
   plasma configurations*, Physics of Fluids B **5**, 3195--3206 (1993).
   https://doi.org/10.1063/1.860656
   In particular, p. 3197, Eqs. (12)--(13) and the accompanying discussion.
