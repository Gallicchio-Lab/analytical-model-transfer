from __future__ import print_function
import pandas as pd
import numpy as np
import random
import math
import pickle
from numpy.polynomial.hermite import hermgauss
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ## Class Based on Tensorflow to Optimize the Analytic Model of Alchemical Binding
# 
# Author: emilio.gallicchio@gmail.com, [compmolbiophysbc.org](http://compmolbiophysbc.org)
# 
# References:
# 
# 1. Denise Kilburg and Emilio Gallicchio. Analytical Model of the Free Energy of Alchemical Molecular Binding. J. Chem. Theory Comput. 14, 6183–6196 (2018). [doi:10.1021/acs.jctc.8b00967](http://dx.doi.org/doi:10.1021/acs.jctc.8b00967)  [pdf of submitted manuscript](https://www.dropbox.com/s/td1mlagbbg0siqp/analytical_theory_ms4.pdf?dl=0)
# 
# 2. Rajat K. Pal and Emilio Gallicchio. Perturbation Potentials to Overcome Order/Disorder Transitions in Alchemical Binding Free Energy Calculations. J. of Chem. Phys.. 151, 124116 (2019).[doi:10.1063/1.5123154](https://dx.doi.org/10.1063/1.5123154)  [Arxiv](https://arxiv.org/abs/1907.06636)
# 
# 3. Emilio Gallicchio and Ronald M Levy, Recent Theoretical and Computational Advances for Modeling Protein-Ligand Binding Affinities. Advances in Protein Chemistry and Structural Biology, 85, 27-80, (2011). [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3445424/)    [doi: 10.1016/B978-0-12-386485-7.00002-8](https://dx.doi.org/10.1016%2FB978-0-12-386485-7.00002-8)
# 
# ### The Model
# 
# The model (see papers) provides the probability density $p_\lambda(u)$ of the binding energy as a function of the alchemical progress parameter $\lambda$ for an alchemical perturbation potential of the form $W_\lambda(u)$. This software module derives the parameters of the model by Maximum Likelihood estimation using the set of binding energies from alchemical molecular simulations at a set of $\lambda$ values. 
# 
# The central quantity of the model is $p_0(u)$, the probability density of the binding energy $u$ in the uncoupled ensemble ($\lambda=0$) which is assumed to be the weighted sum of a small number of modes $i$:
# $$
# p_0(u) = \frac{\sum_{i=1}^{n_{\rm modes}} c_i p_{0i}(u)}{\sum_{i=1}^{n_{\rm modes}} c_i}
# $$
# where $c_i$ are adjustable weights and $p_{0i}(u)$ is the probability density corresponding to mode $i$ described analytically as:
# $$
# p_{0i}(u) = p_{bi} g(u;\bar{u}_{bi},\sigma_{bi}) + (1-p_{bi}) \int_{0}^{+\infty}p_{WCA} (u';n_{li},\epsilon_{i},\tilde{u}_i) g(u-u';\bar{u}_{bi},\sigma_{bi}) du' 
# $$
# 
# where $g(u;\bar{u},\sigma)$ is the normalized Gaussian density function of mean $\bar{u}$ and standard deviation $\sigma$ and
# $$
#   p_{WCA}(u;n_l , \epsilon, \tilde{u}) = n_{l}\left[1-\frac{(1+x_{C})^{1/2}}{(1+x)^{1/2}}\right]^{n_{l}-1} 
#   \frac{H(u)}{4\epsilon_{LJ}}\frac{(1+x_{C})^{1/2}}{x(1+x)^{3/2}}\, , 
# $$
# where $x = \sqrt{u/\epsilon+\tilde{u}/\epsilon+1}$ and $x_C = \sqrt{\tilde{u}/\epsilon+1}$. The model for each mode $i$ depends on a number of adjustable parameters corresponding to the following physical quantities:
# 
# -  $c_i$: weight of mode $i$
# -  $p_{bi}$: probability that no atomic clashes occur while in binding mode $i$ at $\lambda =0$
# -  $\bar{u}_{bi}$: the average background interaction energy of binding mode $i$ at $\lambda = 0$
# -  $\sigma_{bi}$: the standard deviation of background interaction energy of binding mode $i$
# -  $n_{li}$: the effective number of statistical uncorrelated atoms of the ligand in  binding mode $i$
# -  $\epsilon_{i}$: the effective $\epsilon$ parameter of an hypothetical Lennard-Jones interaction energy potential describing the receptor-ligand interaction in binding mode $i$
# -  $\tilde{u}_i$: the binding energy value above which the collisional energy is not zero in binding mode $i$
# 
# All other quantities of the alchemical transformation can be obtained from $p_0(u)$. In particular, given $p_0(u)$, the probability density for the binding energy $u$ for the state with perturbation potential $W_\lambda (u)$ is 
# $$
# p_{\lambda}(u)=\frac{1}{K(\lambda)}p_{0}(u) \exp\left[-\beta W_\lambda (u)\right]
# $$
# where $\beta = 1/k_B T$,
# $$
# K(\lambda)=\int_{-\infty}^{+\infty}du \, p_{0}(u) \exp\left[-\beta W_\lambda (u)\right]  =\langle \exp\left[-\beta W_\lambda (u)\right] \rangle_{\lambda=0}
# $$
# is the excess component of the equilibrium constant for binding and
# $$
# \Delta G_b(\lambda) = - \frac{1}{\beta} \ln K(\lambda) \label{eq:gblambda}
# $$
# is the corresponding excess binding free energy profile. Note that for
# a linear perturbation potential, $W_\lambda(u) = \lambda u$,
# in which case $K(\lambda)$ is the double-sided Laplace
# transform of $p_\lambda(u)$.
# 
# The parameters $\theta$ of the model are obtained by Maximum Likelihood estimation. For the linear perturbation potential, for example, the negative of the log-likelihood function is:
# $$
# -\mathcal{\ln L}(\theta)=
# -\sum_{j}\ln p_{\lambda_{j}}(u_{j}|\theta)=
# -\sum_{j}\ln [ e^{-\lambda_{j}u_{j}} p_{0}(u_{j}|\theta)/K(\lambda_{j}|\theta)]
# $$
# where $\theta$ represents the parameters ($u_{bi}$, $\sigma_{bi}$, etc.)
# 


class femodel_tf_optimizer(object):
    """
    Class to optimize analytic binding free energy model with Tensorflow
    """

    """
    ### `applyunits()` converts parameter values from reduced energy units $E/k_BT$ to kcal/mol
    """

    def applyunits(self, ub, sb, pb, elj, uce, nl):
        return [ub*self.kT, sb*self.kT, pb, elj*self.kT, uce, nl]

    """
    ### Constraint functions to limit parameter values within a range
    
    **Parameter Scaling**. The parameters being optimized can have very different ranges of variation. To facilitate scale-free and homogeneous parameter updates, the model parameters (see above) are expressed by means of dimensionless and scale-normalized auxiliary $x$ variables. A parameter $\theta$ is expressed as $\theta = s x + m$, where $x$ is the dimensionless renormalized variable that is optimized, $s$ is a scaling factor, and $m$ is an offset.
 
    Minimum and maximum limits ($\theta_{\rm min}$, $\theta_{\rm max}$) for the parameter $\theta$ translate into
    $$ x_{\rm min} = (\theta_{\rm min} - m)/s$$
    $$ x_{\rm max} = (\theta_{\rm max} - m)/s$$
    limits for $x$. only weight parameters don't have limits.
    """
    
    def consucx(self, ucx): 
        minucex = (self.min_uce_t - self.mid_params_uce_t)/self.scale_params_uce_t
        maxucex = (self.max_uce_t - self.mid_params_uce_t)/self.scale_params_uce_t
        return tf.minimum(maxucex, tf.maximum(minucex,ucx))

    def consnlx(self, nlx):
        minnlx = (self.min_nl_t - self.mid_params_nl_t)/self.scale_params_nl_t
        maxnlx = (self.max_nl_t - self.mid_params_nl_t)/self.scale_params_nl_t
        return tf.minimum(maxnlx, tf.maximum(minnlx, nlx))

    def conspbx(self, pbx):#forces pb to be within 0 and 1
        minpbx = (self.min_pb_t - self.mid_params_pb_t)/self.scale_params_pb_t
        maxpbx = (self.max_pb_t - self.mid_params_pb_t)/self.scale_params_pb_t
        return tf.minimum(maxpbx,tf.maximum(minpbx,pbx))

    def consweightx(self, wx):#forces weights to be within 0 and 1
        return tf.minimum(self.maxwx_t,tf.maximum(self.minwx_t,wx))

    def consubx(self, ubx):
        minubx = (self.min_ub_t - self.mid_params_ub_t)/self.scale_params_ub_t
        maxubx = (self.max_ub_t - self.mid_params_ub_t)/self.scale_params_ub_t
        return tf.minimum(maxubx, tf.maximum(minubx, ubx))
    
    def conssbx(self, sbx):
        minsbx = (self.min_sb_t - self.mid_params_sb_t)/self.scale_params_sb_t
        maxsbx = (self.max_sb_t - self.mid_params_sb_t)/self.scale_params_sb_t
        return tf.minimum(maxsbx, tf.maximum(minsbx, sbx))

    def conseljx(self, ex):
        mineljx = (self.min_elj_t - self.mid_params_elj_t)/self.scale_params_elj_t
        maxeljx = (self.max_elj_t - self.mid_params_elj_t)/self.scale_params_elj_t
        return tf.minimum(maxeljx, tf.maximum(mineljx, ex))

    # Convert list of range elements(min,max) to two lists of each parameter and then create list of tuples
    def convert_range_list_to_min_max_list(self, range_list):
        output_min = []
        output_max = []
        for index, tuple in enumerate(range_list):
            output_min.append(tuple[0])
            output_max.append(tuple[1])
        return [output_min, output_max]

   # Constraint that prevents bound component from not overreaching past the most minimun sample, umin
    # def consubx(self):
    #     return self.ub_t-tf.pow(self.sb_t,2)-(3.*self.sb_t)
    
    # def consubx_smaller(self):
    #     return self.penalty_force_constant*tf.pow(self.consubx() - self.umin*tf.ones([tf.size( self.ubx_t)], dtype=tf.float64), 2)

    # def consubx_larger(self):
    #     return tf.zeros([tf.size(self.ubx_t)], dtype=tf.float64)

    """
    ### Methods to override in child classes
    
     There are two main customization aspects:
     1. The soft core function $u_{\rm sc} = u_{\rm sc}(u)$
     2. The alchemical potential functions $W_\lambda(u)$
     
     The defaults below assume no soft core ($u_{\rm sc}(u) = u$) and a linear alchemical potential, $W_\lambda(u) = \lambda u$.
     
     -  `inverse_soft_core_function()` returns the inverse of the soft core function, that is it returns the list of $u$'s corresponding the input list of $u_{\rm sc}$'s
     
     -  `der_soft_core_function()` returns $d u_{\rm sc}(u)/d u$ for an input list of $u$'s
     
     -  `alchemical_potential()` returns $W_\lambda(u)$ for input lists of $\lambda$'s and $u_{\rm sc}$'s: $W_{\lambda_i}(u_{{\rm sc,}i})$.
     
     -  `alchemical_potential_x()` same as above but returns a matrix $W_{\lambda_j}(u_{{\rm sc,}i})$ of alchemical potential energies for every combination of the supplied list of lambda and binding energy values. In particular this function returns a $N \times M$ matrix where $N$ is the number of binding energy samples and $M$ is the number of $\lambda$ values. See [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) for a guide on tensorflow/numpy broadcasting rules.
    """

    def inverse_soft_core_function(self, uscbind):
        return tf.identity(uscbind)
    def der_soft_core_function(self, ubind):# dusc/du
        return tf.ones([tf.size(ubind)], dtype=tf.float64)
    def alchemical_potential_x(self, lmbds, xscbind):
        return xscbind[:,None] * lmbds['Lambda1'][None,:]
    def alchemical_potential(self, lmbds, uscbind):
        return lmbds['Lambda1']*uscbind
    """
    ### Init
     
     It does most of the work. It returns an instance of the `femodel_tf_optimizer`. The input is:
     
     -  `sdm_data`: a dictionary containing the simulation data. It is assumed to contain at a minimum the following fields: `cycle`, `Lambda`, `ebind`, `Lambda1`, `Lambda2`, `alpha`, `u0`, `w0`. Here `cycle` is a time-stamp, `ebind` is the binding energy, and `Lambda`, `Lambda1`, ... are the parameters of the Ilogistic perturbation function. For a linear perturbation function $\lambda = \lambda_1 = \lambda_2$, and $w_0 = 0$.
     
     -  `reference_params`: a dictionary containing the initial values of the parameters, which are also taken the $m$ mid-point values of the parameter scaling function (see **Parameter Scaling** above). Each item is a list of values, one for each mode:
         .  `ub`: the $\bar{u}_{bi}$ parameters.
         .  `sb`: the $\sigma_{bi}$ parameters.
         .  `pb`: the $p_{bi}$ parameters.
         .  `elj`: the $\epsilon_i$ parameters.
         .  `uce`: $\tilde{u_i}/\epsilon_i$
         .  `nl`: the $n_{li}$ parameters.
         .  `wg`: the $c_i$ weights.     
     
         .  `temperature`: reference simulation temperature. Only used to define reduced energies $E/k_B T$.
     
     The following arguments are optional:
     
     -  `xparams`: initial values of the $x$ parameters (see **Parameter Scaling** above). Used for restarting the optimization. Default is `None` in which case they are set as zero.
     -  `scale_params`: a dictionary with the same format as `reference_params` but holding the scaling parameters $s$ (see **Parameter Scaling** above).
     -  `discard`: samples from `sdm_data` with `cycle` < `discard` will be ignored
     -  `learning_rate`: the learning rate for the Tensorflow minimizer
    """
    
    # Generate tensorflow graph
    def __init__(self, sdm_data, reference_params, temperature, range_params = None,
                 xparams = None,  scale_params = None, discard = 0, learning_rate = 0.01):
        

        self.kT = 1.986e-3*temperature # [kcal/mol]
        beta = 1./self.kT
        self.beta = beta

        self.learning_rate = learning_rate
        
        nmodes = len(reference_params['ub'])
        self.nmodes = nmodes
        
        # Initial parameters [ub, sb, pb, elj, uce, nl]
        reference_ub     = reference_params['ub']
        reference_sb     = reference_params['sb']
        reference_pb     = reference_params['pb']
        reference_elj    = reference_params['elj']
        reference_uce    = reference_params['uce']
        reference_nl     = reference_params['nl'] 
        reference_wg     = reference_params['wg']
        
        assert len(reference_sb)  == nmodes, "invalid number of sb parameters, %d instead of %d" % (len(reference_sb), nmodes)
        assert len(reference_pb)  == nmodes, "invalid number of pb parameters, %d instead of %d" % (len(reference_pb), nmodes)
        assert len(reference_elj) == nmodes, "invalid number of elj parameters, %d instead of %d" % (len(reference_elj), nmodes)
        assert len(reference_uce) == nmodes, "invalid number of uce parameters, %d instead of %d" % (len(reference_uce), nmodes)
        assert len(reference_nl)  == nmodes, "invalid number of nl parameters, %d instead of %d" % (len(reference_nl), nmodes)
        assert len(reference_wg)  == nmodes, "invalid number of weight parameters, %d instead of %d" % (len(reference_wg), nmodes)

        # Scale factor of parameters
        if scale_params is not None:
            scale_ub     = scale_params['ub']
            scale_sb     = scale_params['sb']
            scale_pb     = scale_params['pb']
            scale_elj    = scale_params['elj']
            scale_uce    = scale_params['uce']
            scale_nl     = scale_params['nl']
            scale_wg     = scale_params['wg']
            assert len(scale_ub)  == nmodes, "invalid number of ub parameters, %d instead of %d" % (len(scale_ub), nmodes)
            assert len(scale_sb)  == nmodes, "invalid number of sb parameters, %d instead of %d" % (len(scale_sb), nmodes)
            assert len(scale_pb)  == nmodes, "invalid number of pb parameters, %d instead of %d" % (len(scale_pb), nmodes)
            assert len(scale_elj) == nmodes, "invalid number of elj parameters, %d instead of %d" % (len(scale_elj), nmodes)
            assert len(scale_uce) == nmodes, "invalid number of uce parameters, %d instead of %d" % (len(scale_uce), nmodes)
            assert len(scale_nl)  == nmodes, "invalid number of nl parameters, %d instead of %d" % (len(scale_nl), nmodes)
            assert len(scale_wg)  == nmodes, "invalid number of weight parameters, %d instead of %d" % (len(scale_wg), nmodes)
            
        else:
            scale_ub     = [1. for i in range(nmodes)]
            scale_sb     = [1. for i in range(nmodes)]
            scale_pb     = [1. for i in range(nmodes)]
            scale_elj    = [1. for i in range(nmodes)]
            scale_uce    = [1. for i in range(nmodes)]
            scale_nl     = [1. for i in range(nmodes)]
            scale_wg     = [1. for i in range(nmodes)]
            
        self.mid_params_ub_t  = [tf.constant(p, dtype=tf.float64) for p in reference_ub]
        self.mid_params_sb_t  = [tf.constant(p, dtype=tf.float64) for p in reference_sb]
        self.mid_params_pb_t  = [tf.constant(p, dtype=tf.float64) for p in reference_pb]
        self.mid_params_elj_t = [tf.constant(p, dtype=tf.float64) for p in reference_elj]
        self.mid_params_uce_t = [tf.constant(p, dtype=tf.float64) for p in reference_uce]
        self.mid_params_nl_t  = [tf.constant(p, dtype=tf.float64) for p in reference_nl]
        self.mid_params_wg_t  = [tf.constant(p, dtype=tf.float64) for p in reference_wg]
        
        self.scale_params_ub_t  = [tf.constant(p, dtype=tf.float64) for p in scale_ub]
        self.scale_params_sb_t  = [tf.constant(p, dtype=tf.float64) for p in scale_sb]
        self.scale_params_pb_t  = [tf.constant(p, dtype=tf.float64) for p in scale_pb]
        self.scale_params_elj_t = [tf.constant(p, dtype=tf.float64) for p in scale_elj]
        self.scale_params_uce_t = [tf.constant(p, dtype=tf.float64) for p in scale_uce]
        self.scale_params_nl_t  = [tf.constant(p, dtype=tf.float64) for p in scale_nl]
        self.scale_params_wg_t =  [tf.constant(p, dtype=tf.float64) for p in scale_wg]
        
        # Render limits of parameters to lists of min,max
        min_ub_list = self.convert_range_list_to_min_max_list(range_params['ub'])[0]
        max_ub_list = self.convert_range_list_to_min_max_list(range_params['ub'])[1]

        min_sb_list = self.convert_range_list_to_min_max_list(range_params['sb'])[0]
        max_sb_list = self.convert_range_list_to_min_max_list(range_params['sb'])[1]

        min_elj_list = self.convert_range_list_to_min_max_list(range_params['elj'])[0]
        max_elj_list = self.convert_range_list_to_min_max_list(range_params['elj'])[1]

        min_uce_list = self.convert_range_list_to_min_max_list(range_params['uce'])[0]
        max_uce_list = self.convert_range_list_to_min_max_list(range_params['uce'])[1]

        min_nl_list = self.convert_range_list_to_min_max_list(range_params['nl'])[0]
        max_nl_list = self.convert_range_list_to_min_max_list(range_params['nl'])[1]

        min_pb_list = self.convert_range_list_to_min_max_list(range_params['pb'])[0]
        max_pb_list = self.convert_range_list_to_min_max_list(range_params['pb'])[1]

        # Render min,max lists to tensors 
        self.min_ub_t = tf.constant([p for p in (min_ub_list)], dtype=tf.float64)
        self.max_ub_t = tf.constant([p for p in (max_ub_list)], dtype=tf.float64)

        self.min_sb_t = tf.constant([p for p in (min_sb_list)], dtype=tf.float64)
        self.max_sb_t = tf.constant([p for p in (max_sb_list)], dtype=tf.float64)

        self.min_pb_t = tf.constant([p for p in (min_pb_list)], dtype=tf.float64)
        self.max_pb_t = tf.constant([p for p in (max_pb_list)], dtype=tf.float64)

        self.min_uce_t = tf.constant([p for p in (min_uce_list)], dtype=tf.float64)
        self.max_uce_t = tf.constant([p for p in (max_uce_list)], dtype=tf.float64)

        self.min_elj_t = tf.constant([p for p in (min_elj_list)], dtype=tf.float64)
        self.max_elj_t = tf.constant([p for p in (max_elj_list)], dtype=tf.float64)

        self.min_nl_t = tf.constant([p for p in (min_nl_list)], dtype=tf.float64)
        self.max_nl_t = tf.constant([p for p in (max_nl_list)], dtype=tf.float64)

        # Miscellania constants
        self.zero = tf.constant(0.0, dtype=tf.float64)
        self.pi = tf.constant(np.pi, dtype=tf.float64)
        self.eps = tf.constant(1.e-6, dtype=tf.float64) #small regularization factor
        self.sq2 = tf.constant(np.math.sqrt(2.), dtype=tf.float64)
        self.sq2pi = tf.math.sqrt(2.*self.pi)
        self.epsu = tf.constant(1.e-8, dtype=tf.float64) #smallest positive perturbation energy

        # Parameter constraints for weights
        self.maxweight_t = tf.constant([ 1.0 for i in range(self.nmodes)], dtype=tf.float64)
        self.minweight_t = tf.constant([ 0.0 for i in range(self.nmodes)], dtype=tf.float64)
        self.maxwx_t = (self.maxweight_t - self.mid_params_wg_t)/self.scale_params_wg_t
        self.minwx_t = (self.minweight_t - self.mid_params_wg_t)/self.scale_params_wg_t
        
        if xparams is not None:
            #[ub, sb, pb, elji, uc, nl]
            assert len(xparams['ub'])   == nmodes, "invalid number of x ub parameters, %d instead of %d"  % (len(xparams['ub']), nmodes)
            assert len(xparams['sb'])   == nmodes, "invalid number of x sb parameters, %d instead of %d"  % (len(xparams['sb']), nmodes)
            assert len(xparams['pb'])   == nmodes, "invalid number of x pb parameters, %d instead of %d"  % (len(xparams['pb']), nmodes)
            assert len(xparams['elj'])  == nmodes, "invalid number of x elj parameters, %d instead of %d" % (len(xparams['elj']), nmodes)
            assert len(xparams['uce'])  == nmodes, "invalid number of x uce parameters, %d instead of %d" % (len(xparams['uce']), nmodes)
            assert len(xparams['nl'])   == nmodes, "invalid number of x nl parameters, %d instead of %d"  % (len(xparams['nl']), nmodes)
            assert len(xparams['wg'])   == nmodes, "invalid number of x weight parameters, %d instead of %d"  % (len(xparams['wg']), nmodes)
            
            self.ubx_t = tf.Variable(xparams['ub'],  dtype=tf.float64, constraint=self.consubx)
            self.sbx_t = tf.Variable(xparams['sb'],  dtype=tf.float64, constraint=self.conssbx)
            self.pbx_t = tf.Variable(xparams['pb'],  dtype=tf.float64, constraint=self.conspbx)
            self.ex_t  = tf.Variable(xparams['elj'], dtype=tf.float64, constraint=self.conseljx)
            self.ucx_t = tf.Variable(xparams['uce'], dtype=tf.float64, constraint=self.consucx)
            self.nlx_t = tf.Variable(xparams['nl'],  dtype=tf.float64, constraint=self.consnlx)
            self.wgx_t = tf.Variable(xparams['wg'],  dtype=tf.float64, constraint=self.consweightx)
        else:
            zeroxpars = [0. for i in range(nmodes)]
            self.ubx_t = tf.Variable(zeroxpars,  dtype=tf.float64, constraint=self.consubx)
            self.sbx_t = tf.Variable(zeroxpars,  dtype=tf.float64, constraint=self.conssbx)
            self.pbx_t = tf.Variable(zeroxpars,  dtype=tf.float64, constraint=self.conspbx)
            self.ex_t  = tf.Variable(zeroxpars, dtype=tf.float64, constraint=self.conseljx)
            self.ucx_t = tf.Variable(zeroxpars, dtype=tf.float64, constraint=self.consucx)
            self.nlx_t = tf.Variable(zeroxpars,  dtype=tf.float64, constraint=self.consnlx)
            self.wgx_t = tf.Variable(zeroxpars,  dtype=tf.float64, constraint=self.consweightx)


        # These implement the parameter scaling (see **Parameter Scaling** above)
        self.ub_t  = self.ubx_t*self.scale_params_ub_t  + self.mid_params_ub_t
        self.sb_t  = self.sbx_t*self.scale_params_sb_t  + self.mid_params_sb_t
        self.pb_t  = self.pbx_t*self.scale_params_pb_t  + self.mid_params_pb_t
        self.elj_t = self.ex_t*self.scale_params_elj_t + self.mid_params_elj_t
        self.uce_t = self.ucx_t*self.scale_params_uce_t + self.mid_params_uce_t
        self.nl_t  = self.nlx_t*self.scale_params_nl_t  + self.mid_params_nl_t
        self.wg_t  = self.wgx_t*self.scale_params_wg_t  + self.mid_params_wg_t

        # Load the simulation data into 1D tensors (lambda, binding energy data etc)        
        sdm_data['u'] = sdm_data['ebind']*beta
        sdm_data['mask'] = sdm_data['cycle'] > discard
        rel_data = sdm_data.loc[sdm_data['mask'] == True]
        rel_data['u'] = rel_data['u'].astype(np.float64)

        # load (u,lambda) data pairs in constant tensors
        self.usc = tf.constant(np.array(rel_data['u']),dtype=tf.float64)
        self.nsamples = tf.size(self.usc) 
        self.u   = self.inverse_soft_core_function(self.usc)
        self.upsc = self.der_soft_core_function(self.u)
        self.lambdas = {}
        self.lambdas['stateID'] = tf.constant(np.array(rel_data['stateID']),dtype=tf.float64)
        self.lambdas['Lambda1'] = tf.constant(np.array(rel_data['Lambda1']),dtype=tf.float64)
        self.lambdas['Lambda2'] = tf.constant(np.array(rel_data['Lambda2']),dtype=tf.float64)
        self.lambdas['alpha'] = tf.constant(np.array(rel_data['alpha']),dtype=tf.float64)
        self.lambdas['u0'] = tf.constant(np.array(rel_data['u0']),dtype=tf.float64)
        self.lambdas['w0'] = tf.constant(np.array(rel_data['w0']),dtype=tf.float64)


        """
        We use Gauss-Hermite quadrature to evaluate the convolution integrals:
         $$
         \int_{-\infty}^{+\infty} f(x) e^{-x^2} \simeq \sum_{k=1}^n w_k f(x_k)
         $$
         where $x_k$ and $w_k$ are Gauss-Hermite nodes and weights, respectively. To apply the Gauss-Hermite formula we use the variable transformation $x = (u' - u + \bar{u}_b)/\sqrt{2 \sigma^2}$ to the convolution integral obtaining:
         $$
         \int_{0}^{+\infty}p_{WCA} (u') g(u-u') du' = 
         \frac{1}{\sqrt{\pi}} \int_{-\infty}^{+\infty} p_{WCA}(\sqrt{2} \sigma x + u - \bar{u}_b) e^{-x^2}
         $$
         where we assume that $p_{WCA}(u')$ is set to zero for $u'\lt 0$.
        """

        xg,wg = hermgauss(19)

        self.n_gauss = tf.constant(xg.size, dtype=tf.int64)
        self.x_gauss = tf.constant(xg, dtype=tf.float64)
        self.w_gauss = tf.constant(wg, dtype=tf.float64)

        """
         The `gauss_b` tensor holds the value of the normal distribution for each u/parameter combination using broadcasting rules. The resulting tensor is $n_{\rm modes} \times N$, where $n_{\rm modes}$ is the number of modes and $N$ is the number of binding energy samples:
         $$
         g_{ij} = \frac{1}{\sqrt{2} \sigma_p} e^{-(u_j - \bar{u}_{i})^2/2 \sigma_i^2}
         $$

         this tensor contains the value of the normal distribution for each u,parameter combination
         using broadcasting rules
        """
        self.gauss_b = tf.math.exp(-tf.pow(self.u-self.ub_t[:,None],2)/(2.0*tf.pow(self.sb_t[:,None],2)))/(self.sq2pi*self.sb_t[:,None])

        """
         The following defines a 3-D tensor $P{\rm WCA}_{ijk}$ of shape $n_{\rm modes} \times N \times n_G$ holding the values of the $p_{WCA}(u)$ function in which, the $k$ dimension (the fastest) refers to the $n_G$ nodes $x_k$ for the Gauss-Hermite integration sum, the $j$ dimension refers to the $N$ binding energy samples $u_j$, and the $i$ dimension refers to one of the $n_{\rm modes}$ binding modes.
         
         Relative to the definition above, some numerical tricks are employed to avoid NaN's for negative values and to smoothly set $p_{WCA}(u)$ to zero for $u<0$:
         $$
         p_{WCA}(u) = n_{l} \left[ 1 - z(x)^{1/12}  + \epsilon_\delta \right]^{n_{l}-1} 
           \frac{S(u)}{4\epsilon}\frac{(1+x_{C})^{1/2}}{x(1+x)^{3/2}}\, , 
         $$
         $$
         z(x) = {\rm tanh} \left\{   \left[ \frac{(1+x_{C})^{1/2}}{(1+x)^{1/2}} \right]^{12}     \right\}
         $$
         
         -  `xc`: is a 1D tensor of size $n_{\rm modes}$ which holds $x_C = \sqrt{\tilde{u}/\epsilon+1}$ for each mode
         -  `ac`: 1D tensor, same as above but for $\sqrt{1 + x_C}$
         -  `yA`: 3D tensor which implements the variable transformation for the Gauss-Hermite quadrature. It has the same dimensions of the $P{\rm WCA}_{ijk}$ tensor discussed above.
         -  `xA`: 3D tensor of the same shape as `yA` which, together with `x1A`, implements $x = \sqrt{| u/\epsilon+\tilde{u}/\epsilon+1 |}$
         -  `bA`: 3D tensor, $\sqrt(1 + x)$
         -  `z1A`: 3D tensor, implements the $z(x)$ function
         -  `zA`: 3D tensor, implements $z(x)^{1/12}$
         -  `fcore2A`: 3D tensor which implements $\left[ 1 - z(x)^{1/12}  + \epsilon_\delta \right]^{n_{l}-1}$
         -  `fcore3A`: 3D tensor which implements $( 1/4\epsilon) (\sqrt{1+x_{C}}/(x(1+x)^{3/2})$
         -  `fcore4A`: 3D tensor implements the switching function $S(u) = {\rm Sigmoid}(u/\delta_u)$
         -  `pwca`: final product, 3D tensor of shape $n_{\rm modes} \times N \times n_G$
        """
        self.xc = tf.math.sqrt(self.uce_t + 1.)
        self.ac = tf.math.sqrt(1.+ self.xc)
        self.yA  = self.sq2*self.sb_t[:,None, None]*self.x_gauss + self.u[:,None] - self.ub_t[:,None,None]
        self.unoyA = tf.ones(tf.shape(self.yA), dtype=tf.float64)
        self.yA_safe = tf.where(self.yA < self.epsu, self.epsu*self.unoyA, self.yA) 
        self.xA = tf.pow(self.yA_safe/self.elj_t[:,None,None]  + self.uce_t[:,None,None] + 1.,1/2)
        self.bA  = tf.sqrt(1.+ self.xA)
        self.zA  = self.ac[:,None,None]/self.bA
        self.fcore2A = tf.pow(1.-self.zA, self.nl_t[:,None,None]-1.)
        self.fcore3A = self.ac[:,None,None]/(self.xA*tf.pow(self.bA,3)*4.*self.elj_t[:,None,None])
        self.pwca = tf.where(self.yA < self.epsu, 0*self.unoyA, self.nl_t[:,None,None]*self.fcore2A*self.fcore3A)


        """
         This performs the convolution by Gauss-Hermite quadrature:
         $$
         C_{ij} = \sum_k P{WCA}_{ijk} w_k
         $$
         implemented as a matrix times vector operation. The resulting 2D tensor `conv` has shape $n_{\rm modes} \times N$.
        """
        self.conv = tf.linalg.matvec(self.pwca, self.w_gauss)/tf.math.sqrt(self.pi)

        """
        This combines the Gaussian component with the convolution component weighting them according to the `pb` parameters for each mode. The resulting tensor `p0i` of shape $n_{\rm modes} \times N$, holds the $p_0(u)$ function values for each mode.
        """
        self.p0i  = self.pb_t[:,None]*self.gauss_b + (1. - self.pb_t[:,None])*self.conv

        """
        Performs the weighted average to get the overall $p_0(u)$. The weights are stored in the 1D tensor `wg_t` of length $n_{\rm modes}$. The `expand_dims` function converts it to a $1 \times n_{\rm modes}$ function so that we can do a matrix multiplication between the $1 \times  n_{\rm modes}$ weight matrix and the $n_{\rm modes} \times N$ `p0i` tensor. The result is a $1 \times N$ vector as expected for the $p_0(u)$ for each sample $u$.
        """
        self.wsum = tf.reduce_sum(self.wg_t)
        self.weights = tf.expand_dims(self.wg_t,0)/self.wsum
        self.p0 = tf.linalg.matmul(self.weights,self.p0i)

        """
         Computes the probability density of the soft-core binding energy $u_{\rm sc}$: 
         $$
         p_{0}(u_{\rm sc})=p_{0}(u)/(du_{\rm sc}/du)
         $$
        """
        self.p0sc = self.p0/self.upsc

        """
        # Define a grid in binding energy space to evaluate 
        # $$
        # K(\lambda) = \int_{-\infty}^{+\infty} p_0(u_{\rm sc}) e^{-W_\lambda(u_{\rm sc})} du_{\rm sc}
        # $$
        # This is a uniform grid between the smallest and the largest observed soft-core binding energy values--*fix me: not suitable for a calculation without soft-core*.
        # 
        # The grid is stored in the tensor `xsc` of length $N_{\rm grid}$.
        """
        self.umin = tf.math.reduce_min(self.usc)
        self.umax = tf.math.reduce_max(self.usc)
        self.nx = tf.constant(1000.0, dtype=tf.float64)
        self.dxsc = (self.umax-self.umin)/(self.nx-1.)
        self.xsc = tf.range(self.umin,self.umax,self.dxsc)
        self.xu = self.inverse_soft_core_function(self.xsc)
        self.upscX = self.der_soft_core_function(self.xu)


        """
         The code section below mirrors the one above to compute the `pwca` but for the grid in binding energy space rather than for the samples. The result is a 3D tensor `pwcax` of shape $n_{\rm modes} \times N_{\rm grid} \times n_G$.
         
         As above, the `pwcax` tensor is first reduced using the Gauss-Hermite weights to get the convolution. Then the resulting $p_{0i}(u)$ are combined to get the $p_0(u)$ values over the grid.
         
         Then finally $p_0(u_{\rm sc})$ is evaluated as above. 
        """
        self.xcX = tf.math.sqrt(self.uce_t + 1.)
        self.acX = tf.math.sqrt(1.+ self.xcX)
        self.yX  = self.sq2*self.sb_t[:,None, None]*self.x_gauss + self.xu[:,None] - self.ub_t[:,None,None]
        self.unoyX = tf.ones(tf.shape(self.yX), dtype=tf.float64)
        self.yX_safe = tf.where(self.yX < self.epsu, self.epsu*self.unoyX, self.yX)
        self.xX = tf.pow(self.yX_safe/self.elj_t[:,None,None]  + self.uce_t[:,None,None] + 1.,1/2)
        self.bX  = tf.sqrt(1.+ self.xX)
        self.zX  = self.acX[:,None,None]/self.bX
        self.fcore2X = tf.pow(1.-self.zX, self.nl_t[:,None,None]-1.)
        self.fcore3X = self.acX[:,None,None]/(self.xX*tf.pow(self.bX,3)*4.*self.elj_t[:,None,None])
        self.pwcax = tf.where(self.yX < self.epsu, 0*self.unoyX,  self.nl_t[:,None,None]*self.fcore2X*self.fcore3X)
        
        self.convx = tf.linalg.matvec(self.pwcax,self.w_gauss)/tf.math.sqrt(self.pi)
        self.gauss_bx = tf.math.exp(-tf.pow(self.xu-self.ub_t[:,None],2)/(2.0*tf.pow(self.sb_t[:,None],2)))/(self.sq2pi*self.sb_t[:,None])
        self.p0ix  = self.pb_t[:,None]*self.gauss_bx + (1. - self.pb_t[:,None])*self.convx
        self.p0x = tf.linalg.matmul(self.weights,self.p0ix)
        self.p0scx = self.p0x/self.upscX


        """
         Do the $K(\lambda)$ integral
         $$
         K(\lambda) = \int_{-\infty}^{+\infty} p_0(u_{\rm sc}) e^{-W_\lambda(u_{\rm sc})} du_{\rm sc} \simeq \sum_k du_k p_0(u_{{\rm sc},k}) \exp[ -W_{\lambda_j}(u_{{\rm sc},k}) ]
         $$
         for each $\lambda$ in the samples. Here `fsamplesX` is a $1 \times N_{\rm grid}$ tensor and `explX` is a 2D tensor of shape $N_{\rm grid} \times N$. After matrix multiplication, which implements the summation, we get a $1 \times N$ tensor, that is $K(\lambda)$ for each sample in the input. *fix me: many $\lambda$'s are repeated so one could save time by computing only the unique ones and keeping track of how many of each kind are there*
        """
        self.fsamplesX = self.dxsc*self.p0scx
        self.explX = tf.math.exp(- self.alchemical_potential_x(self.lambdas,self.xsc))
        self.kl = tf.matmul(self.fsamplesX,self.explX)

        # Finally, we define the "cost" function, that is the negative of the log likelihood:
        # $$
        # -\mathcal{\ln L}=
        # -\sum_{j}\ln p_{\lambda_{j}}(u_{j})
        # $$
        # where
        # $$
        # p_{\lambda_j}(u_j) = \exp[-W_{\lambda_j}(u_{j})] p_{0}(u_{j})/K(\lambda_{j})
        # $$
        self.expl = tf.math.exp(- self.alchemical_potential(self.lambdas,self.usc))
        self.pkl = self.expl*self.p0sc/self.kl

        # Cost penalty associated to umin constraint
        # self.penalty_force_constant = tf.constant(100., dtype=tf.float64)
        # self.pen_cond = tf.math.greater(self.consubx(), self.umin*self.umin*tf.ones([tf.size( self.ubx_t)], dtype=tf.float64))
        # self.cost_pen = tf.where(self.pen_cond, self.consubx_larger(), self.consubx_smaller())

        # self.cost = -tf.reduce_sum(tf.math.log(self.pkl)) + tf.reduce_sum(self.cost_pen)

        #Cost penalty associated with unordered nl's
        nldiffs = self.nl_t[1:] - self.nl_t[:-1]
        self.penalty = tf.reduce_sum(tf.constant(10.0,dtype=tf.float64) * tf.where(nldiffs > 0, self.zero*nldiffs, tf.pow(nldiffs,2))) 
        eljdiffs = self.elj_t[1:] - self.elj_t[:-1]
        self.penalty += tf.reduce_sum(tf.constant(10.0,dtype=tf.float64) * tf.where(eljdiffs > 0, self.zero*eljdiffs, tf.pow(eljdiffs,2))) 
        ucediffs = self.uce_t[1:] - self.uce_t[:-1]
        self.penalty += tf.reduce_sum(tf.constant(10.0,dtype=tf.float64) * tf.where(ucediffs > 0, self.zero*ucediffs, tf.pow(ucediffs,2))) 
        
        self.cost = -tf.reduce_sum(tf.math.log(self.pkl)) + self.penalty

        # gradient probes
        self.gubx_t = tf.gradients(self.cost,self.ubx_t)
        self.gsbx_t = tf.gradients(self.cost,self.sbx_t)
        self.gpbx_t = tf.gradients(self.cost,self.pbx_t)
        self.gex_t  = tf.gradients(self.cost,self.ex_t)
        self.gucx_t = tf.gradients(self.cost,self.ucx_t)
        self.gnlx_t = tf.gradients(self.cost,self.nlx_t)
        self.gwgx_t = tf.gradients(self.cost,self.wgx_t)

        # Initializer to start tensorflow optimizer session
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)
        self.init = tf.global_variables_initializer()
        self.gradient_cost = self.optimizer.compute_gradients(self.cost)

