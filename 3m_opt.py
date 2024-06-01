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
from femodel_tf_optimizer import femodel_tf_optimizer
import os, sys
import argparse


class femodel(femodel_tf_optimizer):
    """
    Class to optimize analytic binding free energy model with Tensorflow
    """
    kBT = 1.986e-3*300.0
    epsim = tf.constant(1.e-8, dtype=tf.float64)
    
    """    
    Rational soft-core function.

    Special case: when $u_c = 0$, 
    $$
    u_{\rm sc,0}(u)=
    \begin{cases}
    u                                                         & u \le 0 \\
    u_{\rm max} f_{\rm sc}(y)                                 & u  >  0
    \end{cases}
    $$
    with
    $$
    y = \frac{u}{u_{\rm max}}
    $$
    $$
    f_{\rm sc}(y) = \frac{z(y)^{a}-1}{z(y)^{a}+1} \, ,
    $$
    and
    $$
    z(y)=1+2 y/a + 2 (y/a)^2
    $$
    
    Inverse: $u = u_{\rm sc,0}$ for $u \le 0$. For $u > 0$:
    $$
    f_{\rm sc} = u_{\rm sc,0}/{u_{\rm max}}
    $$
    $$
    z = \left( \frac{1 + f_{\rm sc}}{1 - f_{\rm sc}}  \right)^(1/a)
    $$
    y = \frac{a}{2} \left( \sqrt{2 z - 1} -1  \right)
    $$
    $$
    u = u_{\rm max} y
    $$

    The derivative is:
    $$
    \frac{d u_{\rm sc,0}(u) }{du} = \frac{4 z^{a - 1}{(z^a + 1)^2} ( 1 + 2 \frac{y}{a} )
    $$
    for $u > 0$ and $1$ otherwise

    The general case $u_c \ne 0$ is obtained by shifting $u_{\rm sc}(u)$ up by $u_c$ and right by $u_c$:
    $$
    u_{\rm sc}(u) = u_{\rm sc,0}(u - u_c) + u_c
    $$
    For consistency, $u_{\rm max}$ is also shifted up by $u_c$. So the general case $u_c \ne 0$ is obtained
    by the special case $u_c = 0$ by the replacements
    $$
    u \rightarrow u - u_c
    $$
    $$
    u_{\rm max} \rightarrow u_{\rm max} - u_c
    $$

    Inverse for the general case: same as for the special case with the replacements above and:
    $$
    u = ( u_{\rm max} - u_c ) y + u_c
    $$

    Using the chain rule, the derivative for the general case has the same expression as the special case
    with the same replacements.
    

    """
    alphasc = tf.constant(1./16.,dtype=tf.float64)
    umaxsc = tf.constant(50.0/kBT, dtype=tf.float64)
    ubcore = tf.constant(0.0/kBT, dtype=tf.float64)
    def inverse_soft_core_function(self,usc):
        uno = tf.ones([tf.size(usc)], dtype=tf.float64)
        usc_safe1 = tf.where(usc < femodel.ubcore + femodel.epsim,       uno*(femodel.ubcore + femodel.epsim), usc)
        usc_safe2 = tf.where(usc_safe1 > femodel.umaxsc - femodel.epsim, uno*(femodel.umaxsc - femodel.epsim), usc_safe1)
        fsc = (usc_safe2-femodel.ubcore)/(femodel.umaxsc-femodel.ubcore)
        z = tf.math.pow((1.+fsc)/(1.-fsc), 1./femodel.alphasc)
        y = 0.5*femodel.alphasc*(-1.0 + tf.math.sqrt(2.*z - 1.))
        return tf.where(usc <= femodel.ubcore,
                        usc,
                        femodel.ubcore + (femodel.umaxsc - femodel.ubcore)*y)
    
    def der_soft_core_function(self,u):# dusc/du
        y = (u - femodel.ubcore)/(femodel.umaxsc-femodel.ubcore)
        z = 1 + 2.*y/femodel.alphasc + 2.*tf.math.pow(y/femodel.alphasc,2)
        za = tf.math.pow(z,femodel.alphasc)
        return tf.where(u <= femodel.ubcore,
                        tf.ones([tf.size(u)], dtype=tf.float64),
                        (4*(za/z)/tf.math.pow((za+1),2))*(1 + 2.*y/femodel.alphasc))

    def alchemical_potential_x(self, lmbds, xscbind):
        return tf.math.log(1. + tf.math.exp(- (xscbind[:,None]-lmbds['u0'][None,:]) * lmbds['alpha'][None,:])) * ((lmbds['Lambda2'][None,:] - lmbds['Lambda1'][None,:])/(lmbds['alpha'][None,:]+self.eps)) + xscbind[:,None] * lmbds['Lambda2'][None,:] + lmbds['w0'][None,:]
    
    def alchemical_potential(self, lmbds, uscbind):
        # self.eps is there to cover the case lambda1 = lambda2 and alpha = 0 in which case we have a linear alchemical potential
        return ((lmbds['Lambda2'] - lmbds['Lambda1'])/(lmbds['alpha']+self.eps))*tf.math.log(1. + tf.math.exp(-lmbds['alpha']*(uscbind-lmbds['u0']))) + lmbds['Lambda2']*uscbind + lmbds['w0']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optimize", action="store_true", default=0,
                        help="Optimize parameters. Otherwise plot only.")
    parser.add_argument("-c", "--ncycles", type=int, default=100,
                        help="Number of optimization cycles")
    parser.add_argument("-n", "--nsteps", type=int, default=10,
                        help="Number of optimization steps in each cycle")
    parser.add_argument("-r", "--restart", action="store_true", default=0,
                        help="Restart optimization from the parameters saved in the pickle restart file")
    parser.add_argument("-b", "--basename",
                        help="Basename of the optimization job. Defaults to basename of this script.")
    parser.add_argument("-d", "--datafile", default="repl.cycle.potE.temp.lambda.ebind.lambda1.lambda2.alpha.u0.w0.dat",
                        help="Data file")
    parser.add_argument("-l", "--leg", type=float, default=1.0,
                        help="Value of state ID at which to plot the distributions")
    parser.add_argument("-s", "--stateID", nargs="+", type=int, default=0,
                        help="Values of state ID at which to plot the distributions")
    args = parser.parse_args()
    
    restart = args.restart
    production = args.optimize
    testarea = not production

    if args.basename:
        basename = args.basename
    else:
        basename = os.path.basename(os.path.splitext(sys.argv[0])[0])
        
    datafile = args.datafile
    ncycles = args.ncycles
    nsteps = args.nsteps
    states = args.stateID
    
    leg = args.leg

    print("Leg = ", leg)
    if leg == 1:
        direction = leg
    else:
        direction = -1.0
    print("Direction = ", direction)

    sdm_data_raw = pd.read_csv(datafile, delim_whitespace=True,
                           header=None,names=["cycle", "stateID", "temp", "direct", "Lambda1",
                                              "Lambda2", "alpha", "u0", "w0", "potE", "ebind", "bias"])

    lam_col = []
    for row in sdm_data_raw['stateID']:
        lam = 0.05*row
        if (lam > 0.5):
            lam = lam - 0.05
        lam_col.append(lam)

    #print(sdm_data_raw.stateID)

    sdm_data_raw.insert(2, "Lambda", lam_col)

    #print(sdm_data_raw)

    sdm_data = sdm_data_raw[sdm_data_raw["direct"] == direction]

    #print(sdm_data)

    temperature = 300.0
    kT = 1.986e-3*temperature # [kcal/mol]
    beta = 1./kT

    nmodes = 3

    reference_params = {}
    reference_params['ub'] = [ -27.6684*beta, -16.0369*beta, -22.7318*beta ]
    reference_params['sb'] = [ 2.6128*beta,  4.0868*beta, 4.6583*beta ]
    reference_params['pb'] = [ 5.968e-12, 6.136e-19, 7.749e-14 ]
    reference_params['elj'] = [ 10.945*beta, 15.3378*beta, 12.043*beta ]
    reference_params['uce'] = [ 6.999, 40.929, 8.08 ]
    reference_params['nl']  = [ 5.257, 59.535, 22.24 ]
    reference_params['wg'] =  [ 1.409e-6, 1.741e-1, 1.882e-3 ]
    
    scale_params = {}
    scale_params['ub'] =  [ 1.*beta, 1.*beta, 1.*beta ]
    scale_params['sb'] =  [ 0.1*beta, 0.1*beta, 0.1*beta ]
    scale_params['pb'] =  [ 1.e-12, 1.e-19, 1.e-14 ]
    scale_params['elj'] = [ 0.1*beta, 0.1*beta, 0.1*beta ]
    scale_params['uce'] = [ 0.1, 0.1, 0.1 ]
    scale_params['nl']  = [ 0.1, 0.1, 0.1 ]
    scale_params['wg'] =  [ 1.e-6, 1.e-1, 1.e-3 ]

    range_params = {}
    range_params['ub'] = [ (-30.0*beta, 0.0*beta), (-25.0*beta, 50.0*beta), (-30.0*beta, 50.0*beta) ]
    range_params['sb'] = [ (1.0*beta, 6.0*beta), (2.0*beta, 6.0*beta), (2.0*beta, 5.0*beta) ]
    range_params['pb'] = [ (0.0, 1.0), (0.0, 1.0), (0.0, 1.0) ]
    range_params['elj'] = [ (1.0*beta, 12.0*beta), (12.0*beta, 40.0*beta), (10.0*beta, 30.0*beta) ]
    range_params['uce'] = [ (1.0, 8.0), (0.0, 60.0), (7.0, 40.0) ]
    range_params['nl'] = [ (3.0, 10.0), (8.0, 60.0), (8.0, 40.0) ]
    
    learning_rate = 0.05

    discard = 500
    
    xparams = {}
    if restart:
        with open(basename + '.pickle', 'rb') as f:
            best_ubx, best_sbx, best_pbx, best_ex, best_ucx, best_nlx, best_wgx = pickle.load(f)
            xparams['ub'] = best_ubx
            #xparams['ub'][1] = (-2*beta - reference_params['ub'][1])/scale_params['ub'][1]
            xparams['sb'] = best_sbx
            #xparams['sb'][0] = (3.11*beta - reference_params['sb'][0])/scale_params['sb'][0]
            #xparams['sb'][2] = (4.05 - reference_params['sb'][2])/scale_params['sb'][2]
            xparams['pb'] = best_pbx
            #xparams['pb'][0]  = (8.72e-10 - reference_params['pb'][0])/scale_params['pb'][0]
            #xparams['pb'][1]  = (9.0e-13 - reference_params['pb'][1])/scale_params['pb'][1]
            xparams['elj']   = best_ex
            #xparams['elj'][0]   = (8.0*beta - reference_params['elj'][0])/scale_params['elj'][0]
            #xparams['elj'][2]   = (15.0*beta - reference_params['elj'][2])/scale_params['elj'][2]
            xparams['uce']  = best_ucx
            #xparams['uce'][0]  = (1.0 - reference_params['uce'][0])/scale_params['uce'][0]
            #xparams['uce'][2]  = (14.0 - reference_params['uce'][2])/scale_params['uce'][2]
            xparams['nl']  = best_nlx
            #xparams['nl'][1]  = (5.0 - reference_params['nl'][1])/scale_params['nl'][1]
            xparams['wg'] = best_wgx
            #xparams['wg'][0] = (0 - reference_params['wg'][0])/scale_params['wg'][0] 
            #xparams['wg'][1] = (0 - reference_params['wg'][1])/scale_params['wg'][1] 
            #xparams['wg'][2] = (3.42e-34 - reference_params['wg'][2])/scale_params['wg'][2] 
    else:
        xparams['ub']  = [0. for i in range(nmodes) ]
        xparams['sb']  = [0. for i in range(nmodes) ]
        xparams['pb']  = [0. for i in range(nmodes) ]
        xparams['elj']   = [0. for i in range(nmodes) ]
        xparams['uce']  = [0. for i in range(nmodes) ]
        xparams['nl']  = [0. for i in range(nmodes) ]
        xparams['wg']  = [0. for i in range(nmodes) ]

    fe_optimizer = femodel(sdm_data, reference_params, temperature, range_params=range_params,
                  xparams=xparams, scale_params=scale_params, discard=discard, learning_rate=learning_rate)
                                        
    variables = [ fe_optimizer.ubx_t, fe_optimizer.sbx_t , fe_optimizer.ex_t, fe_optimizer.ucx_t, fe_optimizer.nlx_t, fe_optimizer.wgx_t, fe_optimizer.pbx_t ]
    
    #----- test area ----------------

    tf.logging.set_verbosity(tf.logging.ERROR)
    
    if testarea:
        with tf.Session() as sess:
            sess.run(fe_optimizer.init)
            ll = sess.run(fe_optimizer.cost)
            uv = sess.run(fe_optimizer.u)
            uscv = sess.run(fe_optimizer.usc)
            p0v = sess.run(tf.squeeze(fe_optimizer.p0))
            p0scv = sess.run(tf.squeeze(fe_optimizer.p0sc))
            pklv = sess.run(tf.squeeze(fe_optimizer.pkl))
            xscv = sess.run(fe_optimizer.xsc)
            p0scxv = sess.run(tf.squeeze(fe_optimizer.p0scx))
            sts = sess.run(fe_optimizer.lambdas['stateID'])
            upscv = sess.run(fe_optimizer.upsc)
            klv = sess.run(tf.squeeze(fe_optimizer.kl))


            best_ubx = sess.run(fe_optimizer.ubx_t)
            best_sbx = sess.run(fe_optimizer.sbx_t)
            best_pbx = sess.run(fe_optimizer.pbx_t)
            best_ex  = sess.run(fe_optimizer.ex_t)
            best_ucx = sess.run(fe_optimizer.ucx_t)
            best_nlx = sess.run(fe_optimizer.nlx_t)
            best_wgx = sess.run(fe_optimizer.wgx_t)
                        
            best_ub = sess.run(fe_optimizer.ub_t)
            best_sb = sess.run(fe_optimizer.sb_t)
            best_pb = sess.run(fe_optimizer.pb_t)
            best_elj  = sess.run(fe_optimizer.elj_t)
            best_uce = sess.run(fe_optimizer.uce_t)
            best_nl = sess.run(fe_optimizer.nl_t)
            best_wg = sess.run(fe_optimizer.wg_t)
            
            print("Optimized Cost =", ll)
            print("Parameters:")
            print("wg = ", best_wg)
            results = fe_optimizer.applyunits(best_ub, best_sb, best_pb, best_elj, best_uce, best_nl)
            params = ["ub", "sb", "pb", "elj", "uce", "nl"]
            for mode, item in enumerate(results):
                string = ""
                for value in item:
                    string = string + str(value) + " "
                print(params[mode] + " = [" + string + "]")
            print("-------------------------------------------------------------------------")


        # plot
        import matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(states) + 1)))
        for state in states:
            mask = abs(sts - state) < 1.e-6
            hist, bin_edges = np.histogram(uscv[mask]*kT, bins=30, density=True)
            nh = len(hist)
            dx = bin_edges[1] - bin_edges[0]
            xp = bin_edges[0:nh] + 0.5*dx
            c = next(color)
            ax.plot(xp, hist, 'o', markersize = 2, c=c)
            ax.plot(uscv[mask]*kT, pklv[mask]/kT, '+', markersize = 1, c=c)
            #ax.set_xlim([-40*kT,200*kT])
        plt.show()




    #------- training area ----------------------

    if production:
        with tf.Session() as sess:
    
            sess.run(fe_optimizer.init)
            fe_optimizer.opt = fe_optimizer.optimizer.minimize(fe_optimizer.cost, var_list = variables)
            sess.graph.finalize()
            ll = sess.run(fe_optimizer.cost)

            best_loss = ll
            best_ubx = sess.run(fe_optimizer.ubx_t)
            best_sbx = sess.run(fe_optimizer.sbx_t)
            best_pbx = sess.run(fe_optimizer.pbx_t)
            best_ex  = sess.run(fe_optimizer.ex_t)
            best_ucx = sess.run(fe_optimizer.ucx_t)
            best_nlx = sess.run(fe_optimizer.nlx_t)
            best_wgx = sess.run(fe_optimizer.wgx_t)
            
            best_ub = sess.run(fe_optimizer.ub_t)
            best_sb = sess.run(fe_optimizer.sb_t)
            best_pb = sess.run(fe_optimizer.pb_t)
            best_elj  = sess.run(fe_optimizer.elj_t)
            best_uce = sess.run(fe_optimizer.uce_t)
            best_nl = sess.run(fe_optimizer.nl_t)
            best_wg = sess.run(fe_optimizer.wg_t)
            
            gubx = sess.run(fe_optimizer.gubx_t)
            gsbx = sess.run(fe_optimizer.gsbx_t)
            gpbx = sess.run(fe_optimizer.gpbx_t)
            gex = sess.run(fe_optimizer.gex_t)
            gucx = sess.run(fe_optimizer.gucx_t)
            gnlx = sess.run(fe_optimizer.gnlx_t)
            gwgx = sess.run(fe_optimizer.gwgx_t)
            print("Gradients:")
            print(gubx,gsbx,gpbx,gex,gucx,gnlx,gwgx)
            
            print("Start Cost =", ll)
            print("Parameters:")
            print("wg = ", best_wg)

            results = fe_optimizer.applyunits(best_ub, best_sb, best_pb, best_elj, best_uce, best_nl)
            params = ["ub", "sb", "pb", "elj", "uce", "nl"]
            for mode, item in enumerate(results):
                string = ""
                for value in item:
                    string = string + str(value) + " "
                print(params[mode] + " = [" + string + "]")

            print("---------------------------------------")
        
            for step in range(ncycles):
                for i in range(nsteps):
                    #sess.run(fe_optimizer.train) #all variables optimized
                    sess.run(fe_optimizer.opt) #only selected variables are optimized
                gubx = sess.run(fe_optimizer.gubx_t)
                gsbx = sess.run(fe_optimizer.gsbx_t)
                gpbx = sess.run(fe_optimizer.gpbx_t)
                gex = sess.run(fe_optimizer.gex_t)
                gucx = sess.run(fe_optimizer.gucx_t)
                gnlx = sess.run(fe_optimizer.gnlx_t)
                gwgx = sess.run(fe_optimizer.gwgx_t)
                notok = ( np.any(np.isnan(gubx)) or
                          np.any(np.isnan(gsbx)) or
                          np.any(np.isnan(gpbx)) or
                          np.any(np.isnan(gex))  or
                          np.any(np.isnan(gucx)) or
                          np.any(np.isnan(gnlx)) or
                          np.any(np.isnan(gwgx)) )
                

                print("Gradients:")
                print(gubx,gsbx,gpbx,gex,gucx,gnlx,gwgx)
                if notok:
                    print("Gradient error")
                    break
                ll = sess.run(fe_optimizer.cost)

                l_ubx = sess.run(fe_optimizer.ubx_t)
                l_sbx = sess.run(fe_optimizer.sbx_t)
                l_pbx = sess.run(fe_optimizer.pbx_t)
                l_ex  = sess.run(fe_optimizer.ex_t)
                l_ucx = sess.run(fe_optimizer.ucx_t)
                l_nlx = sess.run(fe_optimizer.nlx_t)
                l_wgx = sess.run(fe_optimizer.wgx_t)
            
                l_ub = sess.run(fe_optimizer.ub_t)
                l_sb = sess.run(fe_optimizer.sb_t)
                l_pb = sess.run(fe_optimizer.pb_t)
                l_elj = sess.run(fe_optimizer.elj_t)
                l_uce = sess.run(fe_optimizer.uce_t)
                l_nl = sess.run(fe_optimizer.nl_t)
                l_wg = sess.run(fe_optimizer.wg_t)
                
                if( ll < best_loss ):
                    best_loss = ll
                
                    best_ubx = l_ubx 
                    best_sbx = l_sbx
                    best_pbx = l_pbx
                    best_ex  = l_ex 
                    best_ucx = l_ucx
                    best_nlx = l_nlx 
                    best_wgx = l_wgx

                    best_ub  = l_ub  
                    best_sb  = l_sb 
                    best_pb  = l_pb 
                    best_elj = l_elj
                    best_uce = l_uce
                    best_nl  = l_nl   
                    best_wg  = l_wg
    

                print(step, "Cost =", ll)
                print("Parameters:")
                print("wg = ", l_wg)
                results = fe_optimizer.applyunits(l_ub, l_sb, l_pb, l_elj, l_uce, l_nl)
                params = ["ub", "sb", "pb", "elj", "uce", "nl"]
                for mode, item in enumerate(results):
                    string = ""
                    for value in item:
                        string = string + str(value) + " "
                    print(params[mode] + " = [" + string + "]") 

                print("----------------------------------------------------")

                with open(basename + '.pickle', 'wb') as f:
                    pickle.dump([best_ubx, best_sbx, best_pbx, best_ex, best_ucx, best_nlx, best_wgx],f)


                
            print("----------- End of Optimization -----------", basename);
            print("Optimized Cost =", best_loss)
            print("wg = ", best_wg)
            results = fe_optimizer.applyunits(best_ub, best_sb, best_pb, best_elj, best_uce, best_nl)
            params = ["ub", "sb", "pb", "elj", "uce", "nl"]
            for mode, item in enumerate(results):
                string = ""
                for value in item:
                    string = string + str(value) + " "
                print(params[mode] + " = [" + string + "]")



