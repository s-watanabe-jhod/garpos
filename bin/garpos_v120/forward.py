"""
Created:
    07/01/2020 by S. Watanabe
Modified:
    02/13/2025 to add azimuth
    07/18/2026 add calc_jcb_gamma (BSpline.design_matrix vectorized Jacobian)
    07/18/2026 batch Fortran calls in jacobian_pos
Contains:
    calc_forward
    calc_gamma
    calc_jcb_gamma
    jacobian_pos
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix

# garpos module
from .coordinate_trans import corr_attitude
from .traveltime import calc_traveltime, calc_traveltime_batch


def calc_forward(shots, mp, nMT, icfg, svp, T0):
    """
    Calculate the forward modeling of observation eqs.

    Parameters
    ----------
    shots : DataFrame
        GNSS-A shot dataset.
    mp : ndarray
        complete model parameter vector.
    nMT : int
        number of transponders.
    icfg : configparser
        Config file for inversion conditions.
    svp : DataFrame
        Sound speed profile.
    T0 : float
        Typical travel time.

    Returns
    -------
    shots : DataFrame
        GNSS-A shot dataset in which calculated data is added.
    """

    rsig = float(icfg.get("Inv-parameter","RejectCriteria"))

    # calc ATD offset
    pl0 = mp[(nMT+1)*3+0]
    pl1 = mp[(nMT+1)*3+1]
    pl2 = mp[(nMT+1)*3+2]
    hd0 = shots.head0.values
    hd1 = shots.head1.values
    rl0 = shots.roll0.values
    rl1 = shots.roll1.values
    pc0 = shots.pitch0.values
    pc1 = shots.pitch1.values
    ple0, pln0, plu0 = corr_attitude(pl0, pl1, pl2, hd0, rl0, pc0)
    ple1, pln1, plu1 = corr_attitude(pl0, pl1, pl2, hd1, rl1, pc1)
    shots['ple0'] = ple0
    shots['pln0'] = pln0
    shots['plu0'] = plu0
    shots['ple1'] = ple1
    shots['pln1'] = pln1
    shots['plu1'] = plu1

    # calc Residuals
    cTT, cTO, azimuth = calc_traveltime(shots, mp, nMT, icfg, svp)
    logTTc = np.log( cTT/T0 ) - shots.gamma.values
    ResiTT = shots.logTT.values - logTTc

    shots['calcTT'] = cTT
    shots['TakeOff'] = cTO
    shots['Azimuth'] = azimuth
    shots['logTTc'] = logTTc
    shots['ResiTT'] = ResiTT
    # approximation log(1 + x) ~ x
    shots['ResiTTreal'] = ResiTT * shots.TT.values

    if rsig > 0.1:
        aveRTT = shots[~shots['flag']].ResiTT.mean()
        sigRTT = shots[~shots['flag']].ResiTT.std()
        th0 = aveRTT + rsig * sigRTT
        th1 = aveRTT - rsig * sigRTT
        shots['flag'] = (shots['ResiTT'] > th0) | (shots['ResiTT'] < th1) | shots['iniflag']
        aveRTT1 = shots[~shots['flag']].ResiTT.mean()
        sigRTT1 = shots[~shots['flag']].ResiTT.std()

    return shots


def calc_gamma(mp, shotdat, imp0, spdeg, knots):
    """
    Calculate correction value "gamma" in the observation eqs.

    Parameters
    ----------
    mp : ndarray
        complete model parameter vector.
    shotdat : DataFrame
        GNSS-A shot dataset.
    imp0 : ndarray (len=5)
        Indices where the type of model parameters change.
    spdeg : int
        spline degree (=3).
    knots : list of ndarray (len=5)
        B-spline knots for each component in "gamma".

    Returns
    -------
    gamma : ndarray
        Values of "gamma". Note that scale facter is not applied.
    a : 2-d list of ndarray
        [a0[<alpha>], a1[<alpha>]] :: a[<alpha>] at transmit/received time.
        <alpha> is corresponding to <0>, <1E>, <1N>, <2E>, <2N>.
    """

    a0 = []
    a1 = []
    for k, kn in enumerate(knots):
        if len(kn) == 0:
            a0.append( np.zeros(len(shotdat.ST)) )
            a1.append( np.zeros(len(shotdat.RT)) )
            continue
        ct = mp[imp0[k]:imp0[k+1]]
        bs = BSpline(kn, ct, spdeg, extrapolate=False)
        a0.append( bs(shotdat.ST.values) )
        a1.append( bs(shotdat.RT.values) )

    ls = 1000.  # m/s/m to m/s/km order for gradient

    de0 = shotdat.de0.values
    de1 = shotdat.de1.values
    dn0 = shotdat.dn0.values
    dn1 = shotdat.dn1.values
    mte = shotdat.mtde.values
    mtn = shotdat.mtdn.values

    gamma0_0 =  a0[0]
    gamma0_1 = (a0[1] * de0 + a0[2] * dn0) / ls
    gamma0_2 = (a0[3] * mte + a0[4] * mtn) / ls

    gamma1_0 =  a1[0]
    gamma1_1 = (a1[1] * de1 + a1[2] * dn1) / ls
    gamma1_2 = (a1[3] * mte + a1[4] * mtn) / ls

    gamma0 = gamma0_0 + gamma0_1 + gamma0_2
    gamma1 = gamma1_0 + gamma1_1 + gamma1_2

    gamma = (gamma0 + gamma1)/2.
    a = [a0, a1]

    return gamma, a


def calc_jcb_gamma(shotdat, imp0, spdeg, knots, scale, nmppos):
    """
    Build Jacobian rows for gamma B-spline coefficients using design matrices.

    Parameters
    ----------
    shotdat : DataFrame
        GNSS-A shot dataset (unrejected subset).
    imp0 : ndarray
        Indices where the type of model parameters change.
    spdeg : int
        Spline degree (=3).
    knots : list of ndarray
        B-spline knots.
    scale : float
        Travel time scale factor.
    nmppos : int
        Number of position parameters.

    Returns
    -------
    jcb_gamma : ndarray, shape (n_gamma_params, ndata)
    """
    ls = 1000.
    ndata = shotdat.index.size
    n_gamma = imp0[-1] - imp0[0]
    jcb_gamma = np.zeros((n_gamma, ndata))

    de0 = shotdat.de0.values
    de1 = shotdat.de1.values
    dn0 = shotdat.dn0.values
    dn1 = shotdat.dn1.values
    mte = shotdat.mtde.values
    mtn = shotdat.mtdn.values

    imp = 0
    for k, kn in enumerate(knots):
        if len(kn) == 0:
            continue
        n_coef = imp0[k+1] - imp0[k]
        B0 = BSpline.design_matrix(shotdat.ST.values, kn, spdeg).toarray()
        B1 = BSpline.design_matrix(shotdat.RT.values, kn, spdeg).toarray()

        if k == 0:
            gpc = (B0 + B1) / 2.0
        elif k == 1:
            gpc = (B0 * de0[:, None] + B1 * de1[:, None]) / (2. * ls)
        elif k == 2:
            gpc = (B0 * dn0[:, None] + B1 * dn1[:, None]) / (2. * ls)
        elif k == 3:
            gpc = (B0 + B1) * mte[:, None] / (2. * ls)
        elif k == 4:
            gpc = (B0 + B1) * mtn[:, None] / (2. * ls)

        jcb_gamma[imp:imp + n_coef, :] = -gpc.T * scale
        imp += n_coef

    return jcb_gamma


def jacobian_pos(icfg, mp, slvidx0, shotdat, mtidx, svp, T0):
    """
    Calculate Jacobian matrix for positions.

    Parameters
    ----------
    icfg : configparser
        Config file for inversion conditions.
    mp : ndarray
        complete model parameter vector.
    slvidx0 : list
        Indices of model parameters to be solved.
    shotdat : DataFrame
        GNSS-A shot dataset.
    mtidx : dictionary
        Indices of mp for each MT.
    svp : DataFrame
        Sound speed profile.
    T0 : float
        Typical travel time.

    Returns
    -------
    jcbpos : lil_matrix
        Jacobian matrix for positions.
    """

    # read inversion parameters
    deltap = float(icfg.get("Inv-parameter","deltap"))
    deltab = float(icfg.get("Inv-parameter","deltab"))

    ndata = shotdat.index.size

    MTs = mtidx.keys()
    nMT = len(MTs)
    nmppos = len(slvidx0)

    jcbpos  = lil_matrix( (nmppos, ndata) )
    imp = 0

    gamma = shotdat.gamma.values
    logTTc = shotdat.logTTc.values

    ##################################
    ### Calc Jacobian for Position ###
    ### Batch: center E/N/U        ###
    ##################################
    shotdat_list = []
    mp_list = []
    for j in range(3):
        mpj = mp.copy()
        mpj[nMT*3 + j] += deltap
        shotdat_list.append(shotdat)
        mp_list.append(mpj)

    ####################################
    ### Calc Jacobian for ATD offset ###
    ### Batch: ATD perturbations     ###
    ####################################
    atd_indices = []
    for j in range(3):
        idx = nMT*3 + 3 + j
        if not (idx in slvidx0):
            continue
        atd_indices.append(j)
        mpj = mp.copy()
        mpj[(nMT+1)*3 + j] += deltap
        tmpj = shotdat.copy()

        pl0 = mpj[(nMT+1)*3 + 0]
        pl1 = mpj[(nMT+1)*3 + 1]
        pl2 = mpj[(nMT+1)*3 + 2]
        hd0 = shotdat.head0.values
        hd1 = shotdat.head1.values
        rl0 = shotdat.roll0.values
        rl1 = shotdat.roll1.values
        pc0 = shotdat.pitch0.values
        pc1 = shotdat.pitch1.values
        ple0, pln0, plu0 = corr_attitude(pl0, pl1, pl2, hd0, rl0, pc0)
        ple1, pln1, plu1 = corr_attitude(pl0, pl1, pl2, hd1, rl1, pc1)
        tmpj['ple0'] = ple0
        tmpj['pln0'] = pln0
        tmpj['plu0'] = plu0
        tmpj['ple1'] = ple1
        tmpj['pln1'] = pln1
        tmpj['plu1'] = plu1

        shotdat_list.append(tmpj)
        mp_list.append(mpj)

    # Single batched Fortran call for all perturbations
    batch_results = calc_traveltime_batch(shotdat_list, mp_list, nMT, icfg, svp)

    # Extract center E/N/U results
    for j in range(3):
        cTTj, cTOj, azimuthj = batch_results[j]
        logTTcj = np.log(cTTj / T0) - gamma
        shotdat['jacob%1d' % j] = (logTTcj - logTTc) / deltap

    ### Jacobian for each MT ###
    for mt in MTs:
        for j in range(3):
            idx = mtidx[mt] + j
            if not (idx in slvidx0):
                continue
            jccode = "jacob%1d" % j
            shotdat['hit'] = shotdat[jccode] * (shotdat['MT'] == mt)
            jcbpos[imp,:] = np.array([shotdat.hit.values])
            imp += 1

    ### Jacobian for Center Pos ###
    for j in range(3):
        idx = nMT*3 + j
        if not (idx in slvidx0):
            continue
        jccode = "jacob%1d" % j
        jcbpos[imp,:] = shotdat[jccode].values
        imp += 1

    ### Extract ATD Jacobian from batch results ###
    for i, j in enumerate(atd_indices):
        cTTj, cTOj, azimuthj = batch_results[3 + i]
        logTTcj = np.log(cTTj / T0) - gamma
        jcbpos[imp,:] = (logTTcj - logTTc) / deltap
        imp += 1

    return jcbpos
