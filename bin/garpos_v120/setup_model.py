"""
Created:
    07/01/2020 by S. Watanabe
Modified:
    01/07/2022 by S. Watanabe
        to use cholesky decomposition for calc. inverse
    07/01/2024 by S. Watanabe
        to apply a mode for "array take-over"
        (to solve each position and parallel disp. simultaneously)
    07/18/2026
        remove sksparse dependency (use scipy.sparse.linalg.splu)
Contains:
    init_position
    make_knots
    first_derivative
    second_derivative
    data_correlation
"""
import sys
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, linalg
from scipy.sparse.linalg import splu


def init_position(cfg, MTs):
    """
    Calculate Jacobian matrix for positions.

    Parameters
    ----------
    cfg : configparser
        Config file for site paramters.
    MTs : list
        List of transponders' name.

    Returns
    -------
    mp : ndarray
        complete model parameter vector. (only for position)
    Dipos : csc_matrix
        A priori covariance for position.
    slvidx0 : list
        Indices of model parameters to be solved. (only for position)
    mtidx : dictionary
        Indices of mp for each MT.
    """

    mtidx = {}
    mp = np.array([])
    ae = np.array([])
    for imt, mt in enumerate(MTs):
        mtidx[mt] = imt * 3
        dpos = cfg.get("Model-parameter", mt + "_dPos").split()
        dpos = list(map(float, dpos))
        mp = np.append(mp, dpos[0:3])
        ae = np.append(ae, dpos[3:6])

    dcnt = cfg.get("Model-parameter", "dCentPos").split()
    dcnt = list(map(float, dcnt))
    mp = np.append(mp, dcnt[0:3])
    ae = np.append(ae, dcnt[3:6])
    if len(dcnt) <= 6:
        covNU = 0.0
        covUE = 0.0
        covEN = 0.0
    else:
        covNU = dcnt[6]
        covUE = dcnt[7]
        covEN = dcnt[8]
    ae3 = ae.reshape(int(len(ae)/3),3)
    aee = np.count_nonzero(ae3[:,0] > 1.e-8)
    aen = np.count_nonzero(ae3[:,1] > 1.e-8)
    aeu = np.count_nonzero(ae3[:,2] > 1.e-8)

    if aee > len(MTs) or aen > len(MTs) or aeu > len(MTs):
        print("Error: positions to be solved must be smaller than the number of MTs!")
        sys.exit(1)

    atd = cfg.get("Model-parameter", "ATDoffset").split()
    atd = list(map(float, atd))
    mp = np.append(mp, atd[0:3])
    if atd[3] > 1.e-8:
        ae = np.append(ae, 3.0)
    else:
        ae = np.append(ae, 0.0)
    if atd[4] > 1.e-8:
        ae = np.append(ae, 3.0)
    else:
        ae = np.append(ae, 0.0)
    if atd[5] > 1.e-8:
        ae = np.append(ae, 3.0)
    else:
        ae = np.append(ae, 0.0)

    # set a priori variance for position parameters
    D0pos = lil_matrix(np.diag( ae**2. ))
    # set a priori covariance for dCentPos
    D0pos[len(MTs)*3+1,len(MTs)*3+2] = covNU
    D0pos[len(MTs)*3+2,len(MTs)*3+0] = covUE
    D0pos[len(MTs)*3+0,len(MTs)*3+1] = covEN
    D0pos[len(MTs)*3+2,len(MTs)*3+1] = covNU
    D0pos[len(MTs)*3+0,len(MTs)*3+2] = covUE
    D0pos[len(MTs)*3+1,len(MTs)*3+0] = covEN

    slvidx0 = np.where( ae > 1.e-14 )[0]
    nmppos = len(slvidx0)
    Dpos = lil_matrix( (nmppos, nmppos) )
    for i, ipos in enumerate(slvidx0):
        for j, jpos in enumerate(slvidx0):
            Dpos[i, j] = D0pos[ipos,jpos]
    Dpos = Dpos.tocsc()
    if nmppos == 0:
        Dipos = Dpos
    else:
        Dipos = linalg.inv( Dpos )

    return mp, Dipos, slvidx0, mtidx


def make_knots(shotdat, spdeg, knotintervals):
    """
    Create the B-spline knots for correction value "gamma".

    Parameters
    ----------
    shotdat : DataFrame
        GNSS-A shot dataset.
    spdeg : int
        spline degree (=3).
    knotintervals : list of int (len=5)
        approximate knot intervals.

    Returns
    -------
    knots : list of ndarray (len=5)
        B-spline knots for each component in "gamma".
    """

    sets = shotdat['SET'].unique()
    st0s = np.array([shotdat.loc[shotdat.SET==s, "ST"].min() for s in sets])
    stfs = np.array([shotdat.loc[shotdat.SET==s, "RT"].max() for s in sets])

    st0 = shotdat.ST.values.min()
    stf = shotdat.RT.values.max()
    obsdur = stf - st0

    nknots = [ int(obsdur/knint) if knint!=0 else 0 for knint in knotintervals ]
    knots = [ np.linspace(st0, stf, nall+1) for nall in nknots ]
    dknots = []

    for k, cn in enumerate(knots):

        if nknots[k] == 0:
            knots[k] = np.array([])
            continue

        rmknot = np.array([])
        for i in range(len(sets)-1):
            isetkn = np.where( (knots[k]>stfs[i]) & (knots[k]<st0s[i+1]) )[0]
            if len(isetkn) > 2*(spdeg+2):
                rmknot = np.append(rmknot, isetkn[spdeg+1:-spdeg-1])
        rmknot = rmknot.astype(int)
        if len(rmknot) > 0:
            knots[k] = np.delete(knots[k], rmknot)

        dkn = (stf-st0)/float(nknots[k])
        addkn0 = np.array( [st0-dkn*(n+1) for n in reversed(range(spdeg))] )
        addknf = np.array( [stf+dkn*(n+1) for n in range(spdeg)] )
        knots[k] = np.append(addkn0, knots[k])
        knots[k] = np.append(knots[k], addknf)

    return knots


def first_derivative(knot_vec, spdeg):
    """
    Calculate the constraint matrix for 1st derivative of the B-spline basis

    Parameters
    ----------
    knot_vec : ndarray
        B-spline knot vector.
    spdeg : int
        spline degree (=3).

    Returns
    -------
    H1 : ndarray
        1st derivative constraint matrix of the B-spline basis.
    """
    m = len(knot_vec)-spdeg-1
    if m <= 0:
        return False
    H1 = lil_matrix( (m, m) )
    d_knot_vec = np.diff(knot_vec)
    dknot = d_knot_vec[0]
    icutknot = np.where( d_knot_vec > dknot*2. )
    icutknot = np.append(icutknot, m)
    ic0 = 0

    scl0 = 60.*10.
    scale = (dknot/scl0)**(-1.) / 120.
    for mm in range(0, m-spdeg-1):
        H1[mm,mm] = 80.
        H1[mm,mm+1] = -15.
        H1[mm+1,mm] = -15.
        H1[mm,mm+2] = -24.
        H1[mm+2,mm] = -24.
        H1[mm,mm+3] = -1.
        H1[mm+3,mm] = -1.

    H1_ul = np.array([[  6.,   7., -12.,  -1.],
                      [  7.,  40., -22., -24.],
                      [-12., -22.,  74., -15.],
                      [ -1., -24., -15.,  80.]])
    H1_dr = H1_ul[::-1, ::-1]

    for ic1 in icutknot:
        H1[ic0:ic0+4,ic0:ic0+4] = H1_ul
        H1[ic1-4:ic1,ic1-4:ic1] = H1_dr
        if ic1 != m:
            H1[ic1-4:ic1,ic1:] = 0.
            H1[ic1:,ic1-4:ic1] = 0.
        ic0 = ic1

    H1 = H1.tocsc() * scale

    return H1


def second_derivative(knot_vec, spdeg):
    """
    Calculate the constraint matrix for 2nd derivative of the B-spline basis

    Parameters
    ----------
    knot_vec : ndarray
        B-spline knot vector.
    spdeg : int
        spline degree (=3).

    Returns
    -------
    H2 : ndarray
        2nd derivative constraint matrix of the B-spline basis.
    """
    m = len(knot_vec)-spdeg-1
    if m <= 0:
        return False
    H2 = lil_matrix( (m, m) )
    d_knot_vec = np.diff(knot_vec)
    dknot = d_knot_vec[0]
    icutknot = np.where( d_knot_vec > dknot*2. )
    icutknot = np.append(icutknot, m)
    ic0 = 0

    scl0 = 60.*10.
    scale = (dknot/scl0)**(-3.) / 6.
    for mm in range(0, m-spdeg-1):
        H2[mm,mm] = 16.
        H2[mm,mm+1] = -9.
        H2[mm+1,mm] = -9.
        H2[mm,mm+3] = 1.
        H2[mm+3,mm] = 1.

    H2_ul = np.array([[ 2., -3.,  0.,  1.],
                      [-3.,  8., -6.,  0.],
                      [ 0., -6., 14., -9.],
                      [ 1.,  0., -9., 16.]])
    H2_dr = H2_ul[::-1, ::-1]

    for ic1 in icutknot:
        H2[ic0:ic0+4,ic0:ic0+4] = H2_ul
        H2[ic1-4:ic1,ic1-4:ic1] = H2_dr
        if ic1 != m:
            H2[ic1-4:ic1,ic1:] = 0.
            H2[ic1:,ic1-4:ic1] = 0.
        ic0 = ic1

    H2 = H2.tocsc() * scale

    return H2


def data_correlation(shotdat, TT0, mu_t, mu_m):
    """
    Calculate the covariance matrix for data.

    Parameters
    ----------
    shotdat : DataFrame
        GNSS-A shot dataset.
    TT0 : ndarray (len=ndata)
        Vector of (travel time) / (characteristic travel time).
    mu_t : float
        Correlation length (in sec.).
    mu_m : float
        Ratio of correlation between the different transponders.

    Returns
    -------
    E_factor : SuperLU factor
        LU factorization of E (for solve).
    """

    ndata = shotdat.index.size
    sts = shotdat.ST.values
    mtids = shotdat.mtid.values
    negativedST = shotdat[ (shotdat.ST.diff(1) == 0.) & (shotdat.mtid.diff(1) ==0.) ]
    if len(negativedST) > 0:
        print(negativedST.index)
        print("error in data_correlation 'Negative d-ST'; see setup_model.py")
        sys.exit(1)

    E = lil_matrix( (ndata, ndata) )
    for i, (iMT, iST) in enumerate(zip( mtids, sts )):
        idx = shotdat[ ( abs(sts - iST) < mu_t * 4.)].index
        dshot = np.abs(iST - sts[idx])/mu_t
        dcorr = np.exp(-dshot) * (mu_m + (1.-mu_m)*(iMT==mtids[idx]))
        E[i,idx] = dcorr / TT0[i] / TT0[idx]
    E = E.tocsc()

    # LU decomposition (scipy.sparse.linalg.splu, sksparse-free)
    E_factor = splu(E)

    return E_factor
