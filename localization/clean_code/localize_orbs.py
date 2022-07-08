from tools import *
import numpy as np
import scipy
import time

def evaluate_obj_func_and_grad_for_quick_localize(Cold, S, AO_indices, x):
    """Evaluates and returns the objective function value and gradient for our LBFGS quick localize method.

        Cold: The old orbital coefficients for the molecular orbitals we are trying to localize.

        S: The overlap matrix of the molecule.

        AO_indices: List of lists of AO incidices, each associated with an atom

        x: The elements of the upper triangle of the antisymmetric X from which the rotation matrix is built by exp(X) 
    """
    
    norb = Cold.shape[0] # Number of AOs
    ntl = Cold.shape[1] # Number of MOs we are localizing
    Natom = len(AO_indices)

    # Get the unitary rotation matrix R
    R = unitary_matrix_from_x(ntl, x)
    R_conj = R.conj().copy()
    print("Current deviation from identity = ", np.amax(np.abs(R-np.identity(ntl))) )


    # Get product of original mo coeffs and overlap matrix
    SCold = S @ Cold

    # Evaluate Q
    Q_prep = [np.zeros([ntl,ntl]) for i in range(Natom)]

    for Q,A in zip(Q_prep,AO_indices):
      for r in range(ntl):
        for s in range(ntl):
          for u in A:
            Q[r,s] += 0.5 * (Cold[u,r] * SCold[u,s] + Cold[u,s] * SCold[u,r])

    Q_list = [np.zeros([ntl,ntl]) for i in range(Natom)]
    dQdR_list = [np.zeros([ntl,ntl]) for i in range(Natom)]
    for Q,Qa,dQdR in zip(Q_list,Q_prep,dQdR_list):
        for j in range(ntl):
            for s in range(ntl):
                for r in range(ntl):
                    dQdR[s,j] += Qa[r,s] * R_conj[r,j]
                    Q[j,j] += R_conj[r,j] * Qa[r,s] * R[s,j]


    # Evaluate the objective function value
    ofv = -1.0 * np.sum([np.square(Q) for Q in Q_list])
    print("objective function = %20.12f" % -ofv, flush=True)
    print("")
  
    dfdR = np.zeros([ntl,ntl])

    for i in range(ntl):
      for j in range(ntl):
        for Q,dQdR in zip(Q_list,dQdR_list):
          dfdR[i,j] += -2*Q[j,j] * dQdR[i,j]
   

    # Evaluate the gradient with respect to x
    dx = unitary_matrix_deriv(ntl, x, dfdR)
    
    return ofv, dx

def quick_localize(Cold, S,orbLists, startNoise=0.0, returnR=False, aoGuess=False,tolerance=None):
  """Performs a quick localization on the supplied orbitals and returns the MO coeff matrix or rotation matrix for the localized orbitals.

     Cstart: Matrix containing the orbital coefficients to start the localization from.
             Each column of the matrix (e.g. Cstart[:,3]) holds the coeffs for one orbital.
             The final local orbitals will be orthonormal linear combiantions of
             these starting orbitals.

     orbLists: List of lists of AO indices.  Each list holds the AO indices used to define one atom.

     startNoise: amount of initial noise to add to the guess for the rotation matrix

     returnR: if True, return the orthogonal rotation matrix R instead of the new mo matrix C

     aoGuess: if True, uses projected atomic orbitals to form a guess for the local orbitals

  """

  entry_time = time.time()

  print("")
  print("###########################################")
  print("###### Performing Quick Localization ######")
  print("###########################################")

  # ensure that the input mo coefficients is actually a numpy array and not some funny array view or some such
  Cs = 1.0 * Cold

  # get number of atomic orbitals
  norb = Cs.shape[0]

  # get number of molecular orbitals we are localizing
  ntl = Cs.shape[1]
  print("")
  print("localizing %i molecular orbitals" % ntl)

  # initial rotation is the identity
  R = np.eye(ntl)

  # load the ao overlap matrix
  ovl_ao = np.zeros([norb,norb])
  #load_ovls(norb, ovl_ao)

  # if requested, use projected AOs to set up the initial guess
  if aoGuess:

    # get the occ-occ RHF-ESMF MO overlaps
    S_mo_ao = np.transpose(Cs) @ ovl_ao

    # get a list of the indices of the AOs that are best representable by the input orbitals
    best_aos = [ x[1] for x in sorted( [ ( -1.0 * np.linalg.norm(S_mo_ao[:,i]) ** 2.0, i ) for i in range(norb) ] )[:ntl] ]
    print("")
    print("best AOs for guess:")
    print("")
    print(best_aos)

    # approximately reproduce these AOs as linear combinations of the input orbitals
    aoR = np.concatenate( [ 1.0 * S_mo_ao[:,i:i+1] for i in best_aos ], axis=1 )

    # Orthogonalize the approximate AOs using Lowdin orthogonalization so as to change them as little as possible.
    S_temp = aoR.T @ Cs.T @ ovl_ao @ Cs @ aoR
    np.set_printoptions(linewidth=100000, formatter={'float':lambda x: "%12.6f" % x})
    #print("")
    #print("S_temp:")
    #print("")
    #print(S_temp)
    #print("")
    #print("before orthogalizing best AO approximations, orthonormality error = %.2e" % np.amax( np.abs( np.eye(ntl) - S_temp ) ))
    w, v = np.linalg.eigh(S_temp)
    S_neg_half = v @ np.diag( 1.0 / np.sqrt(w) ) @ v.T
    R = aoR @ S_neg_half

    # Orthogonalize again to improve precision
    S_temp = R.T @ Cs.T @ ovl_ao @ Cs @ R
    w, v = np.linalg.eigh(S_temp)
    S_neg_half = v @ np.diag( 1.0 / np.sqrt(w) ) @ v.T
    R = R @ S_neg_half

    # print orthonormality error
    S_temp = R.T @ Cs.T @ ovl_ao @ Cs @ R
    print("")
    print("after using best AOs for guess, orthonormality error = %.2e" % np.amax( np.abs( np.eye(ntl) - S_temp ) ))

  # use lbfgs to minimize the objective function
  xguess = np.zeros( [ (ntl*(ntl-1))//2 ] )
  xguess = xguess + startNoise * 2.0 * ( np.random.random(xguess.shape) - 0.5 )
  foo = lambda y: evaluate_obj_func_and_grad_for_quick_localize(Cs, S,orbLists , y)
  print("\nStarting x-based LBFGS", flush=True)
  sol = scipy.optimize.minimize(foo, xguess, jac=True, method='L-BFGS-B',tol=tolerance)
  R = R @ unitary_matrix_from_x(Cs.shape[1], sol.x)

  # print the deviation of the rotation matrix from the identity matrix
  print("")
  print("Quick localizization orbital rotation max deviation from identity = %.6f" % np.amax(np.abs(R-np.identity(ntl))) )

  # print the amount of time the localization took
  print("")
  print("Quick localization took %.6f seconds" % ( time.time() - entry_time ))

  # return either the orbital coefficients for the newly localized orbitals or the rotation to get to them
  if returnR:
    return R  # return the rotation matrix used to convert to the localized orbitals
  else:
    return Cs @ R  # return the mo coeffs for the localized orbitals