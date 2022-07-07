from pyscf import scf, gto, lo, molden
from pyscf.lo import PM
import numpy as np
import scipy
import time
from pyscf.gto import mole

def mat_exp_deriv_taylor(M, dfdexpM):
  """Evaluates differentiation through matrix exponentiation via a finite taylor expansion

     When we have some function that depends on the exponential of a matrix M as
     f = f(exp(M)) and the derivatives of f with respect to exp(M) are known,
     this function evaluates and returns the derivatives of f with respect to M.

     Note that this function is not recommended to be used directly.
     Instead, the more accurate mat_exp_deriv can be used, which uses this
     taylor expansion function internally but delivers improved accuracy
     when the matrix M has a significant magnitude.

         M --- the matrix whose exponentiation we are talking about
   dfdexpM --- the derivatives of f with respect to exp(M)

  """

  i10 = np.identity(M.shape[0]) + M       / 10.0
  i9  = np.identity(M.shape[0]) + M @ i10 /  9.0
  i8  = np.identity(M.shape[0]) + M @ i9  /  8.0
  i7  = np.identity(M.shape[0]) + M @ i8  /  7.0
  i6  = np.identity(M.shape[0]) + M @ i7  /  6.0
  i5  = np.identity(M.shape[0]) + M @ i6  /  5.0
  i4  = np.identity(M.shape[0]) + M @ i5  /  4.0
  i3  = np.identity(M.shape[0]) + M @ i4  /  3.0
  i2  = np.identity(M.shape[0]) + M @ i3  /  2.0
  #i1  = np.identity(M.shape[0]) + M @ i2  /  1.0

  # note that exp(M) =  I + M ( I + (1/2) M ( I + (1/3) M (I + ...)  )  )
  #                    |      |             |             |       |  |  |  |
  #                    |      |             |             ---------  |  |  |
  #                    |      |             |                |       |  |  |
  #                    |      |             |               \|/      |  |  |
  #                    |      |             |                i4      |  |  |
  #                    |      |             |                        |  |  |
  #                    |      |             --------------------------  |  |
  #                    |      |                         |               |  |
  #                    |      |                        \|/              |  |
  #                    |      |                         i3              |  |
  #                    |      |                                         |  |
  #                    |      -------------------------------------------  |
  #                    |                         |                         |
  #                    |                        \|/                        |
  #                    |                         i2                        |
  #                    |                                                   |
  #                    -----------------------------------------------------
  #                                      |          
  #                                     \|/         
  #                                      i1         

  dfdi1  = dfdexpM
  dfdi2  = np.transpose(M) @ dfdi1 / 1.0
  dfdi3  = np.transpose(M) @ dfdi2 / 2.0
  dfdi4  = np.transpose(M) @ dfdi3 / 3.0
  dfdi5  = np.transpose(M) @ dfdi4 / 4.0
  dfdi6  = np.transpose(M) @ dfdi5 / 5.0
  dfdi7  = np.transpose(M) @ dfdi6 / 6.0
  dfdi8  = np.transpose(M) @ dfdi7 / 7.0
  dfdi9  = np.transpose(M) @ dfdi8 / 8.0
  dfdi10 = np.transpose(M) @ dfdi9 / 9.0

  dfdM = np.zeros_like(M)

  dfdM = dfdM + dfdi1  @ np.transpose(i2 ) /  1.0
  dfdM = dfdM + dfdi2  @ np.transpose(i3 ) /  2.0
  dfdM = dfdM + dfdi3  @ np.transpose(i4 ) /  3.0
  dfdM = dfdM + dfdi4  @ np.transpose(i5 ) /  4.0
  dfdM = dfdM + dfdi5  @ np.transpose(i6 ) /  5.0
  dfdM = dfdM + dfdi6  @ np.transpose(i7 ) /  6.0
  dfdM = dfdM + dfdi7  @ np.transpose(i8 ) /  7.0
  dfdM = dfdM + dfdi8  @ np.transpose(i9 ) /  8.0
  dfdM = dfdM + dfdi9  @ np.transpose(i10) /  9.0
  dfdM = dfdM + dfdi10                     / 10.0

  return dfdM

def mat_exp_deriv(M, dfdexpM):
  """Evaluates differentiation through matrix exponentiation

     When we have some function that depends on the exponential of a matrix M as
     f = f(exp(M)) and the derivatives of f with respect to exp(M) are known,
     this function evaluates and returns the derivatives of f with respect to M.
     Note that, the larger the magnitude of M, the less accurate this function
     will be.  However, this function should maintain accuracy better than
     the mat_exp_deriv_taylor function.

         M --- the matrix whose exponentiation we are talking about
   dfdexpM --- the derivatives of f with respect to exp(M)

  """

  e2  = False
  e4  = False
  e8  = False
  e16 = True

  # evaluate based on ( exp(M/2) ) ** 2
  if e2:
    M2 = M / 2.0
    expM2 = scipy.linalg.expm(M2)
    dfdexpM2 = dfdexpM @ np.transpose(expM2) + np.transpose(expM2) @ dfdexpM
    dfdM2 = mat_exp_deriv_taylor(M2, dfdexpM2)
    dfdM = dfdM2 / 2.0
    return dfdM

  # evaluate based on ( exp(M/4) ) ** 4
  if e4:
    M4 = M / 4.0
    expM4 = scipy.linalg.expm(M4)
    expM2 = expM4 @ expM4
    dfdexpM2 = dfdexpM  @ np.transpose(expM2) + np.transpose(expM2) @ dfdexpM
    dfdexpM4 = dfdexpM2 @ np.transpose(expM4) + np.transpose(expM4) @ dfdexpM2
    dfdM4 = mat_exp_deriv_taylor(M4, dfdexpM4)
    dfdM = dfdM4 / 4.0
    return dfdM

  # evaluate based on ( exp(M/8) ) ** 8
  if e8:
    M8 = M / 8.0
    expM8 = scipy.linalg.expm(M8)
    expM4 = expM8 @ expM8
    expM2 = expM4 @ expM4
    dfdexpM2 = dfdexpM  @ np.transpose(expM2) + np.transpose(expM2) @ dfdexpM
    dfdexpM4 = dfdexpM2 @ np.transpose(expM4) + np.transpose(expM4) @ dfdexpM2
    dfdexpM8 = dfdexpM4 @ np.transpose(expM8) + np.transpose(expM8) @ dfdexpM4
    dfdM8 = mat_exp_deriv_taylor(M8, dfdexpM8)
    dfdM = dfdM8 / 8.0
    return dfdM

  # evaluate based on ( exp(M/16) ) ** 16
  if e16:
    M16 = M / 16.0
    expM16 = scipy.linalg.expm(M16)
    expM8 = expM16 @ expM16
    expM4 = expM8  @ expM8
    expM2 = expM4  @ expM4
    dfdexpM2  = dfdexpM  @ np.transpose(expM2)  + np.transpose(expM2)  @ dfdexpM
    dfdexpM4  = dfdexpM2 @ np.transpose(expM4)  + np.transpose(expM4)  @ dfdexpM2
    dfdexpM8  = dfdexpM4 @ np.transpose(expM8)  + np.transpose(expM8)  @ dfdexpM4
    dfdexpM16 = dfdexpM8 @ np.transpose(expM16) + np.transpose(expM16) @ dfdexpM8
    dfdM16 = mat_exp_deriv_taylor(M16, dfdexpM16)
    dfdM = dfdM16 / 16.0
    return dfdM

def unitary_matrix_from_x(n, x):
  """Evaluates an n by n unitary matrix using the elements of x.

       The unitary matrix is U = exp(X)
      
       The antisymmetric matrix X has the structure:
      
        X = [  0.0    X_01    X_02    X_03  ...   ]
            [ -X_01   0.0     X_12    X_13  ...   ]
            [ -X_02  -X_12    0.0     X_23  ...   ]
            [ -X_03  -X_13   -X_23    0.0   ...   ]
            [   .      .       .       .          ]
            [   .      .       .       .          ]
            [   .      .       .       .          ]
      
       The elements of the upper triangle of X are stored in the one-dimensional array x as
          x = [ X_01, X_02, X_03, ..., X_12, X_13, X_14, ..., X_23, X_24, X_25 ... ]

  """

  # check input sanity
  correct_len = (n*(n-1)) // 2
  if len(x) != correct_len:
    raise RuntimeError("unitary_matrix_from_x expected x to have length %i but it had length %i instead" % (correct_len, len(x)) )

  # build antisymmetric matrix
  xmat = np.zeros([n,n])
  k = 0
  for i in range(n-1):
    for j in range(i+1,n):
      xmat[i,j] =  1.0 * x[k]
      xmat[j,i] = -1.0 * x[k]
      k += 1
  if k != correct_len:
    raise RuntimeError("unitary_matrix_from_x found a looping error")

  # exponentiate and return the resulting unitary matrix
  return scipy.linalg.expm(xmat)

def unitary_matrix_deriv(n, x, dfdU):
  """Differentiates through a unitary matrix of the type created by unitary_matrix_from_x

           n --- dimension of the unitary matrix
           x --- one-dimensional array holding the elements of the matrix X (see unitary_matrix_from_x)
        dfdU --- matrix of derivatives of a function f w.r.t. the elements of the unitary matrix U = exp(X)

      Returns the derivatives of f with respect to the elements of x
  """

  # check input sanity
  correct_len = (n*(n-1)) // 2
  if len(x) != correct_len:
    raise RuntimeError("unitary_matrix_deriv expected x to have length %i but it had length %i instead" % (correct_len, len(x)) )
  if dfdU.shape != (n,n):
    raise RuntimeError("unitary_matrix_deriv received a dfdU with the wrong shape.")

  # build antisymmetric matrix
  xmat = np.zeros([n,n])
  k = 0
  for i in range(n-1):
    for j in range(i+1,n):
      xmat[i,j] =  1.0 * x[k]
      xmat[j,i] = -1.0 * x[k]
      k += 1
  if k != correct_len:
    raise RuntimeError("unitary_matrix_deriv found a looping error when building antisymmetric matrix")

  # get derivatives w.r.t. antisymmetric matrix
  dfdX = mat_exp_deriv(xmat, dfdU)

  # evauate and return the derivatives w.r.t. elements of the one-dimensional array x
  dx = np.zeros([correct_len])
  k = 0
  for i in range(n-1):
    for j in range(i+1,n):
      dx[k] = dfdX[i,j] - dfdX[j,i]
      k += 1
  if k != correct_len:
    raise RuntimeError("unitary_matrix_deriv found a looping error when evaluating final derivatives")
  return dx

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
    #print('pop= ',Q_list)
    #print('dfdR= ',dfdR)
    #print('dx= ',dx)
    
    return ofv, dx


# Build the molecule
mol = gto.Mole()
mol.atom = """O 0 0 0; H 0 1 0; H 0 0 1"""
mol.basis='6-31G*'
mol.build()
ao_labels = mol.ao_labels().copy()
# Run HF
mf = scf.RHF(mol)
mf.kernel()

# Grab the MO Coeff matrix and Overlap Matrix
original_MOs=np.copy(mf.mo_coeff)
original_ovlp=np.copy(mf.get_ovlp())
pm=PM(mol)
mo = pm.kernel(mf.mo_coeff[:,0:5])
print(pm.cost_function())

water = mol.copy()

with open("water.molden",'w') as f1:
    molden.header(mol,f1)
    molden.orbital_coeff(mol,f1,original_MOs)

with open("water_PM_pyscf.molden",'w') as f1:
    molden.header(mol,f1)
    molden.orbital_coeff(mol,f1,mo)

inv = np.linalg.inv(original_MOs)

rot = inv[:5,:] @ mo
np.savetxt('rot_pyscf.txt',rot)

print('Pipek rotation')
for row in rot:
  for val in row:
    print('{:.2f} \t'.format(val),end='')
  print()

print("Pipek orbital rotation max deviation from identity = %.6f" % np.amax(np.abs(rot-np.identity(5))) )

# Build molecule with ghost atom
mol = gto.Mole()
mol.atom = """O 0 0 0; H 0 1 0; H 0 0 1; ghost-O 1.1e-5 0 0"""
mol.basis={'O':'6-31G*','H':'6-31G*','ghost-O':gto.basis.parse("""
 X   S
  1.0   1.0
  """)}
mol.build()

# Run HF
mf = scf.RHF(mol)
mf.kernel()

# Grab the MO Coeff matrix and Overlap Matrix
new_MOs=np.copy(mf.mo_coeff)
new_ovlp=np.copy(mf.get_ovlp())


def quick_localize(Cold, S,orbLists, startNoise=0.0, returnR=False, aoGuess=False, startR=None):
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
  if startR is not None:
    xguess=startR
  foo = lambda y: evaluate_obj_func_and_grad_for_quick_localize(Cs, S,orbLists , y)
  print("\nStarting x-based LBFGS", flush=True)
  sol = scipy.optimize.minimize(foo, xguess, jac=True, method='L-BFGS-B')
  R = R @ unitary_matrix_from_x(Cs.shape[1], sol.x)

  # print success of optimization
  print("")
  print('Successful optimization: ',sol.success)
  print('Cause of termination: ',sol.message)

  # print the deviation of the rotation matrix from the identity matrix
  print("")
  print("Quick localizization orbital rotation max deviation from identity = %.6f" % np.amax(np.abs(R-np.identity(ntl))) )

  # print the amount of time the localization took
  print("")
  print("Quick localization took %.6f seconds" % ( time.time() - entry_time ))

  print('Quick localization rotation')
  for row in R:
    for val in row:
      print('{:.2f} \t'.format(val),end='')
    print()

  # return either the orbital coefficients for the newly localized orbitals or the rotation to get to them
  if returnR:
    return R  # return the rotation matrix used to convert to the localized orbitals
  else:
    return Cs @ R  # return the mo coeffs for the localized orbitals

ao_indicies = [
  [i for i in range(14)],
  [14,15],
  [16,17]
]

# Use AO guess from PM

def get_ao_indices(ao_labels, natom):
    ao_indices = []
    start = 0
    norb = len(ao_labels)
    for i in range(natom):
        atom = str(i)
        ao_index = []
        for j in range(start,norb):
            if ao_labels[j][0] == atom:
                ao_index.append(j)
            else:
                start = j
                break
        ao_indices.append(ao_index)
    print('The atomic orbital indices for each atom: ',ao_indices)
    return ao_indices

ao_indices = get_ao_indices(ao_labels,3)

water_LMOs = quick_localize(original_MOs[:,0:5],original_ovlp,ao_indicies,startNoise=0)
with open("water_PM.molden",'w') as f1:
    molden.header(water,f1)
    molden.orbital_coeff(water,f1,water_LMOs)

print('USING PYSCF AS GUESS!!!')
water_LMOs = quick_localize(mo,original_ovlp,ao_indicies,startNoise=0)
with open("new_water_PM.molden",'w') as f1:
    molden.header(water,f1)
    molden.orbital_coeff(water,f1,water_LMOs)

print('NOISY TEST')
water_LMOs = quick_localize(original_MOs[:,0:5],original_ovlp,ao_indicies,startNoise=0.1)
with open("noise_water_PM.molden",'w') as f1:
    molden.header(water,f1)
    molden.orbital_coeff(water,f1,water_LMOs)

diff=water_LMOs-mo
for i in range(len(diff)):
    for j in range(len(diff[0])):
        if diff[i,j] < 1e-6:
            diff[i,j]=0
np.savetxt("diff.txt",diff)

def do_ovlp(left,right,S):
    return np.transpose(left)@S@right

ovlp=do_ovlp(mo,water_LMOs,water.intor('int1e_ovlp'))
for i in range(len(ovlp)):
    for j in range(len(ovlp[0])):
        if ovlp[i,j] < 1e-6:
            ovlp[i,j]=0
np.savetxt('do_ovlp.txt',ovlp)

exit()
mol = gto.Mole()
mol.atom = benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
[ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
[ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
[ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
[ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
[ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
[ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
[ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
[ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
[ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
[ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
[ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]
mol.basis='6-31G*'
mol.build()
ao_labels = mol.ao_labels().copy()
# Run HF
mf = scf.RHF(mol)
mf.kernel()
benzene_MOs=np.copy(mf.mo_coeff)
benzene_ovlp=np.copy(mf.get_ovlp())
ao_labels=mol.ao_labels()
ao_indices = get_ao_indices(ao_labels,12)

benzene_LMOs = quick_localize(benzene_MOs[:,0:21],benzene_ovlp,ao_indicies,startNoise=0)
with open("benzene_PM.molden",'w') as f1:
    molden.header(mol,f1)
    molden.orbital_coeff(mol,f1,benzene_LMOs)