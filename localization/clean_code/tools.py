import numpy as np
import scipy

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

def get_ao_indices(ao_labels, natom):
    ao_indices = []
    start = 0
    norb = len(ao_labels)
    for i in range(natom):
        atom = str(i) + ' '
        ao_index = []
        for j in range(start,norb):
            if ao_labels[j].startswith(atom):
                ao_index.append(j)
            else:
                start = j
                break
        ao_indices.append(ao_index)
    print('The atomic orbital indices for each atom: ',ao_indices)
    return ao_indices