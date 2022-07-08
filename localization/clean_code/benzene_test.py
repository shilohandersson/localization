from localize_orbs import *
from tools import *
from pyscf import scf, gto, lo, molden
from pyscf.lo import PM
import numpy as np
from pyscf.gto import mole

# BENZENE
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

benzene_LMOs = quick_localize(benzene_MOs[:,0:21],benzene_ovlp,ao_indices,startNoise=0.0)
with open("jmol/benzene_PM.molden",'w') as f1:
    molden.header(mol,f1)
    molden.orbital_coeff(mol,f1,benzene_LMOs)

pm=PM(mol)
pm.pop_method='mulliken'
pyscf_benezene = pm.kernel(benzene_MOs[:,0:21])
print(pm.cost_function())
with open("jmol/benzene_PM_pyscf.molden",'w') as f1:
    molden.header(mol,f1)
    molden.orbital_coeff(mol,f1,pyscf_benezene)

print('\nUSING PYSCF AS A GUESS')
guess = quick_localize(pyscf_benezene,benzene_ovlp,ao_indices)

print('\nPLAYING WITH TOLERANCE')
tol_benzene_LMOs = quick_localize(benzene_MOs[:,:21],benzene_ovlp,ao_indices,tolerance=1e-6)
with open("jmol/tol_benzene_PM.molden",'w') as f1:
    molden.header(mol,f1)
    molden.orbital_coeff(mol,f1,tol_benzene_LMOs)