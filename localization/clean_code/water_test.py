from localize_orbs import *
from tools import *
from pyscf import scf, gto, lo, molden
from pyscf.lo import PM
import numpy as np
from pyscf.gto import mole

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
pm.pop_method='mulliken'
pyscf_mo = pm.kernel(mf.mo_coeff[:,0:5])
print(pm.cost_function())

water = mol.copy()

with open("jmol/h2o_PM_pyscf.molden",'w') as f1:
    molden.header(water,f1)
    molden.orbital_coeff(water,f1,pyscf_mo)

inv = np.linalg.inv(original_MOs)

pyscf_rot = inv[:5,:] @ pyscf_mo

ao_indices = get_ao_indices(ao_labels,3)

water_LMOs = quick_localize(original_MOs[:,0:5],original_ovlp,ao_indices,startNoise=0)
with open("jmol/h2o_PM.molden",'w') as f1:
    molden.header(water,f1)
    molden.orbital_coeff(water,f1,water_LMOs)

print('\nUSING PYSCF AS GUESS!!!')
water_LMOs = quick_localize(pyscf_mo,original_ovlp,ao_indices,startNoise=0)

print('\nNOISY TEST')
water_LMOs = quick_localize(original_MOs[:,0:5],original_ovlp,ao_indices,startNoise=0.1)
with open("jmol/noise_h2o_PM.molden",'w') as f1:
    molden.header(water,f1)
    molden.orbital_coeff(water,f1,water_LMOs)

