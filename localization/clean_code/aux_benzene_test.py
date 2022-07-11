from localize_orbs import *
from localization_full_overlap_matrix import ghost_basis
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
ao_indices = get_ao_indices(ao_labels,12)

# Build ghost basis for carbon
carbon_basis= ghost_basis()
carbon_basis.add_basis_function('S',[1.0],[1.0])
carbon_basis.add_basis_function('P',[1.0],[1.0])

# Build secondary molecule for doing overlap matrix
ghost_list = [
    "ghost-C 4.673795 6.280948 0",
    "ghost-C 5.901190 5.572311 0",
    "ghost-C 5.901190 4.155037 0"
]

ghost_atom_type = 'ghost-C'

def get_cross_overlap_matrix(ghost_atom_type,ghost_atom,atom_basis,mol):
    ghost_mol = gto.Mole()
    ghost_mol.atom = ghost_atom
    ghost_mol.basis={ghost_atom_type:gto.basis.parse(atom_basis())}
    ghost_mol.build()

    cross_overlap_matrix=mole.intor_cross('int1e_ovlp',mol,ghost_mol)
    return cross_overlap_matrix

for atom in ghost_list:
    cross_overlap_matrix = get_cross_overlap_matrix(ghost_atom_type,ghost_list,carbon_basis,mol)
    print(cross_overlap_matrix.shape)

benzene_LMOs = quick_localize(benzene_MOs[:,0:21],benzene_ovlp,ao_indices,startNoise=0.1)
with open("jmol/noisy_benzene_PM.molden",'w') as f1:
    molden.header(mol,f1)
    molden.orbital_coeff(mol,f1,benzene_LMOs)