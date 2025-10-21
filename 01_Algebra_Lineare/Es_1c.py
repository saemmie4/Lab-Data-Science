import numpy as np
import matplotlib.pylab as plt
import math
from ase import Atoms
from ase.io import write, read

TOLERANCE = 0.01

def center_mass(molecule):
    masses = molecule.get_masses()
    positions = molecule.get_positions()
    center_mass_res = np.array([0.,0.,0.])
    total_mass = 0 
    for j in np.arange(len(masses)):
        center_mass_res += masses[j]*positions[j]
        total_mass+=masses[j]
    
    return center_mass_res / total_mass


def moments_of_inertia(molecule):
    #positions relative to the center of mass
    positions = molecule.get_positions() - center_mass(molecule)
    masses = molecule.get_masses()
    I = np.zeros((3,3))
    for j in np.arange(len(positions)):
        I[0,0] += masses[j]*((positions[j][1]**2 + positions[j][2]**2)) #Ixx
        I[1,1] += masses[j]*((positions[j][0]**2 + positions[j][2]**2)) #Iyy
        I[2,2] += masses[j]*((positions[j][0]**2 + positions[j][1]**2)) #Izz
        
        I[0,1] += -masses[j]*((positions[j][0] * positions[j][1])) #Ixy
        I[1,2] += -masses[j]*((positions[j][1] * positions[j][2])) #Iyz
        I[0,2] += -masses[j]*((positions[j][0] * positions[j][2])) #Ixz
    
    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]

    return np.linalg.eigvalsh(I)
def molecule_type(molecule):
    """returns id del tipo

    tipo,          id tipo:
    * Sferica     = 0
    * Oblata      = 1
    * Prolata     = 2
    * Asimmetrica = 3 """

    mom_in = moments_of_inertia(molecule)
    if math.isclose(mom_in[0], mom_in[1], rel_tol = TOLERANCE):
    # if abs(mom_in[0] - mom_in[1]) < 0.01:
        if math.isclose(mom_in[1], mom_in[2], rel_tol=TOLERANCE):
        # if abs(mom_in[1] - mom_in[2]) < 0.01:
            return 0
        else:
            return 1
    else:
        if math.isclose(mom_in[1], mom_in[2], rel_tol=TOLERANCE):
        # if abs(mom_in[1] - mom_in[2]) < 0.01:
            return 2
        else:
            return 3
    
Id_to_molecule_type_string = ["Sferica", "Oblata", "Prolata", "Asimmetrica"]
number_of_molecule_type = [0,0,0,0]
DataSet = read ("dataset.xyz", index=':')
# masses = [molecule.get_masses() for molecule in DataSet]
# positions = [molecule.get_positions() for molecule in DataSet]
# center_masses = [center_mass(molecule) for molecule in DataSet]
chemical_formulas = [molecule.get_chemical_formula() for molecule in DataSet]
molecule_type_id = [molecule_type(molecule) for molecule in DataSet]

with open("Es_1c.txt", "w") as file_out:
    for j in np.arange(len(chemical_formulas)):
        number_of_molecule_type[molecule_type_id[j]]+=1
        print(chemical_formulas[j] + ": " + Id_to_molecule_type_string[molecule_type_id[j]])
        file_out.write(chemical_formulas[j] + ": " + Id_to_molecule_type_string[molecule_type_id[j]]+ '\n')

    print("\n\nTipo\tConteggio")
    file_out.write("\n\nTipo\tConteggio\n")
    for j in np.arange(len(number_of_molecule_type)):
        print(Id_to_molecule_type_string[j] + ("\t%d" %number_of_molecule_type[j]))
        file_out.write(Id_to_molecule_type_string[j] + ("\t%d" %number_of_molecule_type[j])+ '\n')