from molSimplify.Scripts.generator import startgen_pythonic
import numpy as np
import os 
import shutil 

def reset_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  
    os.makedirs(dir_path)        
    print(f"Setting the output directory: {dir_path}")


def write_DFT_Script(filename, nprocs, iterations, structure_file, charge, multiplicity, constraints=False, gbw_name=None):
    with open(filename,"w") as f:
        f.write("! BP86 def2-TZVPP def2/J UNO\n") # setting the general DFT functional for all elements in the mol - det is the main and the aux basis set 
        f.write("! TightSCF \n") # this sets the scf convergeence threshold - 10^-8 
        f.write("! TightOpt D3BJ \n") # this sets the optimization convergence threshold 
        f.write("! NoFinalGridX \n") # do not print the final integration grid 
        if not gbw_name==None:
            f.write("! MOREAD \n")
            f.write('%moinp "{}.gbw" \n'.format(gbw_name))
        f.write('%basis \n')   # % initiates a block and 'end' ends a block 
        f.write('newgTO Y "SARC-DKH-TZVPP" end \n')  # setting a new set of basis for Y, Sn and I because def2-TZVPP def2/J only runs for upto Kr atoms 
        f.write('newgTO Sn "SARC-DKH-TZVPP" end \n')
        f.write('newgTO I "SARC-DKH-TZVPP" end \n')
        f.write('end \n')
        f.write("%maxcore 3000 \n") # max memory per core - it's in MB - orca overhoots this value by 40 % so when setting the value be careful 
        # meluxina needs 4G - Lumi needs 2G - on local PC it is 3G 
        f.write("%method \n") 
        f.write(" Grid 6 \n") # this sets the grid density for integration over orbitals - 6 is code for the value and not the value itself 
        f.write("end \n")
        f.write("%scf \n")
        f.write("AutoTRAH false \n") # turn off the autoTrah method in converging scf - it messes with the setting of the convergence 
        f.write(" MaxIter 1001 \n") # maxx no of iteration for scf 
        f.write("end \n")
        f.write("%geom \n")
        # if len(constraints)>0:   # setting constrains when optimizing the mol -- this helps fix some atoms, their distances and angles 
        #     f.write("Constraints \n")
        #     for constraint in constraints:
        #         f.write(constraint+" \n")
        #     f.write("end \n")
        if constraints == True:
            f.write('Constraints \n')
            f.write('{A 1 0 4 C} \n') # angles between atom 1 0 and 4 is constant.
            f.write('{A 4 0 7 C} \n') # this keeps the angle between the in-plane oxygens identical and unchanged.
            f.write('{A 7 0 10 C} \n')
            f.write('{A 10 0 13 C} \n')
            f.write('{A 13 0 1 C} \n')
            f.write('{D 4 1 0 7 C} \n') # Applying a dihedral angle constraint specified by the atoms 4 1 0 and 7.
            f.write('{D 7 4 0 10 C} \n') # D force the oxygens and Dy stay in the same plane 
            f.write('{D 10 7 0 13 C} \n')
            f.write('{D 13 10 0 1 C} \n')
            f.write('{D 1 13 0 4 C} \n')
            f.write("end \n")
        f.write(" MaxIter {} \n".format(iterations))  # max no of iteraetions for optimization 
        f.write("end \n")
        f.write("%pal \n")
        f.write(" nprocs {} \n".format(nprocs))  # this sets the no of processors 
        f.write("end \n")
        f.write("* xyzfile {} {} {}.xyz *\n".format(charge, multiplicity, structure_file)) # read the coordinates from an external xyz file 

def write_samples(input_path, out_dir, seed):
    reset_directory(out_dir+(seed.split('.txt'))[0])
    with open(input_path+seed, 'r') as file:
        strings = file.readlines()
    smile_string = [(string.split())[0] for string in strings]
    charges = [int((string.split())[1]) for string in strings]

    for iter, string in enumerate(smile_string):
        work_path=out_dir+f"{(seed.split('.txt'))[0]}/"
        lig=",".join([string,string])
        input_dic={ # molsimplify dic to generate an xyz
                    "-core": "dy_dummy",
                    "-geometry": "li",
                    "-ccatoms": "1,1",#atom of the core to bind each ligand to
                    "-coord": "1,1",#denticity of each ligand
                    "-lig": lig,#which ligands to attach, separated by commas
                    "-keepHs": "Yes, Yes",
                    "-smicat": "[[{}],[{}]]".format(1,1),#connecting atom on the ligands
                    "-ligocc": "1,1",#number of each ligand appearing
                    "-lignum": "2",#number of different ligands
                    "-spin": "1",
                    "-oxstate": "III",
                    "-rundir": work_path,
                    "calccharge":"True",
                    "-ffoption":"N",#which forcefield to use, N=None, B=Before, A=After
                    }
        print(input_dic)
        startgen_pythonic(input_dict=input_dic,write=True,flag=False)
        name="dy_dummy_li_3_smi1_1_smi2_1_s_1"
        filename=work_path+f"{name}/{name}_conf_1/{name}_conf_1.xyz"
        if os.path.isfile(filename):
            reset_directory(work_path+f"sample_{iter}")
            os.rename(filename, work_path+f"sample_{iter}/raw_compound.xyz")
            write_DFT_Script(work_path+f"sample_{iter}/geometry_DFT.inp", 8, 201, "raw_compound", 2*charges[iter]+3, 1)
            shutil.rmtree(work_path+name)
        else:
            shutil.rmtree(work_path+name)




keyword = 'relaxed'
folder_name = 'test5'

out_dir = f'/media/zahra/zahraext/storage/DATA_jupyter/mol_proj/results_n_data/VAE+DNN_results/novel_samples_{keyword}/size_114/{folder_name}/'
input_path = f'/home/zahra/Dropbox/MOLDISC_proj/py_codes/pytorch/masked_VAE+DNN/results_n_data/seeds_{keyword}/size_114/{folder_name}/seeds_for_cas/'
seeds = os.listdir(input_path)


for seed in seeds:
    if 'seed' in seed:
        write_samples(input_path, out_dir, seed)


