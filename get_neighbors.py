import MDAnalysis as mda
from MDAnalysis.analysis import rms
import MDAnalysis.transformations as transformations
from MDAnalysis.lib.distances import distance_array
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import csv

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pairwise RMSD calculation and clustering of oriented protein and nearest cholesterols.')
    parser.add_argument('initial_frame', type=int, help='Initial frame number.')
    parser.add_argument('final_frame', type=int, help='Final frame number.')
    parser.add_argument('chol_nsel', type=int, help='Number of cholesterol molecules to select.')
    return parser.parse_args()

args = parse_arguments()

output_filename = './data/coordinates.txt'

# setup for pairwise rmsd matrix made later
adjusted_coordinates = []

with open(output_filename, 'w') as output_file:
    for replica in range(1, 4):
        base_path = "/data/beavenah/syt7"
        dcd_files = [f"{base_path}/production/c2a/rest/pc_ps_chol/rep_{replica}a/data/dcd/recenter/step7_{i}_com.dcd"\
                     for i in range(args.initial_frame, args.final_frame+1)]
        topology_file = f"{base_path}/build/c2a/pc_ps_chol/rep_{replica}/step5_now.psf"
        u = mda.Universe(topology_file, dcd_files)

        for ts in u.trajectory[::10]:

            cur = u.trajectory.filename
            dcd_num = int(cur.split('_')[-2])
            output_file.write(f"{ts} {dcd_num} {ts.frame % 200}\n")

            print(ts.frame)
            for prot_seg in ('PROA', 'PROB', 'PROC', 'PROD'):
                print(prot_seg)
                # center frame
                protein_total = u.select_atoms('segid %s and not name H*' % prot_seg)
                cbl_target = u.select_atoms('segid %s and resid 229 and name CA' % prot_seg)
                cbl_total = u.select_atoms('segid %s and (resid 166 or resid 172 or resid 225 or resid 227 or resid 233) and not name H*' % prot_seg)
                apx = u.select_atoms('segid %s and resid 255 and name CA' % prot_seg)

                box_dims = u.dimensions[:3]
                cbl_com = cbl_total.center_of_mass()
                membrane = u.select_atoms('segid MEMB')
                membrane_com_z = membrane.center_of_mass()[2]
                trans_vec = [box_dims[0]/2 - cbl_com[0], box_dims[1]/2 - cbl_com[1], box_dims[2]/2 - membrane_com_z]
                u.atoms.translate(trans_vec)

                prot = u.select_atoms('segid PRO*')
                not_prot = u.select_atoms('not segid PRO*')
                transformations.wrap(not_prot, compound='residues')(u.trajectory.ts)
                transformations.wrap(prot, compound='segments')(u.trajectory.ts)

                trans_vec = [-box_dims[0]/2, -box_dims[1]/2, -box_dims[2]/2]
                u.atoms.translate(trans_vec)

                # get the correct leaflet chol selections and rotate
                z_axis = [0, 0, 1]
                if prot_seg == 'PROA' or prot_seg == 'PROC':
                    cholesterol = u.select_atoms('resname CHL1 and name O3 and resid 1:144')
                    offset = 0
                    vec = protein_total.center_of_mass()[:2]
                    angle_rad = np.arctan2(vec[1], vec[0]) + np.pi
                    rotation_angle_deg = -np.degrees(angle_rad)
                else:
                    cholesterol = u.select_atoms('resname CHL1 and name O3 and resid 145:288')
                    offset = 144
                    rotation = R.from_euler('x', 180, degrees=True)
                    rotation_matrix = rotation.as_matrix()
                    u.atoms.positions = np.dot(u.atoms.positions, rotation_matrix.T)
                    vec = protein_total.center_of_mass()[:2]
                    angle_rad = np.arctan2(vec[1], vec[0]) + np.pi
                    rotation_angle_deg = -np.degrees(angle_rad)

                transformations.rotate.rotateby(rotation_angle_deg, direction=z_axis, ag=cbl_total)(u.trajectory.ts)

                # get nearest chol relative distances from the cbl_target
                distances = distance_array(cholesterol.positions, cbl_target.positions, box=u.trajectory.ts.dimensions)
                nearest_dists = distances.min(axis=1)
                sorted_indices = nearest_dists.argsort()
                nearest_indices = sorted_indices[:args.chol_nsel]
                indx1 = nearest_indices[0] + offset + 1
                indx2 = nearest_indices[1] + offset + 1

                # get x,y,z of nearest chol and print
                nearest_cholesterol = cholesterol[nearest_indices]
                nearest_cholesterol_flat = ' '.join(map(str, nearest_cholesterol.positions.flatten()))
                prot_com = protein_total.center_of_mass()
                cbl_com = cbl_total.center_of_mass()
                apx_com = apx.center_of_mass()

                selected_coords = np.vstack((apx_com, prot_com, cbl_com, nearest_cholesterol.positions))
                adjusted_coordinates.append(selected_coords)

                output_string = (f"prot com: {prot_com[0]} {prot_com[1]} {prot_com[2]} "
                                 f"cbl com: {cbl_com[0]} {cbl_com[1]} {cbl_com[2]} "
                                 f"apx com: {apx_com[0]} {apx_com[1]} {apx_com[2]} "
                                 f"chl coords: {nearest_cholesterol_flat} "
                                 f"info: {prot_seg} {replica} {dcd_num} {ts.frame % 200} {indx1} {indx2}\n")
                output_file.write(output_string)

                # undo the rotations
                transformations.rotate.rotateby(-rotation_angle_deg, direction=z_axis, ag=cbl_total)(u.trajectory.ts)
                if prot_seg == 'PROB' or prot_seg == 'PROD':
                    rotation = R.from_euler('x', -180, degrees=True)
                    rotation_matrix = rotation.as_matrix()
                    u.atoms.positions = np.dot(u.atoms.positions, rotation_matrix.T)

# setup for pairwise rmsd matrix
n_frames = len(adjusted_coordinates)
pairwise_rmsd = np.zeros((n_frames, n_frames))
# make the pairwise rmsd matrix
for i in range(n_frames):
    # start j from i to avoid redundant calculations
    for j in range(i, n_frames):
        # center=True translates to minimum, superposition=True translates and rotates to minimum
        rmsd_value = rms.rmsd(adjusted_coordinates[i], adjusted_coordinates[j], center=True)
        pairwise_rmsd[i, j] = rmsd_value
        # rmsd is symmetrical -- save time
        pairwise_rmsd[j, i] = rmsd_value

np.savetxt('./data/rmsd_metric.csv', pairwise_rmsd, delimiter=',')
