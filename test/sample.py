import os, sys

import polyscope as ps
import numpy as np
import scipy.sparse.linalg as sla
from plyfile import PlyData, PlyElement


def export_pcl_with_eigen_to_ply(verts, colors, eigenvecs, eigenvals, filepath):
    num_pts = verts.shape[0]
    num_eigenvecs = eigenvecs.shape[1]
    verts = np.ascontiguousarray(verts)
    colors = np.ascontiguousarray(colors)
    eigenvecs = np.ascontiguousarray(eigenvecs)

    verts = [(verts[i, 0], verts[i, 1], verts[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
            eigenvecs[i, 0], eigenvecs[i, 1], eigenvecs[i, 2],
            eigenvecs[i, 3], eigenvecs[i, 4], eigenvecs[i, 5],
            eigenvecs[i, 6], eigenvecs[i, 7], eigenvecs[i, 8],
            eigenvecs[i, 9]
            )
            for i in range(num_pts)]
    factor = [(
            eigenvals[0], eigenvals[1], eigenvals[2],
            eigenvals[3], eigenvals[4], eigenvals[5],
            eigenvals[6], eigenvals[7], eigenvals[8],
            eigenvals[9]
            )]

    pts_attrib = ['x', 'y', 'z']
    colors_attrib = ['r', 'g', 'b']
    eigenvecs_attrib = [f'evec_{i}' for i in range(num_eigenvecs)]
    pts_dtype = [(f'{attrib}', 'f4') for attrib in pts_attrib]
    colors_dtype = [(f'{attrib}', 'u1') for attrib in colors_attrib]
    eigenvecs_dtype = [(f'{attrib}', 'f4') for attrib in eigenvecs_attrib]

    eigenvals_attrib = [f'eval_{i}' for i in range(num_eigenvecs)]
    eigenvals_dtype = [(f'{attrib}', 'f4') for attrib in eigenvals_attrib]

    vertices = np.array(verts, dtype=pts_dtype + colors_dtype + eigenvecs_dtype)
    factors = np.array(factor, dtype=eigenvals_dtype)


    plyel = PlyElement.describe(vertices, 'vertex')
    factorel = PlyElement.describe(factors, 'factors')

    plydata = PlyData([plyel,
                       factorel
                    #    , colorel
                    #    , eigenvecs_el
                       ], text=False)
    plydata.write(filepath)


# Path to where the bindings live
sys.path.append(os.path.join(os.path.dirname(__file__), "../build/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

import robust_laplacian

# Read input
plydata = PlyData.read("/path/to/ply")
points = np.vstack((
    plydata['vertex']['x'],
    plydata['vertex']['y'],
    plydata['vertex']['z']
)).T
rgbs = np.vstack((
    plydata['vertex']['red'],
    plydata['vertex']['blue'],
    plydata['vertex']['green']
)).T
# for meshes
# tri_data = plydata['face'].data['vertex_indices']
# faces = np.vstack(tri_data)

# Build Laplacian
L, M = robust_laplacian.point_cloud_laplacian(points, mollify_factor=1e-5)

# for meshes
# L, M = robust_laplacian.mesh_laplacian(points, faces, mollify_factor=1e-5)

# Compute some eigenvectors
n_eig = 10
evals, evecs = sla.eigsh(L, n_eig, M, sigma=1e-8)

evecs32 = evecs.astype(np.float32)
evals32 = evals.astype(np.float32)
export_pcl_with_eigen_to_ply(points, rgbs, evecs, evals, 'mug13_eigen.ply')
raise
# Visualize
# ps.init()
# ps_cloud = ps.register_point_cloud("my cloud", points)
# for i in range(n_eig):
#     ps_cloud.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)
#     ps.set_up_dir("z_up")
#     ps.look_at((0., 0., 2.), (0., 0., 0.))
#     ps.set_ground_plane_mode("none")
#     ps.screenshot(filename=f"screenshot_eigenvector_{i}.png")
#     # Disable the current scalar quantity
#     ps_cloud.remove_quantity("eigenvector_"+str(i))
    

# for meshes
# ps_surf = ps.register_surface_mesh("my surf", points, faces)
# for i in range(n_eig):
    # ps_surf.add_scalar_quantity("eigenvector_"+str(i), evecs[:,i], enabled=True)

# ps.show()
