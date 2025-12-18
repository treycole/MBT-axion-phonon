import numpy as np

#NOTE: Change here
addcol = True
nks = [8, 8, 8]

#NOTE: Keep below unchanged
end_pts = [0, 1]
k_vals = [np.linspace(end_pts[0], end_pts[1], nk, endpoint=False) for nk in nks]
sq_mesh = np.stack(np.meshgrid(*k_vals, indexing='ij'), axis=-1)
flat_mesh = sq_mesh.reshape(-1, len(k_vals))
Nk = flat_mesh.shape[0]
if addcol:
    # Add 4th column 
    fourth_column = np.full((flat_mesh.shape[0], 1), 1/Nk)
    flat_mesh = np.hstack((flat_mesh, fourth_column))
print(flat_mesh.shape)

# Save as CSV
np.savetxt('k_mesh.csv', flat_mesh, delimiter=' ', fmt='%.12f')
# Save as TXT
np.savetxt('k_mesh.txt', flat_mesh, fmt='%.12f')
