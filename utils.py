import torch
import numpy as np
import math
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def initializeCubes(resol, cube_D, cube_Dcenter, cube_overlapping_ratio, BB):
    """
    generate {N_cubes} 3D overlapping cubes, each one has {N_cubeParams} embeddings
    for the cube with size of cube_D^3 the valid prediction region is the center part, say, cube_Dcenter^3
    E.g. cube_D=32, cube_Dcenter could be = 20. Because the border part of each cubes don't have accurate prediction because of ConvNet.

    ---------------
    inputs:
        resol: resolusion of each voxel in the CVC (mm)
        cube_D: size of the CVC (Colored Voxel Cube)
        cube_Dcenter: only keep the center part of the CVC, because of the boundery effect of ConvNet.
        cube_overlapping_ratio: pertantage of the CVC are covered by the neighboring ones
        BB: bounding box, numpy array: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
    outputs:
        cubes_param_np: (N_cubes, N_params) np.float32
        cube_D_mm: scalar

    ---------------
    usage:
    >>> cubes_param_np, cube_D_mm = initializeCubes(resol=1, cube_D=22, cube_Dcenter=10, cube_overlapping_ratio=0.5, BB=np.array([[3,88],[-11,99],[-110,-11]]))
    xyz bounding box of the reconstructed scene: [ 3 88], [-11  99], [-110  -11]
    >>> print cubes_param_np[:3]
    [([   3.,  -11., -110.], [0, 0, 0],  1.)
     ([   3.,  -11., -105.], [0, 0, 1],  1.)
     ([   3.,  -11., -100.], [0, 0, 2],  1.)]
    >>> print cubes_param_np['xyz'][18:22]
    [[   3.  -11.  -20.]
     [   3.  -11.  -15.]
     [   3.   -6. -110.]
     [   3.   -6. -105.]]
    >>> np.allclose(cubes_param_np['xyz'][18:22], cubes_param_np[18:22]['xyz'])
    True
    >>> print cube_D_mm
    22
    """

    cube_D_mm = resol * cube_D   # D size of each cube along each axis,
    cube_Center_D_mm = resol * cube_Dcenter   # D size of each cube's center that is finally remained
    cube_stride_mm = cube_Center_D_mm * cube_overlapping_ratio # the distance between adjacent cubes,
    safeMargin = (cube_D_mm - cube_Center_D_mm)/2

    print('xyz bounding box of the reconstructed scene: {}, {}, {}'.format(*BB))
    N_along_axis = lambda _min, _max, _resol: int(math.ceil((_max - _min) / _resol))
    N_along_xyz = [N_along_axis( (BB[_axis][0] - safeMargin), (BB[_axis][1] + safeMargin), cube_stride_mm) for _axis in range(3)]   # how many cubes along each axis
    # store the ijk indices of each cube, in order to localize the cube
    cubes_ijk = np.indices(tuple(N_along_xyz))
    N_cubes = cubes_ijk.size / 3   # how many cubes
    N_cubes = int(N_cubes)
    cubes_param_np = np.empty(((N_cubes),), dtype=[('xyz', np.float32, (3,)), ('ijk', np.int32, (3,)), ('resol', np.float32)])    # attributes for each CVC (colored voxel cube)
    cubes_param_np['ijk'] = cubes_ijk.reshape([3,-1]).T  # i/j/k grid index
    cubes_xyz_min = cubes_param_np['ijk'] * cube_stride_mm + (BB[:,0][None,:] - safeMargin)
    cubes_param_np['xyz'] = cubes_xyz_min    # x/y/z coordinates (mm)
    cubes_param_np['resol'] = resol

    return cubes_param_np, cube_D_mm
def gen_colored_cubes(position, img, xyz, resol, colorize_cube_D):
    min_x, min_y, min_z = xyz
    indx_xyz = range(0, colorize_cube_D)
    indx_x, indx_y, indx_z = np.meshgrid(indx_xyz, indx_xyz, indx_xyz, indexing='ij')
    indx_x = indx_x * resol + min_x
    indx_y = indx_y * resol + min_y
    indx_z = indx_z * resol + min_z
    homogen_1s = np.ones(colorize_cube_D ** 3, dtype=np.float64)
    pts_4D = np.vstack([indx_x.flatten(), indx_y.flatten(), indx_z.flatten(), homogen_1s])
    homo = np.zeros(4,dtype = np.float64)
    homo[3] = 1
    position = np.vstack([position,homo])
    #colored_cube = np.zeros((3, colorize_cube_D, colorize_cube_D, colorize_cube_D))

    # perspective projection
    projection_M = position
    pts_3D = projection_M @ pts_4D  #(4,4) * (4, colorize_cube_D ***3)
    pts_3D = pts_3D[:-1]
    pts_3D[:-1] /= pts_3D[-1]  # the result is vector: [w,h,1], w is the first dim
    pts_2D = pts_3D[:-1].round().astype(np.int32)
    pts_w, pts_h = pts_2D[0], pts_2D[1]
    # access rgb of corresponding model_img using pts_2D coordinates
    pts_RGB = np.zeros((colorize_cube_D ** 3, 3))
    max_h, max_w, _ = img.shape
    inScope_pts_indx = (pts_w < max_w) & (pts_h < max_h) & (pts_w >= 0) & (pts_h >= 0)
    pts_RGB[inScope_pts_indx] = img[pts_h[inScope_pts_indx], pts_w[inScope_pts_indx]]
    colored_cube= pts_RGB.T.reshape((3, colorize_cube_D, colorize_cube_D, colorize_cube_D))

    return colored_cube    # [views_N, 3, colorize_cube_D, colorize_cube_D, colorize_cube_D]


def gen_colored_cubes_viewpair(selected_viewPairs, xyz, resol, cameraPOs, models_img, colorize_cube_D, visualization_ON=False, \
                               occupiedCubes_01=None):
    """
    inputs:
    selected_viewPairs: (N_cubes, N_select_viewPairs, 2)
    xyz, resol: parameters for each occupiedCubes (N,params)
    occupiedCubes_01: multiple occupiedCubes (N,)+(colorize_cube_D,)*3
    return:
    coloredCubes = (N*N_select_viewPairs,3*2)+(colorize_cube_D,)*3
    """
    N_cubes, N_select_viewPairs = selected_viewPairs.shape[:2]
    coloredCubes = np.zeros((N_cubes, N_select_viewPairs * 2, 3) + (colorize_cube_D,) * 3,
                            dtype=np.float32)  # reshape at the end

    for _n_cube in range(0, N_cubes):  ## each cube
        if visualization_ON:
            if occupiedCubes_01 is None:
                print
                'error: [func]gen_coloredCubes, occupiedCubes_01 should not be None when visualization_ON==True'
            occupiedCube_01 = occupiedCubes_01[_n_cube]
        else:
            occupiedCube_01 = None
        ##randViewIndx = random.sample(range(1,cameraPOs.shape[0]),N_randViews)

        # (N_cubes, N_select_viewPairs, 2) ==> (N_select_viewPairs*2,).
        selected_views = selected_viewPairs[_n_cube].flatten()
        # because selected_views could include duplicated views, this case is not the best way. But if the N_select_viewPairs is small, it doesn't matter too much
        coloredCube = gen_colored_cubes(view_set=selected_views, \
                                        position=cameraPOs, imgs=models_img, xyz=xyz[_n_cube],
                                        resol=resol[_n_cube], \
                                        visualization_ON=visualization_ON, colorize_cube_D=colorize_cube_D,
                                        densityCube=occupiedCube_01)

        # [a,b,c] ==> [a,b,a,c,b,c]
        ##all_pairIndx = ()
        ##for _pairIndx in itertools.combinations(range(0,N_randViews),2):
        ##all_pairIndx += _pairIndx
        ##all_pairIndx = list(all_pairIndx)

        # # [a,b,c,d,e,f,g,h,i,j] ==> [a,b,g,c,f,e]
        # all_pairIndx = []
        # for _pairIndx in itertools.combinations(range(0,N_randViews),2):
        #     all_pairIndx.append(_pairIndx)
        # all_pairIndx = random.sample(all_pairIndx, N_select_viewPairs)
        # all_pairIndx = [x for pair_tuple in all_pairIndx for x in pair_tuple] ## [(a,),(a,b),(a,b,c)] ==> [a,a,b,a,b,c]

        coloredCubes[_n_cube] = coloredCube

    return coloredCubes.reshape((N_cubes * N_select_viewPairs, 3 * 2) + (colorize_cube_D,) * 3)

