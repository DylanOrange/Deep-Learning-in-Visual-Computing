import numpy as np
from plyfile import PlyData, PlyElement

def write_ply(points, face_data, filename, text=True):
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])

    face = np.empty(len(face_data),dtype=[('vertex_indices', 'i4', (4,))])
    face['vertex_indices'] = face_data

    ply_faces = PlyElement.describe(face, 'face')
    ply_vertexs = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ply_vertexs, ply_faces], text=text).write(filename)

def occ2points(occ, dim):
    points  = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                if occ[i,j,k] == True:
                    points.append(np.array([i,j,k]))
    return np.array(points)

def generate_faces(points):
    corners = np.zeros((8*len(points),3))
    faces = np.zeros((6*len(points),4))
    for index in range(len(points)):
        corners[index*8]= np.array([points[index,0]-0.5, points[index,1]-0.5, points[index,2]-0.5])#左下后
        corners[index*8+1]= np.array([points[index,0]+0.5, points[index,1]-0.5, points[index,2]-0.5])#右下后
        corners[index*8+2]= np.array([points[index,0]-0.5, points[index,1]+0.5, points[index,2]-0.5])#左下前
        corners[index*8+3]= np.array([points[index,0]+0.5, points[index,1]+0.5, points[index,2]-0.5])#右下前
        corners[index*8+4]= np.array([points[index,0]-0.5, points[index,1]-0.5, points[index,2]+0.5])#左上后
        corners[index*8+5]= np.array([points[index,0]+0.5, points[index,1]-0.5, points[index,2]+0.5])#右上后
        corners[index*8+6]= np.array([points[index,0]-0.5, points[index,1]+0.5, points[index,2]+0.5])#左上前
        corners[index*8+7]= np.array([points[index,0]+0.5, points[index,1]+0.5, points[index,2]+0.5])#右上前
        faces[index*6]= np.array([len(points)+8*index, len(points)+8*index+1,len(points)+8*index+2,len(points)+8*index+3])
        faces[index*6+1]= np.array([len(points)+8*index+4, len(points)+8*index+5,len(points)+8*index+6,len(points)+8*index+7])
        faces[index*6+2]= np.array([len(points)+8*index+2, len(points)+8*index+3,len(points)+8*index+6,len(points)+8*index+7])
        faces[index*6+3]= np.array([len(points)+8*index+0, len(points)+8*index+1,len(points)+8*index+4,len(points)+8*index+5])
        faces[index*6+4]= np.array([len(points)+8*index+0, len(points)+8*index+2,len(points)+8*index+4,len(points)+8*index+6])
        faces[index*6+5]= np.array([len(points)+8*index+1, len(points)+8*index+3,len(points)+8*index+5,len(points)+8*index+7])
    return corners, faces

def writeocc(file_path,save_path):
    occ_data = np.load(file_path)
    print(occ_data.files)
    occ = occ_data['arr_0']
    points = occ2points(occ,64)
    corners, faces = generate_faces(points)
    points = np.concatenate((points,corners),axis=0)
    write_ply(points, faces, save_path)

