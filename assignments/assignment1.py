import trimesh

def face_to_edge(face):
    return [[face[i], face[(i+1)%3]] for i in range(face.size)]

def subdivision_loop(mesh: trimesh.Trimesh, iterations=1):
    # print(mesh.vertices)
    # print(mesh.faces)
    new_verticies = []
    new_faces = []
    # for each face 
    for face in mesh.faces:
        # create a list call edges
        edges = face_to_edge(face)
        vertices = [mesh.vertices[v] for v in face]
        for edge in edges:
            new_verticies.append((mesh.vertices[edge[0]] + mesh.vertices[edge[1]])/2)
        # we need to replace the face with 4 new faces
        # the first face is the original face
        
        print(edges)
        print(vertices)            


    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    return mesh

def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    return mesh

if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('../assets/cube.obj', process=False)
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')

    # apply loop subdivision over the loaded mesh
    # mesh_subdivided = mesh.subdivide_loop(iterations=1)
    
    # TODO: implement your own loop subdivision here
    mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
    # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    # mesh_subdivided.export('../assets/cube_subdivided.obj')
    
    # quadratic error mesh decimation
    mesh_decimated = mesh.simplify_quadric_decimation(8)
    
    # TODO: implement your own quadratic error mesh decimation here
    # mesh_decimated = simplify_quadric_error(mesh, face_count=1)
    
    # print the new mesh information and save the mesh
    print(f'Decimated Mesh Info: {mesh_decimated}')
    # mesh_decimated.export('../assets/cube_decimated.obj')