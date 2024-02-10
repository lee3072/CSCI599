import trimesh
import numpy as np

def get_new_vertex_index(new_vertices, a, b):
    return new_vertices[(a,b) if a < b else (b,a)]

def subdivision_loop(mesh: trimesh.Trimesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    edges = mesh.edges.copy()
    
    v_e = {}
    v_v = {}
    for edge in edges:
        v1, v2 = edge
        if v1 not in v_e:
            v_e[v1] = [edge]
        else:
            v_e[v1] += [edge]
        if v2 not in v_e:
            v_e[v2] = [edge]
        else:
            v_e[v2] += [edge]
        # if v1 not in v_v:
        if v1 not in v_v:
            v_v[v1] = [v2]
        else:
            v_v[v1] += [v2]
        
    print(v_e)
    print(v_v) 
    
    # print(f'Vertices: {vertices.shape[0]}')
    # print(f'Faces: {faces.shape[0]}')
    # print(f'Edges: {edges.shape[0]}')
    # print(faces)

    # new_vertices = {}
    # # for each edge 
    # for edge in edges:
    #     a, b = edge
    #     # Find vertices connected to a
    #     a_adj = {e[1] for e in edges if e[0] == a and e[1] != b}.union({e[0] for e in edges if e[1] == a and e[0] != b})
    #     # Find vertices connected to b
    #     b_adj = {e[1] for e in edges if e[0] == b and e[1] != a}.union({e[0] for e in edges if e[1] == b and e[0] != a})
    #     # Find common vertices between a_adj and b_adj
    #     common = list(a_adj.intersection(b_adj))
    #     # if inboundary vertex, m = 3/4 * a + 1/8 * b + 1/8 * c
    #     # if on boundary vertex, m = 1/2 * a + 1/2 * b
    #     if len(common) == 2:
    #         m = vertices[a] * 3/8 + vertices[b] * 3/8 + vertices[common[0]] * 1/8 + vertices[common[1]] * 1/8
    #     else:
    #         m = vertices[a] * 1/2 + vertices[b] * 1/2
    #     # print(a,b,m)
    #     vertices = np.vstack([vertices, m])
    #     new_vertices[(a,b) if a < b else (b,a)] = len(vertices) - 1

    # # for each face
    # for f in range(faces.shape[0]):
    #     face = faces[f].copy()
    #     # create new faces with one even vertex and two odd vertices connected with it
    #     # replace the old face with three odd vertices
    #     for i in range(face.shape[0]+1):
    #         a, b, c = face[i%3], face[(i+1)%3], face[(i+2)%3]
    #         v1 = get_new_vertex_index(new_vertices, a, b)
    #         v2 = get_new_vertex_index(new_vertices, a, c) if i == face.shape[0] else b
    #         v3 = get_new_vertex_index(new_vertices, b, c)
    #         if i == face.shape[0]: 
    #             faces[f] = [v1, v2, v3]
    #         else: 
    #             faces = np.vstack([faces, [v1, v2, v3]])
                    
    # # for even vertices find all connected vertices and calculate the new position
    # for i in range(mesh.vertices.shape[0]):
    #     break
    # # print(vertices)
    # # print(new_vertices)
    # # print(vertices[new_vertices[(6,7)]])
    return mesh


if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('../assets/cube.obj', process=False)
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')

    # TODO: implement your own loop subdivision here
    mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
    # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    # mesh_subdivided.export('../assets/cube_subdivided.obj')
    