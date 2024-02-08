import trimesh
import numpy as np

def face_to_edge(face):
    return [[face[i], face[(i+1)%3]] for i in range(face.size)]

def subdivision_loop(mesh: trimesh.Trimesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    edges = mesh.edges

    new_vertices = {}
    # for each edge 
    for edge in edges:
        a, b = edge
        # Find vertices connected to a but not b
        a_adj = {e[1] for e in edges if e[0] == a and e[1] != b}.union({e[0] for e in edges if e[1] == a and e[0] != b})
        # Find vertices connected to b but not a
        b_adj = {e[1] for e in edges if e[0] == b and e[1] != a}.union({e[0] for e in edges if e[1] == b and e[0] != a})
        # Find common vertices between a_adj and b_adj
        common = list(a_adj.intersection(b_adj))
        if len(common) == 2:
            m = vertices[a] * 3/8 + vertices[b] * 3/8 + vertices[common[0]] * 1/8 + vertices[common[1]] * 1/8
        else:
            m = vertices[a] * 1/2 + vertices[b] * 1/2
        print(a,b,m)
        vertices = np.vstack([vertices, m])
        if a < b:
            new_vertices[(a,b)] = len(vertices) - 1
        else:
            new_vertices[(b,a)] = len(vertices) - 1
    print(vertices)
    print(new_vertices)
    print(vertices[new_vertices[(6,7)]])
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
    