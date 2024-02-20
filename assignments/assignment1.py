import trimesh
import numpy as np
from icecream import ic
# import sorted dict
from sortedcontainers import SortedDict


ic.configureOutput(includeContext=True, prefix='DEBUG| ')
ic.disable()

def order_edge(a, b):
    return (a, b) if a < b else (b, a)


def subdivision_loop(mesh: trimesh.Trimesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    for i in range(iterations):
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        edges = mesh.edges.copy()
        
        v_e = {}    # Vertex to Connected Edges Dictionary
        v_v = {}    # Vertex to Connected Vertices Dictionary
        e_f = {}    # Edge to Connected Faces Dictionary
        e_nv = {}   # Edge to New Vertex Dictionary

        for e in range(edges.shape[0]):
            edge = edges[e]
            # Sort the edge to make sure the smaller vertex is first
            edge = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
            edges[e] = edge
            # Add the edge to the vertex to connected edge dictionary
            v1, v2 = edge
            v_e.setdefault(v1, set()).add(edge)
            v_e.setdefault(v2, set()).add(edge)
            # Add the vertex to the vertex to connected vertex dictionary
            v_v.setdefault(v1, set()).add(v2)
            v_v.setdefault(v2, set()).add(v1)

        ic(f'# Vertices: {vertices.shape[0]}')
        ic(f'# Faces: {faces.shape[0]}')
        ic(f'# Edges: {edges.shape[0]}')
        ic(f'Vertex to Connected Edges:\n{v_e}\n')
        ic(f'Vertex to Connected Vertices:\n{v_v}\n')

        for edge in edges:
            a, b = edge
            common = list(set(v_v[a]).intersection(set(v_v[b])))
            if len(common) == 2:
                m = vertices[a] * 3/8 + vertices[b] * 3/8 + vertices[common[0]] * 1/8 + vertices[common[1]] * 1/8
            else:
                m = vertices[a] * 1/2 + vertices[b] * 1/2
            vertices = np.vstack([vertices, m])
            # Add the new vertex to the edge to new vertex dictionary
            e_nv[tuple(edge)] = len(vertices) - 1

        ic(f'Edge to New Vertices:\n{e_nv}\n')

        # Establish connectivity with the new vertices
        for f in range(mesh.faces.shape[0]):
            face = faces[f].copy()
            # create new faces with one even vertex and two odd vertices connected with it
            # replace the old face with three odd vertices
            for i in range(face.shape[0]+1):
                a, b, c = face[i%3], face[(i+1)%3], face[(i+2)%3]
                v1 = e_nv[order_edge(a, b)]
                v2 = e_nv[order_edge(a, c)] if i == face.shape[0] else b
                v3 = e_nv[order_edge(b, c)]
                if i == face.shape[0]: 
                    # replace the old face with three odd vertices
                    nf = [v2, v1, v3]
                    faces[f] = nf # Make Sure Normal is Correct
                    # remove original vertices from the vertex to connected vertex dictionary
                    if a in v_v:
                        if b in v_v[a]: v_v[a].remove(b)
                        if c in v_v[a]: v_v[a].remove(c)
                    if b in v_v:
                        if a in v_v[b]: v_v[b].remove(a)
                        if c in v_v[b]: v_v[b].remove(c)
                    if c in v_v:
                        if a in v_v[c]: v_v[c].remove(a)
                        if b in v_v[c]: v_v[c].remove(b)
                    # remove original edges from the vertex to connected edge dictionary
                    if a in v_e:
                        if order_edge(a, b) in v_e[a]: v_e[a].remove(order_edge(a, b))
                        if order_edge(a, c) in v_e[a]: v_e[a].remove(order_edge(a, c))
                    if b in v_e:
                        if order_edge(a, b) in v_e[b]: v_e[b].remove(order_edge(a, b))
                        if order_edge(b, c) in v_e[b]: v_e[b].remove(order_edge(b, c))
                    if c in v_e:
                        if order_edge(a, c) in v_e[c]: v_e[c].remove(order_edge(a, c))
                        if order_edge(b, c) in v_e[c]: v_e[c].remove(order_edge(b, c))
                else: 
                    # create new faces with one even vertex and two odd vertices connected with it
                    nf = [v1, v2, v3]
                    faces = np.vstack([faces, nf])
                # add new edges to the vertex to connected vertex dictionary
                v_v.setdefault(v1, set()).add(v2)
                v_v.setdefault(v1, set()).add(v3)
                v_v.setdefault(v2, set()).add(v1)
                v_v.setdefault(v2, set()).add(v3)
                v_v.setdefault(v3, set()).add(v1)
                v_v.setdefault(v3, set()).add(v2)
                # add new edges to the vertex to connected edge dictionary
                v_e.setdefault(v1, set()).add(order_edge(v1, v2))
                v_e.setdefault(v1, set()).add(order_edge(v1, v3))
                v_e.setdefault(v2, set()).add(order_edge(v1, v2))
                v_e.setdefault(v2, set()).add(order_edge(v2, v3))
                v_e.setdefault(v3, set()).add(order_edge(v1, v3))
                v_e.setdefault(v3, set()).add(order_edge(v2, v3))
                e_f.setdefault(tuple(order_edge(v1, v2)), []).append(nf)
                e_f.setdefault(tuple(order_edge(v1, v3)), []).append(nf)
                e_f.setdefault(tuple(order_edge(v2, v3)), []).append(nf)


        ic(f'Updated Vertex to Connected Vertices:\n{v_v}\n')
        ic(f'Updated Vertex to Connected Edges:\n{v_e}\n')
        ic(f'Updated Faces:\n{faces}\n')
        ic(f'Edge to Connected Faces:\n{e_f}\n')

        # for even vertices find all connected vertices and calculate the new position
        for i in range(mesh.vertices.shape[0]):
            v = vertices[i]
            connected = v_v[i]
            boundary = False
            for c in connected:
                if len(e_f[order_edge(i,c)]) == 1:
                    boundary = True
            if boundary:
                avg = np.sum([vertices[j] for j in connected], axis=0) / len(connected)
                v = 1/4 * avg + 3/4 * v
            else:
                beta = 1/len(connected) * (5/8 - (3/8 + 1/4 * np.cos(2 * np.pi / len(connected)))**2)
                v = v * (1-len(connected)*beta) + np.sum([vertices[j] for j in connected], axis=0) * beta
            vertices[i] = v
        ic(f'Final Vertices:\n{vertices}\n')
        ic(f'Final Faces:\n{faces}\n')

        # Create the new mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)


    return mesh


def simplify_quadric_error(mesh: trimesh.Trimesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    if mesh.faces.shape[0] <= face_count:
        return mesh

    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    normals = mesh.face_normals.copy()
    edges = mesh.edges.copy()
    
    for e in range(edges.shape[0]):
        edge = edges[e]
        # Sort the edge to make sure the smaller vertex is first
        edge = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
        edges[e] = edge

    ic.enable()

    v_e = {}    # Vertex to Connected Edges Dictionary
    v_f = {}    # Vertex to Connected Faces Dictionary
    e_f = {}    # Edge to Connected Faces Dictionary
    v_k = {}    # Vertex to Quadric Error Matrix Dictionary
    e_k = {}    # Edge to Quadric Error Matrix Dictionary
    e_m = {}    # Edge to New Vertex Dictionary
    c_e = SortedDict() # Cost to Edge Dictionary
    # make faces with normals
    for f in range(mesh.faces.shape[0]):
        
        # we also need to make edge to face dictionary
        for i in range(3):
            e = order_edge(faces[f,i], faces[f,(i+1)%3])
            v_e.setdefault(faces[f,i], set()).add(e)
            e_f.setdefault(e, set()).add(f)
            v_f.setdefault(faces[f,i], set()).add(f)

    # calculate the quadric error for each vertex
    for v in range(vertices.shape[0]):
        k = np.zeros((4,4))
        for f in v_f[v]:
            n = normals[f]
            # calculate the distance from the plane to the origin
            d = -np.dot(n, vertices[v])
            # calculate the homogeneous coordinates of the plane
            hc = np.append(n, d)
            # calculate the quadric error matrix
            k += np.outer(hc, hc)
        v_k[v] = k
    
    for e in range(edges.shape[0]):
        v1, v2 = edges[e]
        e_k[e] = v_k[v1] + v_k[v2]
        # find the m that minimizes the quadric error
        # Deriative of the quadric error with respect to m. 
        # https://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/resources/Matrix_derivatives_cribsheet.pdf
        # dm of (m^T * e_k[e] * m) = 2 * e_k[e] * m
        # However, since e_k[e] is a 4x4 matrix and the orthogonal digit of m is 1, 
        # we can solve for m by B and w where B is the 3x3 matrix and w is the 3x1 vector of m
        B = e_k[e][:3,:3]
        w = e_k[e][:3,3]
        m = np.dot(-np.linalg.inv(B), w)
        # add orthogonal digit to m
        e_m[e] = m
        m = np.append(m, 1)
        ic(e_k[e], B, w, m)
        # store cost of collapsing the edge m^T * e_k[e] * m
        cost = np.transpose(m).dot(e_k[e]).dot(m)
        c_e.setdefault(cost, set()).add(e)

    ic(v_f)
    ic(e_f)
    ic(v_k)
    ic(e_k)
    ic(c_e)
    ic(e_m)

    # remove the edge with the smallest cost
    for i in range(mesh.faces.shape[0] - face_count): # need to fix
        es = c_e.peekitem(0)[1]
        e = es.pop()
        if len(es) == 0:
            c_e.popitem(0)
        else:
            c_e.peekitem(0)[1].update(es) 
        v1, v2 = edges[e]
        m = e_m[e]

        # replace v1 with m in the vertices
        vertices[v1] = m

        # for every face that contains v2, replace them with the v1
        for f in list(v_f[v2]):
            face = faces[f]
            if v1 in face:
                ic(f'Face {f} contains {v1}')
                # Remove face with merging edge
                faces = np.delete(faces, f, axis=0)
                v_f[v1].remove(f)
                v_f[v2].remove(f)
                # Update e_f for edges of this face
                for edge in [(face[i], face[(i+1)%3]) for i in range(3)]:
                    edge = order_edge(edge[0], edge[1])
                    if e_f[edge]:
                        ic(e_f[edge])
                        e_f[edge].remove(f)
            else:
                # Replace v2 with v1 in the face
                faces[f] = [v1 if x == v2 else x for x in face]
                v_f[v1].add(f)
                v_f[v2].remove(f)
        
        # update the vertex to connected faces dictionary
        v_f[v1].update(v_f[v2])

        # # for every edge that contains v2, replace them with the v1
        # for e in v_e[v2]:
        #     if e[0] == v2:
        #         e[0] = v1


    ic(c_e)


    return mesh

if __name__ == '__main__':
    # Load mesh and ic information
    # mesh = trimesh.load_mesh('./assets/cube.obj', process=False)
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    ic(f'Mesh Info: {mesh}')

    # TODO: implement your own loop subdivision here
    # mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
    # # ic the new mesh information and save the mesh
    # ic(f'Subdivided Mesh Info: {mesh_subdivided}')
    # mesh_subdivided.export('./assets/cube_subdivided.obj')

    # TODO: implement your own quadratic error mesh decimation here
    mesh_decimated = simplify_quadric_error(mesh, face_count=1)
    
    # print the new mesh information and save the mesh
    ic(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('./assets/cube_decimated.obj')