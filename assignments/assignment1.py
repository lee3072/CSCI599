import trimesh
import numpy as np
from icecream import ic
# import sorted dict
from sortedcontainers import SortedDict


class SortedDictSet:
    def __init__(self):
        self.sorted_dict = SortedDict()  # SortedDict to keep key sorted
        self.values_dict = {}  # Regular dictionary to store value to key mapping

    def add(self, key, value):
        self.values_dict[value] = key
        self.sorted_dict.setdefault(key, set()).add(value)

    def removeKey(self, key):
        values = self.sorted_dict.pop(key)
        for value in values:
            self.values_dict.pop(value)

    def removeValue(self, value):
        key = self.values_dict.pop(value)
        self.sorted_dict[key].remove(value)
        if len(self.sorted_dict[key]) == 0:
            self.sorted_dict.pop(key)

    def popValue(self):
        _, values = self.sorted_dict.peekitem(0)
        value = values.pop()
        if len(values) == 0:
            self.sorted_dict.popitem(0)
        self.values_dict.pop(value)
        return value

    def updateKey(self, value, new_key):
        self.removeValue(value)
        self.add(new_key, value)

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
    
    v_f = {}    # Vertex to Connected Faces Dictionary      # USED: quadric error for each vertex calculation
    v_k = {}    # Vertex to Quadric Error Matrix Dictionary # USED: quadric error for each edge calculation
    c_e = SortedDictSet() # Cost to Edge Dictionary            # USED: find edge to collapse
    e_k = {}    # Edge to Quadric Error Matrix Dictionary   # USED: collapse edge
    e_m = {}    # Edge to New Vertex Position Dictionary    # USED: collapse edge
    v_e = {}    # Vertex to Connected Edges Dictionary      # USED: collapse edge update edges

    for e in range(edges.shape[0]):
        edge = edges[e]
        # Sort the edge to make sure the smaller vertex is first
        edge = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
        for v in edge: v_e.setdefault(v, set()).add(edge)
        edges[e] = edge

    ic.enable()
    
    # create vertex to connected edges dictionary
    # create vertex to connected faces dictionary
    for f in range(faces.shape[0]):
        for i in range(faces[f].shape[0]):
            v_f.setdefault(faces[f][i], set()).add(f)

    def quadric_error_vertex(v):
        k = np.zeros((4,4))
        for f in v_f[v]:
            n = normals[f]
            # calculate the distance from the plane to the origin
            d = -np.dot(n, vertices[v])
            # calculate the homogeneous coordinates of the plane
            hc = np.append(n, d)
            # calculate the quadric error matrix
            k += np.outer(hc, hc)
        return k
    
    def quadric_error_edge(edge):
        v1, v2 = edge
        k = v_k[v1] + v_k[v2]
        # find the m that minimizes the quadric error
        # since k is a 4x4 matrix and the orthogonal digit of m is 1, 
        # we can solve for m by B and w where B is the 3x3 matrix and w is the 3x1 vector of m
        B = k[:3,:3]
        w = k[:3,3]
        m = np.dot(-np.linalg.inv(B), w)
        # add orthogonal digit to m
        M = np.append(m, 1)
        # store cost of collapsing the edge m^T * ek * m
        cost = np.transpose(M).dot(k).dot(M)
        return k, m, cost

    # calculate the quadric error for each vertex
    for v in range(vertices.shape[0]): v_k[v] = quadric_error_vertex(v)

    # calculate the quadric error for each edge
    for e in range(edges.shape[0]):
        edge = tuple(edges[e])
        e_k[edge], e_m[edge], cost = quadric_error_edge(edge)
        c_e.add(key=cost, value=edge)

    # ic.disable()
    ic(vertices)                                                # DONE 1: replace v1 position with m
                                                                # TODO 2: remove v2 from vertices             
    ic(faces)                                                   # DONE 1: update v2 to v1, 
                                                                # DONE 2: remove faces connected to both v1 and v2
    ic(edges)                                                   # DONE: replace v2 with v1 for all dependent edges
    ic(normals)                                                 # DONE 1: update normal for faces connected to v1 or v2, but not both 
                                                                # DONE 2: remove normals dependent on faces connected to both v1 and v2
    ic(v_f) # USED: quadric error for each vertex calculation   # DONE: removed faces connected to both v1 and v2
    ic(v_k) # USED: quadric error for each edge calculation     # 
    ic(c_e) # USED: find edge to collapse                       # DONE 1: removed e from c_e
                                                                # TODO 2: update cost for all dependent edges
    ic(e_k) # USED: collapse edge                               # DONE 1: removed e from e_k
    ic(e_m) # USED: collapse edge                               # DONE 1: removed e from e_m
                                                                # TODO 2: update m for all dependent edges              
    ic(v_e) # USED: collapse edge update edges                  # DONE: if it is a collapsed edge, remove it from v_e[v1]
    ic.enable()
    ic(edges)
    
    removed_faces = set()
    # collapse edges until the target face count is reached
    while faces.shape[0] - len(removed_faces) > face_count:
        # remove the edge with the smallest cost
        edge = c_e.popValue()
        v1, v2 = edge
        ic(edge, v1, v2)

        # replace v1 position with m
        vertices[v1] = e_m[edge]

        # remove edge from e_m
        e_m.pop(edge)
        # remove edge from e_k
        e_k.pop(edge)

        # update normal for faces connected to v1 or v2, but not both
        common_faces = []
        for f in v_f[v1]:
            face = faces[f]
            if v2 not in face:
                # calculate the normal of the face
                normals[f] = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
            else:
                common_faces.append(f)
                removed_faces.add(f)
        
        for f in v_f[v2]:
            face = faces[f]
            if v1 not in face:
                # ic(face)
                # replace v2 with v1 for all dependent faces 
                for i in range(face.shape[0]): 
                    if face[i] == v2: face[i] = v1
                ic(face)
                # calculate the normal of the face
                normals[f] = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
                v_f[v1].add(f)
        # remove v2 from v_f
        v_f.pop(v2)
        # remove v2 from v_k
        v_k.pop(v2)

        # remove faces and dependent normals connected to both v1 and v2 
        for f in common_faces:
            # faces = np.delete(faces, f, axis=0)
            # normals = np.delete(normals, f, axis=0)
            # remove original faces from the vertex to connected faces dictionary
            if v1 in v_f:
                if f in v_f[v1]: v_f[v1].remove(f)
            for i in range(faces.shape[0]):
                edge = tuple(order_edge(faces[f][i%3], faces[f][(i+1)%3]))
                ic(edge, v1, v2, v_e[v1], v_e[v2])
                if v1 in edge: v_e[v1].discard(edge)
                if v2 in edge: v_e[v2].discard(edge)

        # ic(v_e[v2],v2)
        # replace v2 with v1 for all dependent edges 
        for edge in v_e[v2]:
            # if it is a collapsed edge, remove it from v_e[v1] and skip
            if edge[0] == v1 or edge[1] == v1:
                v_e[v1].remove(edge)
                ic('remove',edge)
                continue 
            if edge[0] == v2: 
                v_e[edge[1]].remove(edge)
                edge = (v1, edge[1])
                v_e[edge[1]].add(edge)
            if edge[1] == v2: 
                v_e[edge[0]].remove(edge)
                edge = (edge[0], v1)
                v_e[edge[0]].add(edge)
            ic('add', edge)
            v_e[v1].add(edge)
        v_e.pop(v2)

        # update quadric error for v1
        v_k[v1] = quadric_error_vertex(v1)

        # update quadric error for all dependent vertices
        for edge in v_e[v1]:
            # find vertex not equal to v1
            v = edge[0] if edge[0] != v1 else edge[1]
            # calculate the quadric error for vertex v
            v_k[v] = quadric_error_vertex(v)

        # update quadric error for all edges connected to dependent vertices
        for edge in v_e[v1]:
            # find vertex not equal to v1
            v = edge[0] if edge[0] != v1 else edge[1]
            for con_edge in v_e[v]:
                ic(con_edge)
                e_k[con_edge], e_m[con_edge], cost = quadric_error_edge(con_edge)
                c_e.updateKey(value=con_edge, new_key=cost)
        ic('a')
    ic('test')
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
    mesh_decimated = simplify_quadric_error(mesh, face_count=3)
    
    # print the new mesh information and save the mesh
    ic(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('./assets/cube_decimated.obj')