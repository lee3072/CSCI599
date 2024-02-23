import trimesh
import numpy as np
from icecream import ic
# import sorted dict
from sortedcontainers import SortedDict


class DictSet:
    def __init__(self):    
        self.key_dict = {}
        self.value_dict = {}
        self.key_dependent_dict = {}
        self.value_dependent_dict = {}

    def add(self, key, value, key_dependent=None, value_dependent=None):
        self.key_dict.setdefault(key, set()).add(value)
        self.value_dict.setdefault(value, set()).add(key)
        self.key_dependent_dict[key] = key_dependent
        self.value_dependent_dict[value] = value_dependent
    
    def removeKey(self, key):
        if key not in self.key_dict: return
        values = self.key_dict.pop(key)
        vd = {}
        for value in values:
            self.value_dict[value].discard(key)
            if len(self.value_dict[value]) == 0: self.value_dict.pop(value)
            vd[value] = self.value_dependent_dict[value]
        return vd, self.key_dependent_dict[key]

    def removeValue(self, value):
        keys = self.value_dict.pop(value)
        kd = {}
        for key in keys:
            self.key_dict[key].discard(value)
            if len(self.key_dict[key]) == 0:self.key_dict.pop(key)
            kd[key] = self.key_dependent_dict[key]
        return kd, self.value_dependent_dict[value]
    
    def getByKey(self, key):
        values = self.key_dict[key]
        vd = {}
        for value in values:
            vd[value] = self.value_dependent_dict[value]
        return vd, self.key_dependent_dict[key]
    
    def getByValue(self, value):
        keys = self.value_dict[value]
        kd = {}
        for key in keys:
            kd[key] = self.key_dependent_dict[key]
        return kd, self.value_dependent_dict[value]

    def updateKeyDependent(self, key, new_key_dependent):
        self.key_dependent_dict[key] = new_key_dependent

class SortedDictSet:
    def __init__(self):
        self.sorted_dict = SortedDict()  # SortedDict to keep key sorted
        self.value_dict = {}            # Regular dictionary to store value to key mapping
        self.value_dependent_dict = {}

    def add(self, key, value, value_dependent=None):
        self.sorted_dict.setdefault(key, set()).add(value)
        self.value_dict[value] = key
        self.value_dependent_dict[value] = value_dependent

    def removeKey(self, key):
        values = self.sorted_dict.pop(key)
        vd = {}
        for value in values:
            self.value_dict.pop(value)
            vd[value] = self.value_dependent_dict.pop(value)
        return vd

    def removeValue(self, value):
        if value not in self.value_dict: return
        key = self.value_dict.pop(value)
        self.sorted_dict[key].discard(value)
        if len(self.sorted_dict[key]) == 0: self.sorted_dict.pop(key)
        return key, self.value_dependent_dict.pop(value)

    def popFirstValue(self):
        key, values = self.sorted_dict.peekitem(0)
        value = values.pop()
        if len(values) == 0: self.sorted_dict.popitem(0)
        self.value_dict.pop(value)
        return key, value, self.value_dependent_dict.pop(value)

    def updateKeyForValue(self, value, new_key):
        _, dependent = self.removeValue(value)
        self.add(new_key, value, dependent)

    def updateValue(self, value, new_value):
        if value not in self.value_dict: return
        key, dependent = self.removeValue(value)
        self.add(key, new_value, dependent)

    def updateDependent(self, value, new_value_dependent):
        self.value_dependent_dict[value] = new_value_dependent

    def getDependent(self, value):
        return self.dependent_dict[value]

    def getByKey(self, key):
        values = self.sorted_dict[key]
        value_dependent = {}
        for value in values:
            value_dependent[value] = self.dependent_dict[value]
        return value_dependent
    
    def getByValue(self, value):
        key = self.value_dict[value]
        return key, self.dependent_dict[value]

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


# this class initialize with a trimesh object
class CustomTriMesh:
    # private method to calculate the quadric error for each vertex
    def __quadric_error_vertex(self, v):
        k = np.zeros((4,4))
        faces_normal,_ = self.faceNormal_vertex.getByValue(v)
        for n in faces_normal.values():
            # calculate the distance from the plane to the origin
            d = -np.dot(n, self.vertices[v])
            # calculate the homogeneous coordinates of the plane
            hc = np.append(n, d)
            # calculate the quadric error matrix
            k += np.outer(hc, hc)
        return k
    
    # private method to calculate the m and cost for each edge
    def __minimum_cost(self, v1, v2):
        k = self.v_k[v1] + self.v_k[v2]
        # find the m that minimizes the quadric error
        # since k is a 4x4 matrix and the orthogonal digit of m is 1, 
        # we can solve for m by B and w where B is the 3x3 matrix and w is the 3x1 vector of m
        B = k[:3,:3]
        w = k[:3,3]
        # Calculate the determinant of B
        det = np.linalg.det(B)
        threshold = 1e-6
        if abs(det) < threshold:
            m = np.dot(-np.linalg.pinv(B), w)
        else:
            m = np.dot(-np.linalg.inv(B), w)
        # add orthogonal digit to m
        M = np.append(m, 1)
        # store cost of collapsing the edge m^T * ek * m
        cost = np.transpose(M).dot(k).dot(M)
        return m, cost

    def __init__(self, mesh: trimesh.Trimesh, face_count):
        self.face_count = face_count
        # self.faces = mesh.faces.copy()
        # self.normals = mesh.face_normals.copy()

        self.vertices = mesh.vertices.copy()
        self.v_v = {}                               # Vertex Index to Connected Vertices Index Dictionary
        self.faceNormal_vertex = DictSet()          # Face to Vertex Index, Vertex Index to Face, with Face dependent Normal Dictionary 
        self.v_k = {}                               # Vertex Index to Quadric Error Matrix Dictionary
        self.cost_EdgeM = SortedDictSet()           # Cost to Edge, Edge to Cost, with dependent M Dictionary

        for f in range(mesh.faces.shape[0]):
            face = tuple(mesh.faces[f])
            normal = tuple(mesh.face_normals[f])
            for i in range(len(face)):
                ic(face, face[i], normal)
                self.faceNormal_vertex.add(key=face, value=face[i], key_dependent=normal)
                self.v_v.setdefault(face[i], set()).add(face[(i+1)%3])
                self.v_v.setdefault(face[i], set()).add(face[(i+2)%3])
        
        ic(self.v_v)
        # ic.disable()
        ic(self.faceNormal_vertex.key_dict, self.faceNormal_vertex.value_dict, self.faceNormal_vertex.key_dependent_dict, self.faceNormal_vertex.value_dependent_dict)
        # ic.enable()

        for v in range(self.vertices.shape[0]): self.v_k[v] = self.__quadric_error_vertex(v)
        
        ic(self.v_k)
        unique_edges = set()
        for key in self.v_v:
            values = self.v_v[key]
            for value in values:
                v1, v2 = key, value
                edge = order_edge(v1, v2)
                if edge not in unique_edges:
                    unique_edges.add(edge)
                    m, cost = self.__minimum_cost(v1,v2)
                    self.cost_EdgeM.add(key=cost, value=edge, value_dependent=m)
        ic(self.cost_EdgeM.sorted_dict, self.cost_EdgeM.value_dict, self.cost_EdgeM.value_dependent_dict)
        while len(self.faceNormal_vertex.key_dict) > face_count:
            self.reduce_face()


    def reduce_face(self):
        cost, edge, m = self.cost_EdgeM.popFirstValue()
        ic(cost, edge, m)
        v1, v2 = edge
        # replace v1 position with m
        self.vertices[v1] = m
        # remove v2 from Vertex Index to Quadric Error Matrix Dictionary
        self.v_k.pop(v2)
        # remove v2 from Vertex Index to Connected Vertices Index Dictionary
        v2_connected_vertex = self.v_v.pop(v2)
        # replace v2 with v1 for all dependent vertices
        for v in v2_connected_vertex:
            self.v_v[v].discard(v2)
            self.v_v[v].add(v1)
            # add v to connected vertices for v1
            self.v_v[v1].add(v)
            # ic(v, v1, v2, order_edge(v2, v))
            if v == v1:
                # ic('removing value', order_edge(v2, v), self.cost_EdgeM.value_dependent_dict)
                self.cost_EdgeM.removeValue(value=order_edge(v2, v))
                # ic(self.cost_EdgeM.value_dependent_dict)
            else:
                # make sure replacement is also effective on cost_EdgeM
                self.cost_EdgeM.updateValue(value=order_edge(v2, v), new_value=order_edge(v1, v))
            
        # replace v2 with v1 for faces, and delete faces connected to both v1 and v2
        v1_faces_normal, _ = self.faceNormal_vertex.getByValue(v1)
        v2_faces_normal, _ = self.faceNormal_vertex.getByValue(v2)
        v1_faces, v1_normals = v1_faces_normal.keys(), v1_faces_normal.values()
        for face, normal in v2_faces_normal.items():
            self.faceNormal_vertex.removeKey(face)
            if face not in v1_faces:    # only add faces that are not connected to both v1 and v2
                self.faceNormal_vertex.add(key=face, value=v1, key_dependent=normal)

        # recalculate normal for vertices connected to v1
        for v in self.v_v[v1]:
            # get all faces connected to v
            face_normal, _ = self.faceNormal_vertex.getByValue(v)
            for face in face_normal.keys():
                normal = np.cross(self.vertices[face[1]] - self.vertices[face[0]], self.vertices[face[2]] - self.vertices[face[0]])
                self.faceNormal_vertex.updateKeyDependent(key=face, new_key_dependent=normal)

        # recalculate quadric error for v1
        self.v_k[v1] = self.__quadric_error_vertex(v1)
        # recalculate quadric error for vertices connected to v1
        for v in self.v_v[v1]:
            self.v_k[v] = self.__quadric_error_vertex(v)
        # recalculate cost for edges connected to vertices connected to v1
        for v in self.v_v[v1]:
            for cv in self.v_v[v]:
                if v == cv: continue
                edge = order_edge(v, cv)
                m, cost = self.__minimum_cost(v, cv)
                self.cost_EdgeM.removeValue(value=edge)
                self.cost_EdgeM.add(key=cost, value=edge, value_dependent=m)
  
        ic(self.vertices, self.faceNormal_vertex.value_dict, self.faceNormal_vertex.value_dependent_dict, self.v_k, self.cost_EdgeM.sorted_dict, self.cost_EdgeM.value_dict, self.cost_EdgeM.value_dependent_dict)
        return

    
        

def simplify_quadric_error(mesh: trimesh.Trimesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    ic.enable()
    CustomTriMesh(mesh, face_count)

    ic('test')
    exit()
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
    # ic.enable()
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
    mesh_decimated = simplify_quadric_error(mesh, face_count=5)
    
    # print the new mesh information and save the mesh
    ic(f'Decimated Mesh Info: {mesh_decimated}')
    mesh_decimated.export('./assets/cube_decimated.obj')