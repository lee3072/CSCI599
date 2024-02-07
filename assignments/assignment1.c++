#include <iostream>
#include <fstream>
#include <string>
#include <vector>

struct Vertex {
    float x, y, z;
};

struct Face {
    int v1, v2, v3;
};

struct H_Vertex {
    float x, y, z;
    H_Edge *H_Edge;
};

struct H_Edge {
    H_Vertex *H_Vertex;
    H_Edge *pair;
    H_Face *H_Face;
    H_Edge *next;
};

struct H_Face {
    H_Edge *H_Edge;
};

// Extract only verticies and faces from .obj file
// convert to half-H_Edge data structure
void importObj(std::string filename) {
    std::vector<H_Vertex> H_Verticies;
    std::vector<H_Face> H_Faces;
    std::vector<H_Edge> H_Edges;

    std::vector<Vertex> verticies;
    std::vector<Face> faces;

    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (line[0] == 'v') {
                // Extract verticies
                Vertex vertex;
                sscanf(line.c_str(), "v %f %f %f", &Vertex.x, &Vertex.y, &Vertex.z);
                verticies.push_back(vertex);
            } else if (line[0] == 'f') {
                // Extract H_Faces
                Face face;
                sscanf(line.c_str(), "f %d %d %d", &face.v1, &face.v2, &face.v3);
                faces.push_back(face);
            }
        }
        file.close();
    }

    
    // Extract verticies and H_Faces
    // Convert to half-H_Edge data structure

}

int main() {
    importObj("cube.obj");
    return 0;
}