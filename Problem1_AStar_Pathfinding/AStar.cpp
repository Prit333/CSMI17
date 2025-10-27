#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace std;

// --- Grid and Node Setup ---
#define GRID_SIZE 50
#define OBSTACLE_DENSITY 0.25 // 25%
#define NUM_TRIALS 100

// Represents a single cell in the grid
struct Node {
    int x, y; // Coordinates
    double f, g, h; // A* values
    Node* parent;

    Node(int x_ = 0, int y_ = 0, double g_ = 0, double h_ = 0, Node* p = nullptr)
        : x(x_), y(y_), g(g_), h(h_), f(g_ + h_), parent(p) {}

    // Comparison for the priority queue (min-heap)
    bool operator>(const Node& other) const {
        return f > other.f;
    }
};

// --- Heuristic Functions ---
double heuristic_manhattan(int x, int y, int gx, int gy) {
    return abs(x - gx) + abs(y - gy);
}

double heuristic_euclidean(int x, int y, int gx, int gy) {
    return sqrt(pow(x - gx, 2) + pow(y - gy, 2));
}

double heuristic_zero(int x, int y, int gx, int gy) {
    return 0.0;
}

// --- A* Algorithm ---
// Function pointer for passing the heuristic
using HeuristicFunc = double(*)(int, int, int, int);

struct AStarResult {
    int pathLength = 0;
    int nodesExpanded = 0;
    double executionTimeMs = 0;
};

AStarResult a_star_search(const vector<vector<int>>& grid, int startX, int startY, int goalX, int goalY, HeuristicFunc heuristic) {
    auto startTime = chrono::high_resolution_clock::now();

    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0}; // 4-directional movement

    priority_queue<Node, vector<Node>, greater<Node>> openSet;
    vector<vector<bool>> closedSet(GRID_SIZE, vector<bool>(GRID_SIZE, false));

    Node* startNode = new Node(startX, startY, 0, heuristic(startX, startY, goalX, goalY));
    openSet.push(*startNode);

    AStarResult result;
    int nodesExpanded = 0;

    while (!openSet.empty()) {
        Node current = openSet.top();
        openSet.pop();
        nodesExpanded++;

        if (current.x == goalX && current.y == goalY) {
            // Path found
            auto endTime = chrono::high_resolution_clock::now();
            result.executionTimeMs = chrono::duration<double, milli>(endTime - startTime).count();
            result.nodesExpanded = nodesExpanded;

            // Reconstruct path to get length
            int length = 0;
            Node* temp = new Node(current.x, current.y, current.g, current.h, current.parent);
            while (temp->parent != nullptr) {
                length++;
                temp = temp->parent;
            }
            result.pathLength = length;
            
            // Clean up dynamically allocated memory (basic cleanup)
            // A full implementation would track and delete all created nodes
            delete startNode; 
            
            return result;
        }

        if (closedSet[current.x][current.y]) {
            continue;
        }
        closedSet[current.x][current.y] = true;

        // Explore neighbors
        for (int i = 0; i < 4; ++i) {
            int nextX = current.x + dx[i];
            int nextY = current.y + dy[i];

            // Check boundaries
            if (nextX < 0 || nextX >= GRID_SIZE || nextY < 0 || nextY >= GRID_SIZE) {
                continue;
            }

            // Check obstacle or closed set
            if (grid[nextX][nextY] == 1 || closedSet[nextX][nextY]) {
                continue;
            }

            double newG = current.g + 1; // Move cost is 1
            double newH = heuristic(nextX, nextY, goalX, goalY);
            Node* successor = new Node(nextX, nextY, newG, newH, new Node(current.x, current.y, current.g, current.h, current.parent));
            
            openSet.push(*successor);
        }
    }

    // No path found
    auto endTime = chrono::high_resolution_clock::now();
    result.executionTimeMs = chrono::duration<double, milli>(endTime - startTime).count();
    return result; // pathLength will be 0
}

// --- Main Experiment Loop ---
int main() {
    srand(time(0)); // Seed random number generator

    double totalNodes_man = 0, totalPath_man = 0, totalTime_man = 0;
    double totalNodes_euc = 0, totalPath_euc = 0, totalTime_euc = 0;
    double totalNodes_dij = 0, totalPath_dij = 0, totalTime_dij = 0;
    int successCount = 0;

    cout << "Running " << NUM_TRIALS << " trials..." << endl;

    for (int i = 0; i < NUM_TRIALS; ++i) {
        // 1. Create Grid
        vector<vector<int>> grid(GRID_SIZE, vector<int>(GRID_SIZE, 0));
        
        // 2. Add Obstacles
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                if ((double)rand() / RAND_MAX < OBSTACLE_DENSITY) {
                    grid[r][c] = 1; // 1 = obstacle
                }
            }
        }

        // 3. Get valid Start and Goal
        int startX, startY, goalX, goalY;
        do {
            startX = rand() % GRID_SIZE;
            startY = rand() % GRID_SIZE;
        } while (grid[startX][startY] == 1);
        
        do {
            goalX = rand() % GRID_SIZE;
            goalY = rand() % GRID_SIZE;
        } while (grid[goalX][goalY] == 1 || (startX == goalX && startY == goalY));

        // --- Run all 3 heuristics on the SAME grid ---
        AStarResult res_man = a_star_search(grid, startX, startY, goalX, goalY, heuristic_manhattan);
        AStarResult res_euc = a_star_search(grid, startX, startY, goalX, goalY, heuristic_euclidean);
        AStarResult res_dij = a_star_search(grid, startX, startY, goalX, goalY, heuristic_zero);

        if (res_man.pathLength > 0) { // Check if a path was found
            successCount++;
            totalNodes_man += res_man.nodesExpanded;
            totalPath_man += res_man.pathLength;
            totalTime_man += res_man.executionTimeMs;

            totalNodes_euc += res_euc.nodesExpanded;
            totalPath_euc += res_euc.pathLength;
            totalTime_euc += res_euc.executionTimeMs;

            totalNodes_dij += res_dij.nodesExpanded;
            totalPath_dij += res_dij.pathLength;
            totalTime_dij += res_dij.executionTimeMs;
        } else {
            i--; // Retry this trial with a new grid if no path was possible
        }
    }

    // --- Print Final Averaged Results ---
    cout << "\n--- Final Results (Averaged over " << successCount << " successful trials) ---" << endl;
    cout << fixed << setprecision(2);

    cout << "\nMetric: Manhattan Distance" << endl;
    cout << "  Average Nodes Expanded: " << (totalNodes_man / successCount) << endl;
    cout << "  Average Path Length:    " << (totalPath_man / successCount) << endl;
    cout << "  Average Exec. Time (ms):" << (totalTime_man / successCount) << endl;

    cout << "\nMetric: Euclidean Distance" << endl;
    cout << "  Average Nodes Expanded: " << (totalNodes_euc / successCount) << endl;
    cout << "  Average Path Length:    " << (totalPath_euc / successCount) << endl;
    cout << "  Average Exec. Time (ms):" << (totalTime_euc / successCount) << endl;

    cout << "\nMetric: Zero Heuristic (Dijkstra's)" << endl;
    cout << "  Average Nodes Expanded: " << (totalNodes_dij / successCount) << endl;
    cout << "  Average Path Length:    " << (totalPath_dij / successCount) << endl;
    cout << "  Average Exec. Time (ms):" << (totalTime_dij / successCount) << endl;

    return 0;
}