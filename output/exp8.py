import copy
import heapq
from itertools import permutations

INF = float('inf')

def reduce_matrix(mat):
    """Reduces row and column minima and returns the reduced matrix and cost."""
    m = [row[:] for row in mat]
    n = len(m)
    cost = 0

    # 1. Row reduction
    for i in range(n):
        row_min = min(m[i])
        if row_min > 0 and row_min != INF:
            cost += row_min
            m[i] = [x - row_min if x != INF else INF for x in m[i]]

    # 2. Column reduction
    for j in range(n):
        col_min = min(m[i][j] for i in range(n))
        if col_min > 0 and col_min != INF:
            cost += col_min
            for i in range(n):
                if m[i][j] != INF:
                    m[i][j] -= col_min

    return m, cost


def tsp_branch_and_bound(cost_matrix):
    """Solves TSP using Branch and Bound with Matrix Reduction."""
    n = len(cost_matrix)
    initial_mat, initial_bound = reduce_matrix(cost_matrix)

    # Priority queue stores tuples: (bound, current_node, path, reduced_matrix)
    pq = [(initial_bound, 0, [0], initial_mat)]

    while pq:
        bound, u, path, mat = heapq.heappop(pq)

        # Return when a full tour is found
        if len(path) == n:
            path.append(0)
            return path, bound + cost_matrix[u][0]

        # Explore outgoing unvisited cities
        for v in range(n):
            if v not in path and mat[u][v] != INF:
                # Create a copy of matrix for next state
                new_mat = [row[:] for row in mat]

                # Set row u and col v to INF
                for k in range(n):
                    new_mat[u][k] = INF
                    new_mat[k][v] = INF
                # Prevent returning directly to start node before tour completion
                new_mat[v][0] = INF

                # Reduce updated matrix
                red_mat, red_cost = reduce_matrix(new_mat)
                new_bound = bound + mat[u][v] + red_cost

                heapq.heappush(pq, (new_bound, v, path + [v], red_mat))

    return None, INF


def tsp_brute_force(cost_matrix, n):
    """Brute force implementation for verification."""
    cities = list(range(1, n))
    best_cost = INF
    best_path = None

    for perm in permutations(cities):
        path = [0] + list(perm) + [0]
        c = sum(cost_matrix[path[i]][path[i + 1]] for i in range(n))
        if c < best_cost:
            best_cost = c
            best_path = path

    return best_path, best_cost


# --- 5-city Cost Matrix ---
cost = [
    [INF, 10, 8, 9, 7],
    [10, INF, 10, 5, 6],
    [8, 10, INF, 8, 9],
    [9, 5, 8, INF, 6],
    [7, 6, 9, 6, INF],
]

n = 5
cities = ["A", "B", "C", "D", "E"]

# Run algorithms
bb_path, bb_cost = tsp_branch_and_bound(cost)
bf_path, bf_cost = tsp_brute_force(cost, n)

# Print Matrix
print("5-City TSP - Cost Matrix:")
print(f'{"":>4}', " ".join(f"{c:>5}" for c in cities))
for i, row in enumerate(cost):
    r = ["INF" if x == INF else str(x) for x in row]
    print(f"{cities[i]:>4}", " ".join(f"{v:>5}" for v in r))

print("\n--- Results ---")
print(
    f'Branch & Bound: {" -> ".join(cities[i] for i in bb_path)} | Cost: {bb_cost}'
)
print(
    f'Brute Force:    {" -> ".join(cities[i] for i in bf_path)} | Cost: {bf_cost}'
)