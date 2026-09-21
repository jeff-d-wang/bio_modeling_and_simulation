import collections
import math
import sys

# Test Case 1: One best chain
# python3 hw1.py <<'EOF'
# 3 0 2
# 0.0 1.0 3.0
# 100.0 0.0 1.5
# 100.0 100.0 0.0
# EOF
# Expected Output:
# Edges in the union of most probable paths from TF 0 to TF2:
# (0 --> 1),       (w(0,1) = 1.0)
# (1 --> 2),       (w(1,2) = 1.5)

# Test Case 2: Two best chains
# python3 hw1.py <<'EOF'
# 4 0 3
# 0.0 0.5 0.9 5.0
# 100.0 0.0 100.0 0.8
# 100.0 100.0 0.0 0.4
# 100.0 100.0 100.0 0.0
# EOF
# Expected Output:
# Edges in the union of most probable paths from TF 0 to TF3:
# (0 --> 1),       (w(0,1) = 0.5)
# (0 --> 2),       (w(0,2) = 0.9)
# (1 --> 3),       (w(1,3) = 0.8)
# (2 --> 3),       (w(2,3) = 0.4)

def most_probable_regulatory_chain():
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # first line: TF count, start node, end node
    n = int(input_data[0])
    s = int(input_data[1])
    t = int(input_data[2])

    # n x n matrix of -log(p(i, j))
    matrix_weights = []
    idx = 3
    for i in range(n):
        row = []
        for j in range(n):
            row.append(float(input_data[idx]))
            idx += 1
        matrix_weights.append(row)

    # dijkstra
    dist = [math.inf] * n
    parents = [set() for _ in range(n)]

    dist[s] = 0.0
    visited = [False] * n

    for _ in range(n):
        min_d = math.inf
        u = -1
        for i in range(n):
            if not visited[i] and dist[i] < min_d:
                min_d = dist[i]
                u = i

        if u == -1 or min_d == math.inf:
            break  # nothing left reachable

        visited[u] = True

        for v in range(n):
            if u == v or visited[v]:
                continue

            w_uv = matrix_weights[u][v]
            new_dist = dist[u] + w_uv

            if new_dist < dist[v] - 1e-9:
                dist[v] = new_dist
                parents[v] = {u}
            elif math.isclose(new_dist, dist[v], abs_tol=1e-9):
                parents[v].add(u)  # tied path, keep both parents

    # walk back from t to get all optimal path
    union_edges = set()

    if dist[t] == math.inf:
        print("No path found.")
        return

    queue = collections.deque([t])
    visited_nodes = {t}

    while queue:
        v = queue.popleft()
        for u in parents[v]:
            union_edges.add((u, v))
            if u not in visited_nodes:
                visited_nodes.add(u)
                queue.append(u)

    print(f"Edges in the union of most probable paths from TF {s} to TF{t}:")
    for u, v in sorted(union_edges):
        print(f"({u} --> {v}),\t (w({u},{v}) = {matrix_weights[u][v]})")


if __name__ == "__main__":
    most_probable_regulatory_chain()