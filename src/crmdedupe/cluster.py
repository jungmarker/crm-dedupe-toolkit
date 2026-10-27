"""Turn matched pairs into clusters (connected components) with union-find."""


def clusters(ids, pairs) -> dict:
    """Return {record_id: cluster_id} where cluster_id is the smallest id in the component."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}
