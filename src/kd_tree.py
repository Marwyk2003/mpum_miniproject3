import numpy as np
from heapq import heappush, heappushpop, nsmallest, heapify


class Point:
    def __init__(self, p: np.ndarray, c: int):
        self.p = p
        self.c = c

    def __str__(self):
        return f'{self.p} -> {self.c}'

class ProximityPoint:
    def __init__(self, dist: float, p: Point):
        self.dist = dist
        self.p = p

    def __lt__(self, other):
        return other.dist < self.dist

    def __str__(self):
        return f'd({self.p})  = {self.dist}'


class KDNode:
    def __init__(self, boundaries: list[tuple[float, float]], capacity=15):
        self.boundaries = boundaries
        self.children = [None, None]
        self.split_idx = -1
        self.split_val = -1
        self.points = []
        self.leaf = True
        self.capacity = capacity
        # print(split_idx)

    def insert(self, p: Point) -> None:
        if len(self.points) <= self.capacity and self.leaf:
            # quad is an empty leaf, change its value
            self.points += [p]
        else:
            if self.leaf:
                s = np.zeros(p.p.shape)
                for q in self.points:
                    s += q.p
                # print(self.points)
                s /= len(self.points)
                s -= np.full(p.p.shape, 0.5)
                s = np.abs(s)

                split = np.argmin(s)
                if s[split] >= 0.5:
                    self.points += [p]
                    return
                self.split_idx = split
                self.split_val = 0.5


            q_key, q_bound = self.quarter(p)
            if self.children[q_key] is None:
                # split quad
                self.children[q_key] = KDNode(q_bound, capacity=self.capacity)
            # make it its child problem
            self.children[q_key].insert(p)

            if self.leaf:
                for q in self.points:
                    # insert quads value if it was a leaf before
                    q2_key, q2_bound = self.quarter(q)
                    if self.children[q2_key] is None:
                        self.children[q2_key] = KDNode(q2_bound, capacity=self.capacity)
                    self.children[q2_key].insert(q)
                self.points = []
            self.leaf = False

    def find_closest(self, p: np.ndarray, k: int) -> list[ProximityPoint]:
        closest = [ProximityPoint(float('inf'), None)]
        heapify(closest)
        self.__find_closest(Point(p, -1), k, closest)
        return closest

    def __find_closest(self, p, k, closest) -> None:
        if self.leaf:
            for q in self.points:
                dist = self.dist(p.p, q.p)
                if len(closest) >= k:
                    heappushpop(closest, ProximityPoint(dist, q))
                else:
                    heappush(closest, ProximityPoint(dist, q))

        # assert closest[0] == nsmallest(1, closest)[0].dist
        if self.quad_dist(p) < closest[0].dist:
            # children = filter(lambda x: x is not None, self.children)
            # for c in sorted(children, key=lambda x: x.quad_dist(p)):
            for c in self.children:
                if c is not None:
                # this should speed up converging
                    c.__find_closest(p, k, closest)

    def quarter(self, p: Point) -> tuple[int, list[tuple[float, float]]]:
        # print(p.p[1],self.split_idx)
        idx = 0 if p.p[self.split_idx] <= self.split_val else 1
        boundaries = [x for x in self.boundaries]
        boundaries[self.split_idx] = tuple(sorted([self.boundaries[self.split_idx][idx], 0.5]))
        return idx, boundaries

    def dist(self, p: np.ndarray, q: np.ndarray) -> float:
        # print(p, q, np.sum((p - q) ** 2))
        return np.sum((p - q) ** 2)

    def quad_dist(self, p: Point) -> float:
        inside = True
        q = np.zeros(p.p.shape)
        for i in range(p.p.shape[0]):
            x1, x2 = self.boundaries[i]
            x = p.p[i]
            q[i] = x1 if x <= x1 else x2 if x >= x2 else x
            if q[i] != x:
                inside = False
        return 0 if inside else self.dist(p.p, q)

    def debug(self, indent=0, split_idx=None, val=None):
        print('  '*indent, end='')
        if self.leaf:
            print(f'points: {len(self.points)}')
            # print('  '*indent, end='')
            # print(*self.points, sep='\n'+'  '*indent)
            if split_idx is not None:
                for p in self.points:
                    assert p.p[split_idx] == val
        else:
            print(f'{self.split_idx}, {self.split_val}')
            for i, c in enumerate(self.children):
                if c is not None:
                    print('  '*indent, end='')
                    print(c.boundaries)
                    c.debug(indent+1, self.split_idx, i)
