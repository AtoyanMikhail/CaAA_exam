from typing import List, Dict, Set, Tuple
import random
from collections import defaultdict

class Graph:
    """Представление графа через список смежности"""
    def __init__(self, vertices: int):
        self.vertices = vertices
        self.edges = []  # список ребер
        self.adj = defaultdict(list)  # список смежности
    
    def add_edge(self, u: int, v: int, weight: int = 1):
        """Добавление ребра в граф"""
        self.edges.append((u, v, weight))
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))

class DisjointSet:
    """Структура данных для непересекающихся множеств"""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Находит представителя множества"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        """Объединяет два множества"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

def karger_min_cut(graph: Graph) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Алгоритм Каргера для нахождения минимального разреза.
    
    Args:
        graph: Граф
        
    Returns:
        (вес разреза, список ребер в разрезе)
    """
    n = graph.vertices
    ds = DisjointSet(n)
    edges = graph.edges.copy()
    random.shuffle(edges)
    
    # Объединяем вершины, пока не останется две
    vertices = n
    for u, v, w in edges:
        if vertices <= 2:
            break
            
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            vertices -= 1
    
    # Находим ребра разреза
    cut_edges = []
    cut_weight = 0
    
    for u, v, w in edges:
        if ds.find(u) != ds.find(v):
            cut_edges.append((u, v))
            cut_weight += w
    
    return cut_weight, cut_edges

def karger_stein_min_cut(graph: Graph) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Оптимизированный алгоритм Каргера-Штейна для нахождения минимального разреза.
    
    Args:
        graph: Граф
        
    Returns:
        (вес разреза, список ребер в разрезе)
    """
    def contract(graph: Graph, t: int) -> Graph:
        """Схлопывает граф до t вершин"""
        n = graph.vertices
        ds = DisjointSet(n)
        edges = graph.edges.copy()
        random.shuffle(edges)
        
        vertices = n
        for u, v, w in edges:
            if vertices <= t:
                break
                
            if ds.find(u) != ds.find(v):
                ds.union(u, v)
                vertices -= 1
        
        # Создаем новый граф
        new_graph = Graph(t)
        new_edges = defaultdict(int)
        
        for u, v, w in edges:
            pu, pv = ds.find(u), ds.find(v)
            if pu != pv:
                new_edges[(pu, pv)] += w
        
        for (u, v), w in new_edges.items():
            new_graph.add_edge(u, v, w)
        
        return new_graph
    
    def recursive_min_cut(graph: Graph) -> Tuple[int, List[Tuple[int, int]]]:
        """Рекурсивная часть алгоритма"""
        n = graph.vertices
        if n <= 6:
            return karger_min_cut(graph)
        
        t = int(n / (2 ** 0.5) + 1)
        g1 = contract(graph, t)
        g2 = contract(graph, t)
        
        cut1 = recursive_min_cut(g1)
        cut2 = recursive_min_cut(g2)
        
        return min(cut1, cut2, key=lambda x: x[0])
    
    return recursive_min_cut(graph)

def find_min_cut(graph: Graph, num_trials: int = 100) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Находит минимальный разрез, запуская алгоритм Каргера-Штейна несколько раз.
    
    Args:
        graph: Граф
        num_trials: Количество попыток
        
    Returns:
        (вес минимального разреза, список ребер в разрезе)
    """
    best_cut = float('inf'), []
    
    for _ in range(num_trials):
        cut = karger_stein_min_cut(graph)
        if cut[0] < best_cut[0]:
            best_cut = cut
    
    return best_cut

if __name__ == "__main__":
    # Пример использования
    g = Graph(6)
    # Добавляем ребра графа
    edges = [
        (0, 1, 2), (0, 2, 3), (1, 2, 1),
        (1, 3, 4), (2, 3, 2), (2, 4, 3),
        (3, 4, 1), (3, 5, 2), (4, 5, 3)
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    
    print("Алгоритм Каргера:")
    cut_weight, cut_edges = karger_min_cut(g)
    print(f"Вес разреза: {cut_weight}")
    print("Ребра в разрезе:", cut_edges)
    
    print("\nАлгоритм Каргера-Штейна:")
    cut_weight, cut_edges = karger_stein_min_cut(g)
    print(f"Вес разреза: {cut_weight}")
    print("Ребра в разрезе:", cut_edges)
    
    print("\nМногократный запуск:")
    cut_weight, cut_edges = find_min_cut(g)
    print(f"Вес минимального разреза: {cut_weight}")
    print("Ребра в минимальном разрезе:", cut_edges) 