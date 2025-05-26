from typing import Dict, List, Set, Tuple, Optional
import heapq
import random
import math
from collections import defaultdict

class Graph:
    """Расширенное представление графа"""
    def __init__(self):
        self.adj: Dict[int, List[Tuple[int, float]]] = {}
        self.vertices: Set[int] = set()
        self.landmarks: List[int] = []
        self.landmark_distances: Dict[Tuple[int, int], float] = {}
        self.reach_values: Dict[int, float] = {}
    
    def add_edge(self, u: int, v: int, weight: float = 1.0):
        """Добавление ребра в граф"""
        if u not in self.adj:
            self.adj[u] = []
            self.vertices.add(u)
        if v not in self.adj:
            self.adj[v] = []
            self.vertices.add(v)
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))
    
    def get_weight(self, u: int, v: int) -> float:
        """Получение веса ребра"""
        for neighbor, weight in self.adj[u]:
            if neighbor == v:
                return weight
        return float('inf')
    
    def update_weight(self, u: int, v: int, new_weight: float):
        """Обновление веса ребра"""
        for i, (neighbor, _) in enumerate(self.adj[u]):
            if neighbor == v:
                self.adj[u][i] = (v, new_weight)
                break
        for i, (neighbor, _) in enumerate(self.adj[v]):
            if neighbor == u:
                self.adj[v][i] = (u, new_weight)
                break

def dijkstra_distances(graph: Graph, start: int) -> Dict[int, float]:
    """Вычисление расстояний от начальной вершины до всех остальных"""
    dist = {v: float('inf') for v in graph.vertices}
    dist[start] = 0
    pq = [(0, start)]
    heapq.heapify(pq)
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
            
        for v, w in graph.adj[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return dist

def select_landmarks(graph: Graph, num_landmarks: int) -> List[int]:
    """Выбор ориентиров для алгоритма ALT"""
    # Выбор по удалённости
    landmarks = []
    remaining = set(graph.vertices)
    
    # Начинаем с случайной вершины
    current = random.choice(list(remaining))
    landmarks.append(current)
    remaining.remove(current)
    
    while len(landmarks) < num_landmarks and remaining:
        max_dist = 0
        next_landmark = None
        
        for v in remaining:
            min_dist = float('inf')
            for l in landmarks:
                dist = dijkstra_distances(graph, l)[v]
                min_dist = min(min_dist, dist)
            if min_dist > max_dist:
                max_dist = min_dist
                next_landmark = v
        
        if next_landmark is not None:
            landmarks.append(next_landmark)
            remaining.remove(next_landmark)
    
    return landmarks

def compute_landmark_distances(graph: Graph, landmarks: List[int]):
    """Вычисление расстояний до ориентиров"""
    for l in landmarks:
        dist = dijkstra_distances(graph, l)
        for v in graph.vertices:
            graph.landmark_distances[(v, l)] = dist[v]

def alt_heuristic(graph: Graph, current: int, target: int) -> float:
    """Эвристическая функция для алгоритма ALT"""
    max_estimate = 0
    for l in graph.landmarks:
        # Используем неравенство треугольника
        estimate1 = abs(graph.landmark_distances[(current, l)] - 
                       graph.landmark_distances[(target, l)])
        estimate2 = abs(graph.landmark_distances[(l, target)] - 
                       graph.landmark_distances[(l, current)])
        max_estimate = max(max_estimate, estimate1, estimate2)
    return max_estimate

def compute_reach_values(graph: Graph):
    """Вычисление значений достижимости для алгоритма REACH"""
    for v in graph.vertices:
        # Вычисляем достижимость для каждой вершины
        min_reach = float('inf')
        for s in graph.vertices:
            for t in graph.vertices:
                if s != t and s != v and t != v:
                    d_sv = dijkstra_distances(graph, s)[v]
                    d_vt = dijkstra_distances(graph, v)[t]
                    reach = max(d_sv, d_vt)
                    min_reach = min(min_reach, reach)
        graph.reach_values[v] = min_reach

def reach_heuristic(graph: Graph, current: int, target: int, 
                   base_heuristic: callable) -> float:
    """Эвристическая функция для алгоритма REACH"""
    return max(base_heuristic(current, target), graph.reach_values[current])

def contract_edges(graph: Graph):
    """Оптимизация слияния рёбер"""
    for v in graph.vertices:
        for u, w1 in graph.adj[v]:
            for w, w2 in graph.adj[u]:
                if w != v:
                    # Проверяем неравенство треугольника
                    current_weight = graph.get_weight(v, w)
                    if current_weight > w1 + w2:
                        # Обновляем вес ребра
                        graph.update_weight(v, w, w1 + w2)

def astar_alt(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """Алгоритм A* с оптимизацией ALT"""
    if not graph.landmarks:
        graph.landmarks = select_landmarks(graph, min(16, len(graph.vertices)))
        compute_landmark_distances(graph, graph.landmarks)
    
    return astar(graph, start, end, 
                lambda v: alt_heuristic(graph, v, end),
                lambda v: (v // 3, v % 3))  # Пример функции координат

def astar_reach(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """Алгоритм A* с оптимизацией REACH"""
    if not graph.reach_values:
        compute_reach_values(graph)
    
    base_h = lambda v: manhattan_distance(v // 3, v % 3, end // 3, end % 3)
    return astar(graph, start, end,
                lambda v: reach_heuristic(graph, v, end, base_h),
                lambda v: (v // 3, v % 3))

def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Манхэттенское расстояние между точками"""
    return abs(x1 - x2) + abs(y1 - y2)

def astar(graph: Graph, start: int, end: int, h: callable, 
          get_coords: callable) -> Optional[List[int]]:
    """Базовая реализация A*"""
    if start not in graph.adj or end not in graph.adj:
        return None
        
    open_set = {start}
    closed_set = set()
    
    g_score = {start: 0}
    f_score = {start: h(start)}
    parent = {start: None}
    
    while open_set:
        current = min(open_set, key=lambda x: f_score[x])
        if current == end:
            return reconstruct_path(parent, start, end)
            
        open_set.remove(current)
        closed_set.add(current)
        
        for neighbor, weight in graph.adj[current]:
            if neighbor in closed_set:
                continue
                
            tentative_g = g_score[current] + weight
            
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g >= g_score.get(neighbor, float('inf')):
                continue
                
            parent[neighbor] = current
            g_score[neighbor] = tentative_g
            f_score[neighbor] = g_score[neighbor] + h(neighbor)
    
    return None

def reconstruct_path(parent: Dict[int, Optional[int]], 
                    start: int, end: int) -> Optional[List[int]]:
    """Восстановление пути по словарю родителей"""
    if end not in parent:
        return None
        
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]

if __name__ == "__main__":
    # Пример использования
    g = Graph()
    
    # Создаем сетку 5x5
    for i in range(5):
        for j in range(5):
            v = i * 5 + j
            if i < 4:  # горизонтальные ребра
                g.add_edge(v, v + 5, 1.0)
            if j < 4:  # вертикальные ребра
                g.add_edge(v, v + 1, 1.0)
    
    # Тестируем оптимизации
    start, end = 0, 24
    
    print("Базовый A*:")
    path1 = astar(g, start, end, 
                  lambda v: manhattan_distance(v // 5, v % 5, end // 5, end % 5),
                  lambda v: (v // 5, v % 5))
    print("Путь:", path1)
    
    print("\nA* с ALT:")
    path2 = astar_alt(g, start, end)
    print("Путь:", path2)
    
    print("\nA* с REACH:")
    path3 = astar_reach(g, start, end)
    print("Путь:", path3)
    
    print("\nОптимизация слияния рёбер:")
    contract_edges(g)
    path4 = astar(g, start, end,
                  lambda v: manhattan_distance(v // 5, v % 5, end // 5, end % 5),
                  lambda v: (v // 5, v % 5))
    print("Путь после слияния:", path4) 