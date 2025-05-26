from typing import Dict, List, Set, Tuple, Optional
from collections import deque
import heapq
import math

class Graph:
    """Представление графа через список смежности"""
    def __init__(self):
        self.adj: Dict[int, List[Tuple[int, float]]] = {}
    
    def add_edge(self, u: int, v: int, weight: float = 1.0):
        """Добавление ребра в граф"""
        if u not in self.adj:
            self.adj[u] = []
        if v not in self.adj:
            self.adj[v] = []
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))  # для неориентированного графа

def bfs_shortest_path(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """
    Поиск кратчайшего пути с помощью BFS.
    
    Args:
        graph: Граф
        start: Начальная вершина
        end: Конечная вершина
        
    Returns:
        Список вершин, образующих кратчайший путь, или None, если путь не найден
    """
    if start not in graph.adj or end not in graph.adj:
        return None
        
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        if current == end:
            # Восстанавливаем путь
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
            
        for neighbor, _ in graph.adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    return None

def dijkstra(graph: Graph, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """
    Алгоритм Дейкстры для нахождения кратчайших путей.
    
    Args:
        graph: Граф
        start: Начальная вершина
        
    Returns:
        (словарь расстояний, словарь родителей)
    """
    if start not in graph.adj:
        return {}, {}
        
    dist = {v: float('inf') for v in graph.adj}
    dist[start] = 0
    parent = {start: None}
    
    pq = [(0, start)]
    heapq.heapify(pq)
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
            
        for v, w in graph.adj[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))
    
    return dist, parent

def reconstruct_path(parent: Dict[int, Optional[int]], start: int, end: int) -> Optional[List[int]]:
    """Восстановление пути по словарю родителей"""
    if end not in parent:
        return None
        
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]

def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Манхэттенское расстояние между точками"""
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Евклидово расстояние между точками"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def diagonal_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Диагональное расстояние между точками"""
    return max(abs(x1 - x2), abs(y1 - y2))

def astar(
    graph: Graph,
    start: int,
    end: int,
    h: callable,
    get_coords: callable
) -> Optional[List[int]]:
    """
    Алгоритм A* для поиска кратчайшего пути.
    
    Args:
        graph: Граф
        start: Начальная вершина
        end: Конечная вершина
        h: Эвристическая функция
        get_coords: Функция получения координат вершины
        
    Returns:
        Список вершин, образующих кратчайший путь, или None, если путь не найден
    """
    if start not in graph.adj or end not in graph.adj:
        return None
        
    open_set = {start}
    closed_set = set()
    
    g_score = {start: 0}
    f_score = {start: h(*get_coords(start), *get_coords(end))}
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
            f_score[neighbor] = g_score[neighbor] + h(*get_coords(neighbor), *get_coords(end))
    
    return None

if __name__ == "__main__":
    # Пример использования
    g = Graph()
    
    # Создаем сетку 3x3
    for i in range(3):
        for j in range(3):
            v = i * 3 + j
            if i < 2:  # горизонтальные ребра
                g.add_edge(v, v + 3, 1.0)
            if j < 2:  # вертикальные ребра
                g.add_edge(v, v + 1, 1.0)
    
    # Функция получения координат для A*
    def get_coords(v: int) -> Tuple[int, int]:
        return (v // 3, v % 3)
    
    # Тестируем алгоритмы
    start, end = 0, 8
    
    print("BFS путь:", bfs_shortest_path(g, start, end))
    
    dist, parent = dijkstra(g, start)
    print("Дейкстра путь:", reconstruct_path(parent, start, end))
    print("Расстояния:", dist)
    
    print("A* путь (Манхэттен):", 
          astar(g, start, end, manhattan_distance, get_coords))
    print("A* путь (Евклидово):", 
          astar(g, start, end, euclidean_distance, get_coords))
    print("A* путь (Диагональное):", 
          astar(g, start, end, diagonal_distance, get_coords)) 