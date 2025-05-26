from typing import List, Tuple
import networkx as nx
import heapq

def find(parent: List[int], i: int) -> int:
    """Находит корень множества для элемента i"""
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent: List[int], rank: List[int], x: int, y: int):
    """Объединяет два множества"""
    root_x = find(parent, x)
    root_y = find(parent, y)
    
    if root_x == root_y:
        return
        
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1

def kruskal_mst(graph: nx.Graph) -> List[Tuple[int, int, float]]:
    """
    Реализация алгоритма Краскала для построения МОД.
    
    Args:
        graph: Граф в виде объекта NetworkX
        
    Returns:
        Список рёбер МОД в формате (u, v, weight)
    """
    # Получаем список всех рёбер и сортируем их по весу
    edges = [(u, v, data['weight']) for u, v, data in graph.edges(data=True)]
    edges.sort(key=lambda x: x[2])
    
    # Инициализируем структуры для системы непересекающихся множеств
    n = graph.number_of_nodes()
    parent = list(range(n))
    rank = [0] * n
    
    # Список для хранения рёбер МОД
    mst_edges = []
    
    # Проходим по всем рёбрам в порядке возрастания веса
    for u, v, weight in edges:
        if find(parent, u) != find(parent, v):
            mst_edges.append((u, v, weight))
            union(parent, rank, u, v)
            
    return mst_edges

def prim_mst(graph: nx.Graph) -> List[Tuple[int, int, float]]:
    """
    Реализация алгоритма Прима для построения МОД.
    
    Args:
        graph: Граф в виде объекта NetworkX
        
    Returns:
        Список рёбер МОД в формате (u, v, weight)
    """
    if not graph.nodes():
        return []
        
    # Инициализация
    n = graph.number_of_nodes()
    visited = set()
    mst_edges = []
    
    # Начинаем с произвольной вершины
    start_vertex = list(graph.nodes())[0]
    visited.add(start_vertex)
    
    # Приоритетная очередь для рёбер
    edges_queue = []
    for neighbor in graph.neighbors(start_vertex):
        weight = graph[start_vertex][neighbor]['weight']
        heapq.heappush(edges_queue, (weight, start_vertex, neighbor))
    
    # Пока не посетим все вершины
    while len(visited) < n and edges_queue:
        weight, u, v = heapq.heappop(edges_queue)
        
        if v not in visited:
            mst_edges.append((u, v, weight))
            visited.add(v)
            
            for neighbor in graph.neighbors(v):
                if neighbor not in visited:
                    weight = graph[v][neighbor]['weight']
                    heapq.heappush(edges_queue, (weight, v, neighbor))
    
    return mst_edges

def create_example_graph() -> nx.Graph:
    """
    Создаёт пример графа для демонстрации работы алгоритма.
    """
    G = nx.Graph()
    edges = [
        (0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2),
        (2, 3, 4), (2, 4, 5), (3, 4, 7), (3, 5, 2),
        (4, 5, 6)
    ]
    G.add_weighted_edges_from(edges)
    return G

if __name__ == "__main__":
    # Пример использования обоих алгоритмов
    G = create_example_graph()
    
    print("Алгоритм Краскала:")
    mst_kruskal = kruskal_mst(G)
    for u, v, weight in mst_kruskal:
        print(f"({u}, {v}) с весом {weight}")
    
    print("\nАлгоритм Прима:")
    mst_prim = prim_mst(G)
    for u, v, weight in mst_prim:
        print(f"({u}, {v}) с весом {weight}") 