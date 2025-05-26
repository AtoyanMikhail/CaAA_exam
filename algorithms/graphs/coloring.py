from typing import List, Dict, Set, Optional
from itertools import combinations
import random
from collections import defaultdict

class Graph:
    """Представление графа через список смежности"""
    def __init__(self, vertices: int):
        self.vertices = vertices
        self.adj = defaultdict(list)
    
    def add_edge(self, u: int, v: int):
        """Добавление ребра в граф"""
        self.adj[u].append(v)
        self.adj[v].append(u)

def is_valid_coloring(graph: Graph, colors: List[int]) -> bool:
    """
    Проверяет, является ли раскраска допустимой.
    
    Args:
        graph: Граф
        colors: Список цветов для каждой вершины
        
    Returns:
        True, если раскраска допустима
    """
    for u in range(graph.vertices):
        for v in graph.adj[u]:
            if colors[u] == colors[v]:
                return False
    return True

def brute_force_3coloring(graph: Graph) -> Optional[List[int]]:
    """
    Алгоритм полного перебора для раскраски в 3 цвета.
    
    Args:
        graph: Граф
        
    Returns:
        Список цветов для каждой вершины или None, если раскраска невозможна
    """
    def try_coloring(vertex: int, colors: List[int]) -> bool:
        if vertex == graph.vertices:
            return True
            
        for color in range(3):
            colors[vertex] = color
            if is_valid_coloring(graph, colors[:vertex+1]):
                if try_coloring(vertex + 1, colors):
                    return True
            colors[vertex] = -1
            
        return False
    
    colors = [-1] * graph.vertices
    if try_coloring(0, colors):
        return colors
    return None

def two_color_restricted_3coloring(graph: Graph) -> Optional[List[int]]:
    """
    Перебор с учетом выбора только из 2 цветов для каждой вершины.
    
    Args:
        graph: Граф
        
    Returns:
        Список цветов для каждой вершины или None, если раскраска невозможна
    """
    def get_available_colors(vertex: int, colors: List[int]) -> Set[int]:
        used_colors = {colors[v] for v in graph.adj[vertex] if colors[v] != -1}
        return {0, 1, 2} - used_colors
    
    def try_coloring(vertex: int, colors: List[int]) -> bool:
        if vertex == graph.vertices:
            return True
            
        available_colors = get_available_colors(vertex, colors)
        for color in available_colors:
            colors[vertex] = color
            if try_coloring(vertex + 1, colors):
                return True
            colors[vertex] = -1
            
        return False
    
    colors = [-1] * graph.vertices
    if try_coloring(0, colors):
        return colors
    return None

def subset_3coloring(graph: Graph) -> Optional[List[int]]:
    """
    Перебор подмножеств размера <= n/3.
    
    Args:
        graph: Граф
        
    Returns:
        Список цветов для каждой вершины или None, если раскраска невозможна
    """
    n = graph.vertices
    max_subset_size = n // 3
    
    # Перебираем все возможные подмножества вершин для каждого цвета
    for size in range(max_subset_size + 1):
        for subset in combinations(range(n), size):
            colors = [-1] * n
            # Раскрашиваем подмножество в цвет 0
            for v in subset:
                colors[v] = 0
                
            # Пробуем раскрасить оставшиеся вершины
            if try_coloring_remaining(graph, colors, 1):
                return colors
                
    return None

def try_coloring_remaining(graph: Graph, colors: List[int], current_color: int) -> bool:
    """Вспомогательная функция для subset_3coloring"""
    if current_color > 2:
        return is_valid_coloring(graph, colors)
        
    # Находим первую нераскрашенную вершину
    for v in range(graph.vertices):
        if colors[v] == -1:
            # Пробуем раскрасить в текущий цвет
            colors[v] = current_color
            if is_valid_coloring(graph, colors):
                if try_coloring_remaining(graph, colors, current_color + 1):
                    return True
            colors[v] = -1
            
    return False

def probabilistic_3coloring(graph: Graph, max_attempts: int = 1000) -> Optional[List[int]]:
    """
    Вероятностный алгоритм раскраски в 3 цвета.
    
    Args:
        graph: Граф
        max_attempts: Максимальное количество попыток
        
    Returns:
        Список цветов для каждой вершины или None, если раскраска не найдена
    """
    for _ in range(max_attempts):
        colors = [random.randint(0, 2) for _ in range(graph.vertices)]
        if is_valid_coloring(graph, colors):
            return colors
    return None

def reduce_to_sat(graph: Graph) -> List[List[int]]:
    """
    Сведение задачи раскраски в 3 цвета к задаче выполнимости.
    
    Args:
        graph: Граф
        
    Returns:
        Список дизъюнктов в формате CNF
    """
    clauses = []
    n = graph.vertices
    
    # Каждая вершина должна быть окрашена хотя бы в один цвет
    for v in range(n):
        clauses.append([3*v + 1, 3*v + 2, 3*v + 3])
    
    # Каждая вершина не может быть окрашена в два цвета одновременно
    for v in range(n):
        for c1 in range(1, 4):
            for c2 in range(c1 + 1, 4):
                clauses.append([-(3*v + c1), -(3*v + c2)])
    
    # Смежные вершины не могут быть окрашены в один цвет
    for u in range(n):
        for v in graph.adj[u]:
            if u < v:  # Чтобы не дублировать условия
                for c in range(1, 4):
                    clauses.append([-(3*u + c), -(3*v + c)])
    
    return clauses

if __name__ == "__main__":
    # Пример использования
    g = Graph(5)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 0)
    
    print("Полный перебор:")
    colors = brute_force_3coloring(g)
    print(colors if colors else "Раскраска невозможна")
    
    print("\nПеребор с двумя цветами:")
    colors = two_color_restricted_3coloring(g)
    print(colors if colors else "Раскраска невозможна")
    
    print("\nПеребор подмножеств:")
    colors = subset_3coloring(g)
    print(colors if colors else "Раскраска невозможна")
    
    print("\nВероятностный алгоритм:")
    colors = probabilistic_3coloring(g)
    print(colors if colors else "Раскраска не найдена")
    
    print("\nСведение к SAT:")
    clauses = reduce_to_sat(g)
    print(f"Количество дизъюнктов: {len(clauses)}")
    print("Первые 5 дизъюнктов:")
    for clause in clauses[:5]:
        print(clause) 