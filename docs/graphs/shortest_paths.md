# Алгоритмы поиска кратчайших путей

## Введение

Поиск кратчайших путей - одна из фундаментальных задач теории графов. В зависимости от свойств графа (наличие отрицательных весов, требования к оптимальности) используются различные алгоритмы.

## BFS для невзвешенных графов

### Теорема
BFS находит кратчайший путь в невзвешенном графе за время O(|V| + |E|).

### Доказательство
1. BFS посещает вершины в порядке возрастания расстояния от начальной
2. Каждая вершина и ребро посещаются не более одного раза
3. Следовательно, сложность O(|V| + |E|)

### Реализация

```python
def bfs_shortest_path(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """Поиск кратчайшего пути с помощью BFS"""
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
```

## Алгоритм Дейкстры

### Теорема
Алгоритм Дейкстры находит кратчайшие пути от начальной вершины до всех остальных за время O(|E| log |V|).

### Доказательство
1. Каждая вершина извлекается из очереди с приоритетом не более одного раза
2. Для каждой вершины выполняется не более |E| операций с очередью
3. Операции с очередью с приоритетом выполняются за O(log |V|)
4. Следовательно, общая сложность O(|E| log |V|)

### Реализация

```python
def dijkstra(graph: Graph, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Алгоритм Дейкстры для нахождения кратчайших путей"""
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
```

## Алгоритм A*

### Теорема
Алгоритм A* находит оптимальный путь, если эвристическая функция допустима.

### Доказательство
1. Эвристическая функция h(n) не переоценивает стоимость пути до цели
2. Функция оценки f(n) = g(n) + h(n) не переоценивает общую стоимость
3. A* всегда выбирает вершину с минимальной оценкой f(n)
4. Следовательно, первый найденный путь до цели оптимален

### Реализация

```python
def astar(
    graph: Graph,
    start: int,
    end: int,
    h: callable,
    get_coords: callable
) -> Optional[List[int]]:
    """Алгоритм A* для поиска кратчайшего пути"""
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
```

## Эвристические функции

### Манхэттенское расстояние

```python
def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Манхэттенское расстояние между точками"""
    return abs(x1 - x2) + abs(y1 - y2)
```

### Евклидово расстояние

```python
def euclidean_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Евклидово расстояние между точками"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
```

### Диагональное расстояние

```python
def diagonal_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Диагональное расстояние между точками"""
    return max(abs(x1 - x2), abs(y1 - y2))
```

## Сложность алгоритмов

### BFS
- Время: O(|V| + |E|)
- Память: O(|V|)

### Дейкстра
- Время: O(|E| log |V|)
- Память: O(|V|)

### A*
- Время: O(|E| log |V|)
- Память: O(|V|)

## Применения

### Навигация
- Поиск маршрутов в картах
- Планирование движения роботов
- Управление беспилотниками

### Сети
- Маршрутизация пакетов
- Оптимизация трафика
- Балансировка нагрузки

### Игры
- Поиск пути в стратегиях
- Искусственный интеллект
- Процедурная генерация 