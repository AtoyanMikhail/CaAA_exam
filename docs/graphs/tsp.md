# Алгоритмы решения задачи коммивояжёра

## Введение

Задача коммивояжёра (TSP - Traveling Salesman Problem) заключается в нахождении кратчайшего маршрута, проходящего через все города ровно по одному разу и возвращающегося в начальный город. Это одна из самых известных NP-трудных задач комбинаторной оптимизации.

## Формальное определение

### Задача
Дано: полный граф G = (V, E) с весами рёбер w: E → R+
Требуется: найти гамильтонов цикл минимального веса

### Теорема о NP-трудности
Задача коммивояжёра является NP-трудной.

### Доказательство
1. Задача о гамильтоновом цикле сводится к TSP
2. Задача о гамильтоновом цикле является NP-полной
3. Следовательно, TSP является NP-трудной

## Точные алгоритмы

### Полный перебор

#### Теорема
Полный перебор находит оптимальное решение за время O(n!).

#### Доказательство
1. Количество возможных перестановок n вершин равно n!
2. Для каждой перестановки нужно вычислить длину пути за O(n)
3. Итоговая сложность O(n! * n) = O(n!)

#### Реализация

```python
def brute_force_tsp(graph: Graph) -> Tuple[List[int], float]:
    """Полный перебор для задачи коммивояжёра"""
    min_path = float('inf')
    best_path = []
    
    # Перебираем все перестановки вершин
    for perm in permutations(range(graph.n)):
        current_length = 0
        for i in range(len(perm)):
            current_length += graph.get_weight(perm[i], perm[(i + 1) % len(perm)])
            
        if current_length < min_path:
            min_path = current_length
            best_path = list(perm)
            
    return best_path, min_path
```

### Метод ветвей и границ

#### Теорема
Метод ветвей и границ с оценкой через МОД имеет сложность O(2^n).

#### Доказательство
1. Количество подзадач не превышает 2^n
2. Для каждой подзадачи вычисление оценки занимает O(n²)
3. Итоговая сложность O(2^n * n²) = O(2^n)

#### Реализация

```python
def branch_and_bound_tsp(graph: Graph) -> Tuple[List[int], float]:
    """Метод ветвей и границ для задачи коммивояжёра"""
    best_path = float('inf')
    best_solution = []
    visited = set()
    
    def mst_bound(current: int) -> float:
        """Вычисление нижней оценки через МОД"""
        unvisited = set(range(graph.n)) - visited - {current}
        if not unvisited:
            return 0
            
        # Находим минимальное остовное дерево
        mst_weight = 0
        edges = []
        for v in unvisited:
            for u in unvisited:
                if u < v:
                    edges.append((graph.get_weight(u, v), u, v))
        heapq.heapify(edges)
        
        # Алгоритм Крускала
        parent = {v: v for v in unvisited}
        def find(v):
            if parent[v] != v:
                parent[v] = find(parent[v])
            return parent[v]
            
        while edges and len(unvisited) > 1:
            w, u, v = heapq.heappop(edges)
            if find(u) != find(v):
                mst_weight += w
                parent[find(u)] = find(v)
                unvisited.remove(u)
                unvisited.remove(v)
                unvisited.add(find(u))
                
        return mst_weight
    
    def branch(vertex: int, current_length: float, current_path: List[int]):
        nonlocal best_path, best_solution
        
        if current_length >= best_path:
            return
            
        if len(visited) == graph.n:
            if current_length < best_path:
                best_path = current_length
                best_solution = current_path.copy()
            return
            
        for next_vertex in range(graph.n):
            if next_vertex not in visited:
                edge_weight = graph.get_weight(vertex, next_vertex)
                if current_length + edge_weight + mst_bound(next_vertex) < best_path:
                    visited.add(next_vertex)
                    current_path.append(next_vertex)
                    branch(next_vertex, current_length + edge_weight, current_path)
                    current_path.pop()
                    visited.remove(next_vertex)
    
    # Начинаем с вершины 0
    visited.add(0)
    branch(0, 0, [0])
    return best_solution, best_path
```

## Приближённые алгоритмы

### 2-приближённый алгоритм

#### Теорема
Алгоритм, основанный на МОД и DFS, даёт 2-приближение для метрического TSP.

#### Доказательство
1. Вес МОД не превосходит веса оптимального гамильтонова цикла
2. DFS-обход МОД проходит каждое ребро дважды
3. Следовательно, длина полученного пути не более чем в 2 раза больше оптимальной

#### Реализация

```python
def approximate_tsp(graph: Graph) -> Tuple[List[int], float]:
    """2-приближённый алгоритм для задачи коммивояжёра"""
    def find_mst() -> List[Tuple[int, int]]:
        """Построение минимального остовного дерева"""
        edges = []
        for u in range(graph.n):
            for v in range(u + 1, graph.n):
                edges.append((graph.get_weight(u, v), u, v))
        heapq.heapify(edges)
        
        parent = {v: v for v in range(graph.n)}
        def find(v):
            if parent[v] != v:
                parent[v] = find(parent[v])
            return parent[v]
            
        mst = []
        while edges and len(mst) < graph.n - 1:
            w, u, v = heapq.heappop(edges)
            if find(u) != find(v):
                mst.append((u, v))
                parent[find(u)] = find(v)
                
        return mst
        
    def dfs_traversal(mst: List[Tuple[int, int]]) -> List[int]:
        """Обход дерева в порядке DFS"""
        adj = [[] for _ in range(graph.n)]
        for u, v in mst:
            adj[u].append(v)
            adj[v].append(u)
            
        visited = set()
        path = []
        
        def dfs(v: int):
            visited.add(v)
            path.append(v)
            for u in adj[v]:
                if u not in visited:
                    dfs(u)
                    
        dfs(0)
        return path
        
    mst = find_mst()
    path = dfs_traversal(mst)
    length = sum(graph.get_weight(path[i], path[(i + 1) % len(path)]) 
                for i in range(len(path)))
                
    return path, length
```

### Локальный поиск

#### 2-opt алгоритм

##### Теорема
2-opt локальный поиск находит локальный оптимум за O(n²) итераций.

##### Доказательство
1. На каждой итерации длина пути уменьшается
2. Всего возможно O(n²) различных путей
3. Следовательно, алгоритм сходится за O(n²) итераций

##### Реализация

```python
def local_search_2opt(graph: Graph, initial_path: Optional[List[int]] = None) -> Tuple[List[int], float]:
    """Локальный поиск с 2-окружением"""
    if initial_path is None:
        path = list(range(graph.n))
        random.shuffle(path)
    else:
        path = initial_path.copy()
        
    def calculate_length(p: List[int]) -> float:
        length = 0
        for i in range(len(p)):
            length += graph.get_weight(p[i], p[(i + 1) % len(p)])
        return length
        
    def swap_edges(p: List[int], i: int, j: int) -> List[int]:
        """Перестановка рёбер в пути"""
        return p[:i+1] + p[i+1:j+1][::-1] + p[j+1:]
        
    current_length = calculate_length(path)
    improved = True
    
    while improved:
        improved = False
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                new_path = swap_edges(path, i, j)
                new_length = calculate_length(new_path)
                
                if new_length < current_length:
                    path = new_path
                    current_length = new_length
                    improved = True
                    break
            if improved:
                break
                
    return path, current_length
```

### Имитация отжига

#### Теорема
Имитация отжига сходится к глобальному оптимуму с вероятностью 1 при бесконечном времени работы.

#### Доказательство
1. Алгоритм может достичь любого состояния с ненулевой вероятностью
2. Вероятность принятия улучшающего хода стремится к 1 при T → 0
3. Следовательно, алгоритм сходится к глобальному оптимуму

#### Реализация

```python
def simulated_annealing_tsp(graph: Graph, 
                          initial_temp: float = 100.0,
                          final_temp: float = 0.1,
                          cooling_rate: float = 0.95) -> Tuple[List[int], float]:
    """Имитация отжига для задачи коммивояжёра"""
    def calculate_length(p: List[int]) -> float:
        length = 0
        for i in range(len(p)):
            length += graph.get_weight(p[i], p[(i + 1) % len(p)])
        return length
        
    def generate_neighbor(path: List[int]) -> List[int]:
        """Генерация соседнего решения"""
        i, j = random.sample(range(len(path)), 2)
        new_path = path.copy()
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path
        
    current_path = list(range(graph.n))
    random.shuffle(current_path)
    current_length = calculate_length(current_path)
    
    best_path = current_path.copy()
    best_length = current_length
    
    T = initial_temp
    while T > final_temp:
        new_path = generate_neighbor(current_path)
        new_length = calculate_length(new_path)
        
        delta = new_length - current_length
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_path = new_path
            current_length = new_length
            
            if current_length < best_length:
                best_path = current_path.copy()
                best_length = current_length
                
        T *= cooling_rate
        
    return best_path, best_length
```

## Сложность алгоритмов

### Точные алгоритмы
- Полный перебор: O(n!)
- Метод ветвей и границ: O(2^n)

### Приближённые алгоритмы
- 2-приближённый алгоритм: O(n² log n)
- Локальный поиск: O(n²) на итерацию
- Имитация отжига: O(n²) на итерацию

## Применения

### Логистика
- Планирование маршрутов доставки
- Оптимизация работы курьеров
- Управление парком транспорта

### Производство
- Планирование последовательности операций
- Оптимизация движения роботов
- Управление конвейерными линиями

### Электроника
- Проектирование печатных плат
- Оптимизация сверления отверстий
- Планирование тестирования микросхем 