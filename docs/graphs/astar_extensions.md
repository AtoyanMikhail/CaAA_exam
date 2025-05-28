# Расширения алгоритма A*

## Введение

Алгоритм A* является одним из самых эффективных алгоритмов поиска кратчайшего пути. Однако для больших графов его производительность может быть улучшена с помощью различных оптимизаций и расширений.

## Базовый алгоритм A*

### Теорема
Алгоритм A* находит оптимальный путь, если эвристическая функция допустима (не переоценивает).

### Доказательство
1. Эвристическая функция h(n) не переоценивает стоимость пути до цели
2. Функция оценки f(n) = g(n) + h(n) не переоценивает общую стоимость
3. A* всегда выбирает вершину с минимальной оценкой f(n)
4. Следовательно, первый найденный путь до цели оптимален

### Реализация

```python
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
```

## Оптимизация ALT (A*, Landmarks, Triangle inequality)

### Теорема
Эвристическая функция ALT даёт более точную оценку, чем манхэттенское расстояние.

### Доказательство
1. Для каждого ориентира l выполняется неравенство треугольника
2. |d(s,l) - d(t,l)| ≤ d(s,t) ≤ d(s,l) + d(t,l)
3. Максимум по всем ориентирам даёт лучшую оценку

### Реализация

```python
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

def select_landmarks(graph: Graph, num_landmarks: int) -> List[int]:
    """Выбор ориентиров для алгоритма ALT"""
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
```

## Оптимизация REACH

### Теорема
Значение достижимости вершины даёт нижнюю оценку на длину любого пути через неё.

### Доказательство
1. Для любого пути s → v → t выполняется d(s,t) ≥ max(d(s,v), d(v,t))
2. Значение достижимости есть минимум max(d(s,v), d(v,t)) по всем s,t
3. Следовательно, это нижняя оценка на длину любого пути через v

### Реализация

```python
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
```

## Оптимизация слияния рёбер

### Теорема
Слияние рёбер не меняет кратчайшие пути в графе.

### Доказательство
1. Если d(u,v) > d(u,w) + d(w,v), то ребро (u,v) не используется в кратчайших путях
2. Замена веса ребра (u,v) на d(u,w) + d(w,v) сохраняет все кратчайшие пути
3. Следовательно, слияние рёбер безопасно

### Реализация

```python
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
```

## Сложность алгоритмов

### Базовый A*
- Время: O(|E| log |V|)
- Память: O(|V|)

### A* с ALT
- Предобработка: O(|V|²)
- Поиск: O(|E| log |V|)
- Память: O(|V|)

### A* с REACH
- Предобработка: O(|V|³)
- Поиск: O(|E| log |V|)
- Память: O(|V|)

### Слияние рёбер
- Время: O(|V| * |E|²)
- Память: O(1)

## Применения

### Навигация
- Поиск маршрутов в картах
- Планирование движения роботов
- Управление беспилотниками

### Игры
- Поиск пути в стратегиях
- Искусственный интеллект
- Процедурная генерация

### Сети
- Маршрутизация пакетов
- Оптимизация трафика
- Балансировка нагрузки 