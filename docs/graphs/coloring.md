# Алгоритмы раскраски графа

## Введение

Раскраска графа - это задача присвоения цветов вершинам графа так, чтобы никакие две смежные вершины не имели одинаковый цвет. Минимальное количество цветов, необходимое для раскраски графа, называется хроматическим числом.

## Теорема о раскраске

### Теорема
Для любого графа G хроматическое число χ(G) удовлетворяет неравенству:
1 ≤ χ(G) ≤ Δ(G) + 1, где Δ(G) - максимальная степень вершины в графе.

### Доказательство
1. Нижняя оценка: граф всегда можно раскрасить хотя бы в один цвет
2. Верхняя оценка: жадный алгоритм использует не более Δ(G) + 1 цветов

## Алгоритмы раскраски

### Полный перебор для 3-раскраски

```python
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
```

### Перебор с ограничением в 2 цвета

```python
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
```

### Перебор подмножеств

```python
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
```

### Вероятностный алгоритм

```python
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
```

## Сведение к задаче выполнимости (SAT)

### Теорема
Задача 3-раскраски графа полиномиально сводится к задаче выполнимости булевых формул (SAT).

### Доказательство
1. Каждой вершине и цвету сопоставляется булева переменная
2. Формула в CNF кодирует условия допустимой раскраски
3. Размер формулы полиномиален от размера графа

### Реализация

```python
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
```

## Сложность алгоритмов

### Полный перебор
- Время: O(3^n)
- Память: O(n)

### Перебор с ограничением
- Время: O(2^n)
- Память: O(n)

### Перебор подмножеств
- Время: O(3^(n/3))
- Память: O(n)

### Вероятностный алгоритм
- Время: O(n * max_attempts)
- Память: O(n)

### Сведение к SAT
- Время: O(n + m)
- Память: O(n + m)

## Применения

### Расписание
- Составление расписания экзаменов
- Планирование задач
- Распределение ресурсов

### Регистрация
- Распределение регистров в компиляторах
- Оптимизация памяти
- Параллельные вычисления

### Сети
- Распределение частот
- Маршрутизация
- Балансировка нагрузки 