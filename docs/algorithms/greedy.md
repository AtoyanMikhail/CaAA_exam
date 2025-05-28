# Жадные алгоритмы

## Введение

Жадные алгоритмы - это алгоритмы, которые на каждом шаге принимают локально оптимальное решение в надежде получить глобально оптимальное решение.

### Основные принципы

1. Жадный выбор: на каждом шаге выбираем локально оптимальное решение
2. Оптимальная подструктура: оптимальное решение задачи содержит оптимальные решения подзадач
3. Отсутствие отмены: принятые решения не отменяются

## Минимальное остовное дерево

### Алгоритм Краскала

```python
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
```

### Алгоритм Прима

```python
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
```

## Оптимальное расписание

### Задача с весами

```python
@dataclass
class Job:
    """Задача с временем выполнения и дедлайном"""
    id: int
    duration: int  # время выполнения
    deadline: int  # дедлайн
    weight: float  # вес/приоритет задачи

def schedule_with_weights(jobs: List[Job]) -> List[Tuple[int, int]]:
    """
    Реализация алгоритма для задачи с весами.
    Сортирует задачи по убыванию веса/времени выполнения.
    
    Args:
        jobs: Список задач
        
    Returns:
        Список пар (id_задачи, время_начала) - оптимальное расписание
    """
    # Сортируем задачи по убыванию веса/времени выполнения
    sorted_jobs = sorted(jobs, key=lambda x: x.weight/x.duration, reverse=True)
    
    current_time = 0
    schedule = []
    
    for job in sorted_jobs:
        schedule.append((job.id, current_time))
        current_time += job.duration
        
    return schedule
```

### Задача с дедлайнами

```python
def schedule_with_deadlines(jobs: List[Job]) -> List[Tuple[int, int]]:
    """
    Реализация алгоритма для задачи с дедлайнами.
    Сортирует задачи по дедлайнам и пытается выполнить их как можно раньше.
    
    Args:
        jobs: Список задач
        
    Returns:
        Список пар (id_задачи, время_начала) - оптимальное расписание
    """
    # Сортируем задачи по дедлайнам
    sorted_jobs = sorted(jobs, key=attrgetter('deadline'))
    
    current_time = 0
    schedule = []
    
    for job in sorted_jobs:
        # Если задача не может быть выполнена до дедлайна, пропускаем её
        if current_time + job.duration <= job.deadline:
            schedule.append((job.id, current_time))
            current_time += job.duration
            
    return schedule
```

## Кодирование Хаффмана

### Построение дерева

```python
@dataclass
class HuffmanNode:
    """Узел дерева Хаффмана"""
    char: str = None  # символ (только для листьев)
    freq: int = 0     # частота
    left: 'HuffmanNode' = None
    right: 'HuffmanNode' = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text: str) -> HuffmanNode:
    """
    Строит дерево Хаффмана для заданного текста.
    
    Args:
        text: Входной текст
        
    Returns:
        Корень дерева Хаффмана
    """
    # Подсчитываем частоты символов
    freq = Counter(text)
    
    # Создаем листья дерева
    heap = []
    for char, count in freq.items():
        node = HuffmanNode(char=char, freq=count)
        heappush(heap, node)
    
    # Строим дерево
    while len(heap) > 1:
        # Берем два узла с минимальными частотами
        left = heappop(heap)
        right = heappop(heap)
        
        # Создаем новый узел
        internal = HuffmanNode(
            freq=left.freq + right.freq,
            left=left,
            right=right
        )
        heappush(heap, internal)
    
    return heap[0] if heap else None
```

### Кодирование и декодирование

```python
def build_codes(root: HuffmanNode, code: str = "", codes: Dict[str, str] = None) -> Dict[str, str]:
    """
    Строит таблицу кодов Хаффмана.
    
    Args:
        root: Корень дерева Хаффмана
        code: Текущий код
        codes: Словарь для хранения кодов
        
    Returns:
        Словарь {символ: код}
    """
    if codes is None:
        codes = {}
    
    if root is None:
        return codes
    
    # Если это лист, сохраняем код
    if root.char is not None:
        codes[root.char] = code if code else "0"
    
    # Рекурсивно обходим левое и правое поддерево
    build_codes(root.left, code + "0", codes)
    build_codes(root.right, code + "1", codes)
    
    return codes

def encode(text: str, codes: Dict[str, str]) -> str:
    """
    Кодирует текст с помощью кодов Хаффмана.
    
    Args:
        text: Входной текст
        codes: Словарь кодов Хаффмана
        
    Returns:
        Закодированный текст
    """
    return ''.join(codes[char] for char in text)

def decode(encoded: str, root: HuffmanNode) -> str:
    """
    Декодирует текст, закодированный с помощью кодов Хаффмана.
    
    Args:
        encoded: Закодированный текст
        root: Корень дерева Хаффмана
        
    Returns:
        Декодированный текст
    """
    if not encoded:
        return ""
    
    result = []
    current = root
    
    for bit in encoded:
        if bit == "0":
            current = current.left
        else:
            current = current.right
            
        # Если достигли листа, добавляем символ и возвращаемся к корню
        if current.char is not None:
            result.append(current.char)
            current = root
    
    return ''.join(result)
```

## Сложность алгоритмов

### Минимальное остовное дерево
- Алгоритм Краскала: O(E log E) времени, O(V) памяти
- Алгоритм Прима: O(E log V) времени, O(V) памяти

### Оптимальное расписание
- Задача с весами: O(n log n) времени, O(n) памяти
- Задача с дедлайнами: O(n log n) времени, O(n) памяти

### Кодирование Хаффмана
- Построение дерева: O(n log n) времени, O(n) памяти
- Кодирование: O(n) времени, O(1) памяти
- Декодирование: O(n) времени, O(1) памяти

## Применения

### Минимальное остовное дерево
- Проектирование сетей
- Кластеризация данных
- Оптимизация маршрутов

### Оптимальное расписание
- Планирование задач
- Распределение ресурсов
- Управление проектами

### Кодирование Хаффмана
- Сжатие данных
- Передача информации
- Оптимизация хранения 