# Суффиксные деревья и массивы

## Суффиксное дерево

### Мотивация

Суффиксное дерево - это структура данных, позволяющая эффективно решать множество задач на строках:
- Поиск подстроки
- Поиск наибольшей общей подстроки
- Поиск всех вхождений образца
- Поиск наибольшего повторяющегося подстроки

### Определение

**Определение**: Суффиксное дерево для строки s - это сжатое префиксное дерево, содержащее все суффиксы строки s.

**Свойства**:
1. Каждый путь от корня до листа соответствует суффиксу
2. Каждое ребро помечено подстрокой
3. Все внутренние вершины (кроме корня) имеют не менее двух детей

### Теорема о размере

**Теорема**: Суффиксное дерево для строки длины n содержит O(n) вершин.

**Доказательство**:
1. Число листьев равно n (по одному на каждый суффикс)
2. Число внутренних вершин не превосходит n-1
3. Итого: O(n) вершин

### Поиск подстроки

**Теорема**: Поиск подстроки в суффиксном дереве выполняется за O(m) времени, где m - длина искомой подстроки.

**Доказательство**:
1. Начинаем с корня
2. На каждом шаге ищем ребро, начинающееся с нужного символа
3. Спускаемся по этому ребру
4. Итого: O(m) операций

```python
class SuffixTreeNode:
    def __init__(self):
        self.children = {}  # символ -> (вершина, начало, конец)
        self.suffix_link = None
        self.start = -1
        self.end = -1

def find_substring(root, pattern):
    node = root
    i = 0
    
    while i < len(pattern):
        if pattern[i] not in node.children:
            return False
            
        node, start, end = node.children[pattern[i]]
        j = start
        
        while i < len(pattern) and j < end:
            if pattern[i] != text[j]:
                return False
            i += 1
            j += 1
            
    return True
```

## Алгоритм Укконена

### Идея алгоритма

Алгоритм Укконена строит суффиксное дерево за линейное время, используя следующие оптимизации:
1. Неявные суффиксные деревья
2. Суффиксные ссылки
3. Правило 3 остановки

### Теорема о сложности

**Теорема**: Алгоритм Укконена строит суффиксное дерево за O(n) времени.

**Доказательство**:
1. Каждый суффикс добавляется за O(1) амортизированного времени
2. Всего n суффиксов
3. Итого: O(n) операций

```python
def build_suffix_tree(text):
    root = SuffixTreeNode()
    root.suffix_link = root
    
    # Глобальные переменные
    active_node = root
    active_edge = 0
    active_length = 0
    remaining = 0
    
    for i in range(len(text)):
        remaining += 1
        last_new_node = None
        
        while remaining > 0:
            if active_length == 0:
                active_edge = i
                
            if text[active_edge] not in active_node.children:
                # Правило 2: создаём новую вершину
                active_node.children[text[active_edge]] = (SuffixTreeNode(), i, len(text))
                if last_new_node:
                    last_new_node.suffix_link = active_node
                last_new_node = None
            else:
                # Продолжаем по существующему ребру
                next_node, start, end = active_node.children[text[active_edge]]
                edge_length = end - start
                
                if active_length >= edge_length:
                    active_node = next_node
                    active_length -= edge_length
                    active_edge += edge_length
                    continue
                    
                if text[start + active_length] == text[i]:
                    # Правило 3: останавливаемся
                    active_length += 1
                    if last_new_node:
                        last_new_node.suffix_link = active_node
                    break
                    
                # Правило 2: разбиваем ребро
                split_node = SuffixTreeNode()
                active_node.children[text[active_edge]] = (split_node, start, start + active_length)
                split_node.children[text[start + active_length]] = (next_node, start + active_length, end)
                split_node.children[text[i]] = (SuffixTreeNode(), i, len(text))
                
                if last_new_node:
                    last_new_node.suffix_link = split_node
                last_new_node = split_node
                
            remaining -= 1
            
            if active_node == root and active_length > 0:
                active_length -= 1
                active_edge = i - remaining + 1
            else:
                active_node = active_node.suffix_link
                
    return root
```

## Суффиксный массив

### Мотивация

Суффиксный массив - это более компактная альтернатива суффиксному дереву, которая:
- Требует меньше памяти
- Проще в реализации
- Позволяет решать те же задачи

### Определение

**Определение**: Суффиксный массив для строки s - это массив индексов всех суффиксов строки s, отсортированных в лексикографическом порядке.

### Теорема о размере

**Теорема**: Суффиксный массив для строки длины n занимает O(n) памяти.

**Доказательство**:
1. Массив содержит n индексов
2. Каждый индекс занимает O(1) памяти
3. Итого: O(n) памяти

### Построение

**Теорема**: Суффиксный массив можно построить за O(n log n) времени.

**Доказательство**:
1. Сортируем все суффиксы
2. Используем сортировку подсчётом для каждого символа
3. Итого: O(n log n) операций

```python
def build_suffix_array(text):
    n = len(text)
    # Начальная сортировка по первому символу
    sa = list(range(n))
    sa.sort(key=lambda i: text[i])
    
    # Сортируем по префиксам длины 2^k
    k = 1
    while k < n:
        # Сортируем по парам (rank[i], rank[i+k])
        sa.sort(key=lambda i: (text[i:i+k], text[i+k:i+2*k]))
        k *= 2
        
    return sa
```

### Поиск подстроки

**Теорема**: Поиск подстроки в суффиксном массиве выполняется за O(m log n) времени, где m - длина искомой подстроки, n - длина текста.

**Доказательство**:
1. Используем бинарный поиск
2. На каждом шаге сравниваем подстроки за O(m)
3. Итого: O(m log n) операций

```python
def find_substring_sa(text, pattern, sa):
    left = 0
    right = len(sa)
    
    while left < right:
        mid = (left + right) // 2
        suffix = text[sa[mid]:]
        
        if pattern < suffix:
            right = mid
        elif pattern > suffix:
            left = mid + 1
        else:
            return sa[mid]
            
    return -1
```

## Сравнение структур

### Преимущества суффиксного дерева
- Быстрый поиск подстроки: O(m)
- Простое нахождение наибольшей общей подстроки
- Эффективное нахождение всех вхождений

### Преимущества суффиксного массива
- Меньше памяти: O(n) против O(n) для дерева
- Проще в реализации
- Эффективное нахождение наибольшего повторяющегося подстроки

## Применения

### Поиск подстрок
- Точный поиск образца
- Поиск всех вхождений
- Поиск наибольшей общей подстроки

### Сжатие данных
- Алгоритм LZ77
- Поиск повторяющихся подстрок
- Оптимизация сжатия

### Биоинформатика
- Поиск последовательностей в ДНК
- Анализ белковых структур
- Сравнение геномов 