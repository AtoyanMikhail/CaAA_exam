# Алгоритм Ахо-Корасик

## Введение

Алгоритм Ахо-Корасик - это алгоритм поиска множества образцов в тексте, использующий препроцессинг для эффективного поиска.

### Основные определения

1. Бор (префиксное дерево): дерево, где каждое ребро помечено символом, а путь от корня до вершины соответствует префиксу некоторого образца
2. Суффиксная ссылка: для вершины v это ссылка на вершину u, такую что строка, соответствующая u, является наибольшим собственным суффиксом строки, соответствующей v
3. Сжатые суффиксные ссылки: суффиксная ссылка, ведущая в терминальную вершину

## Задача точного поиска набора образцов

### Формальная постановка

Даны:
- Текст t = t₁t₂...tₙ
- Множество образцов P = {p₁, p₂, ..., pₖ}

Найти:
- Все вхождения всех образцов в текст

## Бор

### Теорема о построении бора

**Теорема**: Бор для множества образцов P можно построить за O(Σ|p|) времени и памяти, где |p| - суммарная длина всех образцов.

**Доказательство**:
1. Для каждого образца нужно пройти по пути в боре
2. Каждое ребро обрабатывается не более одного раза
3. Итого: O(Σ|p|) операций

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # символ -> вершина
        self.is_terminal = False  # является ли вершина терминальной
        self.pattern_index = -1  # индекс образца, если терминальная
        self.suffix_link = None  # суффиксная ссылка
        self.compressed_suffix_link = None  # сжатая суффиксная ссылка

def build_trie(patterns):
    root = TrieNode()
    
    for i, pattern in enumerate(patterns):
        node = root
        for char in pattern:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_terminal = True
        node.pattern_index = i
        
    return root
```

## Суффиксные ссылки

### Теорема о построении суффиксных ссылок

**Теорема**: Суффиксные ссылки для бора можно построить за O(Σ|p|) времени.

**Доказательство**:
1. Для каждой вершины v суффиксная ссылка ведёт в вершину u, такую что:
   - u соответствует наибольшему собственному суффиксу v
   - u уже обработана (используем BFS)
2. Каждая вершина обрабатывается не более одного раза
3. Итого: O(Σ|p|) операций

```python
def build_suffix_links(root):
    queue = []
    
    # Инициализация для вершин глубины 1
    for char, child in root.children.items():
        child.suffix_link = root
        queue.append(child)
        
    # BFS для остальных вершин
    while queue:
        node = queue.pop(0)
        
        for char, child in node.children.items():
            # Ищем суффиксную ссылку для child
            suffix = node.suffix_link
            while suffix is not root and char not in suffix.children:
                suffix = suffix.suffix_link
                
            if char in suffix.children:
                child.suffix_link = suffix.children[char]
            else:
                child.suffix_link = root
                
            queue.append(child)
            
    return root
```

### Теорема о сжатых суффиксных ссылках

**Теорема**: Сжатые суффиксные ссылки можно построить за O(Σ|p|) времени.

**Доказательство**:
1. Для каждой вершины v сжатая суффиксная ссылка ведёт в ближайшую терминальную вершину
2. Используем уже построенные суффиксные ссылки
3. Каждая вершина обрабатывается не более одного раза
4. Итого: O(Σ|p|) операций

```python
def build_compressed_suffix_links(root):
    queue = [root]
    
    while queue:
        node = queue.pop(0)
        
        # Если текущая вершина терминальная, её сжатая ссылка - она сама
        if node.is_terminal:
            node.compressed_suffix_link = node
        else:
            # Иначе ищем ближайшую терминальную вершину
            suffix = node.suffix_link
            while suffix is not root and not suffix.is_terminal:
                suffix = suffix.suffix_link
            node.compressed_suffix_link = suffix
            
        queue.extend(node.children.values())
        
    return root
```

## Алгоритм поиска

### Теорема о корректности

**Теорема**: Алгоритм Ахо-Корасик находит все вхождения всех образцов в текст.

**Доказательство**:
1. Бор содержит все образцы
2. Суффиксные ссылки позволяют не пропустить вхождения
3. Сжатые суффиксные ссылки позволяют быстро найти все вхождения

### Реализация алгоритма

```python
def aho_corasick(text, patterns):
    # Построение бора
    root = build_trie(patterns)
    
    # Построение суффиксных ссылок
    root = build_suffix_links(root)
    
    # Построение сжатых суффиксных ссылок
    root = build_compressed_suffix_links(root)
    
    # Поиск образцов
    result = [[] for _ in range(len(patterns))]
    node = root
    
    for i, char in enumerate(text):
        # Переходим по суффиксным ссылкам, пока не найдём переход по char
        while node is not root and char not in node.children:
            node = node.suffix_link
            
        if char in node.children:
            node = node.children[char]
            
        # Проверяем все вхождения через сжатые суффиксные ссылки
        current = node
        while current is not root:
            if current.is_terminal:
                result[current.pattern_index].append(i - len(patterns[current.pattern_index]) + 1)
            current = current.compressed_suffix_link
            
    return result
```

## Сложность алгоритма

### Теорема о сложности

**Теорема**: Алгоритм Ахо-Корасик имеет сложность:
- Препроцессинг: O(Σ|p|) времени и памяти
- Поиск: O(n + k) времени, где k - число вхождений

**Доказательство**:
1. Препроцессинг:
   - Построение бора: O(Σ|p|)
   - Построение суффиксных ссылок: O(Σ|p|)
   - Построение сжатых ссылок: O(Σ|p|)
   
2. Поиск:
   - Каждый символ текста обрабатывается не более одного раза
   - Каждое вхождение обрабатывается не более одного раза
   - Итого: O(n + k)

## Применения

### Поиск множества образцов
- Поиск ключевых слов в тексте
- Фильтрация спама
- Анализ логов

### Биоинформатика
- Поиск множества последовательностей в ДНК
- Анализ белковых структур
- Сравнение геномов

### Сжатие данных
- Поиск повторяющихся подстрок
- Оптимизация сжатия
- Дедупликация данных 