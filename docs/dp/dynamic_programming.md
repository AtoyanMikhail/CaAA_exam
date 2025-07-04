# Динамическое программирование

## Введение

Динамическое программирование - это метод решения задач, основанный на разбиении задачи на подзадачи и использовании их решений для построения решения исходной задачи.

### Основные принципы

1. Оптимальная подструктура: решение задачи можно получить из решений её подзадач
2. Перекрывающиеся подзадачи: одни и те же подзадачи решаются многократно
3. Мемоизация: сохранение решений подзадач для повторного использования

## Числа Фибоначчи

### Наивное решение

```python
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
```

**Сложность**: O(2ⁿ) времени, O(n) памяти (стек вызовов)

### Динамическое программирование

```python
def fib_dp(n):
    if n <= 1:
        return n
        
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
        
    return dp[n]
```

**Сложность**: O(n) времени, O(n) памяти

### Оптимизация памяти

```python
def fib_optimized(n):
    if n <= 1:
        return n
        
    prev, curr = 0, 1
    for i in range(2, n + 1):
        prev, curr = curr, prev + curr
        
    return curr
```

**Сложность**: O(n) времени, O(1) памяти

## Максимальная возрастающая подпоследовательность

### Формальная постановка

Дана последовательность a₁, a₂, ..., aₙ. Найти длину наибольшей возрастающей подпоследовательности.

### Сведение к графу

**Теорема**: Задача о максимальной возрастающей подпоследовательности сводится к поиску длиннейшего пути в ациклическом графе.

**Доказательство**:
1. Построим граф G = (V, E), где:
   - V = {1, 2, ..., n}
   - (i, j) ∈ E, если i < j и aᵢ < aⱼ
2. Длина пути в G соответствует длине возрастающей подпоследовательности
3. Наибольший путь даёт максимальную подпоследовательность

```python
def lis_graph(sequence):
    n = len(sequence)
    # Строим граф
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sequence[i] < sequence[j]:
                graph[i].append(j)
                
    # Находим длиннейший путь
    dp = [1] * n
    for i in range(n-1, -1, -1):
        for j in graph[i]:
            dp[i] = max(dp[i], dp[j] + 1)
            
    return max(dp)
```

### Выделение подзадачи

**Теорема**: Пусть dp[i] - длина наибольшей возрастающей подпоследовательности, заканчивающейся в позиции i. Тогда:
dp[i] = max(dp[j] + 1) для всех j < i, где aⱼ < aᵢ

**Доказательство**:
1. Если aⱼ < aᵢ, то можно добавить aᵢ к подпоследовательности, заканчивающейся в j
2. Нужно выбрать максимальную такую подпоследовательность
3. Итого: dp[i] = max(dp[j] + 1) для всех j < i, где aⱼ < aᵢ

```python
def lis_dp(sequence):
    n = len(sequence)
    dp = [1] * n
    
    for i in range(n):
        for j in range(i):
            if sequence[j] < sequence[i]:
                dp[i] = max(dp[i], dp[j] + 1)
                
    return max(dp)
```

**Сложность**: O(n²) времени, O(n) памяти

### Оптимизация с помощью бинарного поиска

```python
def lis_optimized(sequence):
    n = len(sequence)
    dp = [float('inf')] * (n + 1)
    dp[0] = float('-inf')
    
    for x in sequence:
        # Находим позицию для вставки x
        pos = bisect.bisect_right(dp, x)
        if dp[pos-1] < x < dp[pos]:
            dp[pos] = x
            
    # Находим длину наибольшей подпоследовательности
    for i in range(n, 0, -1):
        if dp[i] != float('inf'):
            return i
            
    return 0
```

**Сложность**: O(n log n) времени, O(n) памяти

## Задача о порядке перемножения матриц

### Формальная постановка

Даны матрицы A₁, A₂, ..., Aₙ размеров p₀×p₁, p₁×p₂, ..., pₙ₋₁×pₙ. Найти порядок перемножения, минимизирующий общее число операций.

### Выделение подзадачи

**Теорема**: Пусть dp[i][j] - минимальное число операций для перемножения матриц Aᵢ, Aᵢ₊₁, ..., Aⱼ. Тогда:
dp[i][j] = min(dp[i][k] + dp[k+1][j] + pᵢ₋₁×pₖ×pⱼ) для всех k от i до j-1

**Доказательство**:
1. Разобьём перемножение на две части: (Aᵢ...Aₖ) и (Aₖ₊₁...Aⱼ)
2. Число операций = операции для первой части + операции для второй части + операции для перемножения результатов
3. Итого: dp[i][j] = min(dp[i][k] + dp[k+1][j] + pᵢ₋₁×pₖ×pⱼ)

```python
def matrix_chain_multiplication(dimensions):
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]
    
    # l - длина подпоследовательности
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                dp[i][j] = min(dp[i][j], cost)
                
    return dp[0][n-1]
```

**Сложность**: O(n³) времени, O(n²) памяти

### Восстановление решения

```python
def matrix_chain_multiplication_with_solution(dimensions):
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]
    solution = [[0] * n for _ in range(n)]
    
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    solution[i][j] = k
                    
    def get_solution(i, j):
        if i == j:
            return f"A{i+1}"
        k = solution[i][j]
        return f"({get_solution(i, k)} × {get_solution(k+1, j)})"
        
    return dp[0][n-1], get_solution(0, n-1)
```

## Общие принципы решения задач ДП

### Шаги решения

1. Определить подзадачи
2. Записать рекуррентное соотношение
3. Определить порядок вычисления
4. Реализовать решение
5. Оптимизировать память (если возможно)

### Типичные задачи

1. Задачи на последовательности
   - Наибольшая общая подпоследовательность
   - Расстояние Левенштейна
   - Разбиение строки на слова

2. Задачи на графы
   - Наибольший путь в ациклическом графе
   - Задача коммивояжёра
   - Раскраска графа

3. Задачи на оптимизацию
   - Рюкзак
   - Расписание
   - Размещение ресурсов 