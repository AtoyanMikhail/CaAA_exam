from typing import Dict, List, Tuple
from heapq import heappush, heappop
from dataclasses import dataclass
from collections import Counter

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

def get_compression_ratio(original: str, encoded: str) -> float:
    """
    Вычисляет степень сжатия.
    
    Args:
        original: Исходный текст
        encoded: Закодированный текст
        
    Returns:
        Степень сжатия (отношение размеров)
    """
    original_size = len(original) * 8  # размер в битах
    encoded_size = len(encoded)
    return original_size / encoded_size

if __name__ == "__main__":
    # Пример использования
    text = "ABRACADABRA"
    
    # Строим дерево и получаем коды
    tree = build_huffman_tree(text)
    codes = build_codes(tree)
    
    print("Коды Хаффмана:")
    for char, code in sorted(codes.items()):
        print(f"{char}: {code}")
    
    # Кодируем и декодируем текст
    encoded = encode(text, codes)
    decoded = decode(encoded, tree)
    
    print(f"\nИсходный текст: {text}")
    print(f"Закодированный текст: {encoded}")
    print(f"Декодированный текст: {decoded}")
    print(f"Степень сжатия: {get_compression_ratio(text, encoded):.2f}") 