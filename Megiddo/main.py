#!/usr/bin/env python3
"""
Главный файл для выполнения задачи:
1. Запуск megiddo_optimized.py на файле test20.txt
2. Извлечение прямых, дающих точки пересечения
3. Применение преобразования из transform_new.py
4. Вывод преобразованных прямых
5. Запуск граем метода на преобразованных прямых для получения вершин
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from megiddo_optimized import solve_megiddo, read_input
from transform_new import transform_inequalities
from grahem import Grahem, AngleHull

def main():
    # 1. Чтение данных из файла test50.txt
    filename = os.path.join(os.path.dirname(__file__), "input2.txt")
    p, q, constraints = read_input(filename)
    
    print(f"Задача: max {p}*x + {q}*y")
    print(f"Число ограничений: {len(constraints)}")
    
    # 2. Запуск алгоритма Мегиддо
    print("\nЗапуск алгоритма Мегиддо...")
    x, y, val, status, intersecting_constraints = solve_megiddo(constraints, p, q)
    
    if status != 'optimal':
        print(f"Статус решения: {status}")
        return
    
    print(f"\nРешение найдено: x={x:.6f}, y={y:.6f}, max={val:.6f}")
    
    # 3. Вывод ограничений, дающих точку пересечения
    if intersecting_constraints and len(intersecting_constraints) >= 2:
        print("\nНеравенства, дающие точку пересечения:")
        A1, B1, C1 = intersecting_constraints[0]
        A2, B2, C2 = intersecting_constraints[1]
        print(f"  1: {A1}*x + {B1}*y = {C1}")
        print(f"  2: {A2}*x + {B2}*y = {C2}")
        
        # 4. Применение преобразования
        print("\nПрименяем преобразование...")
        alpha, beta, gamma, beta2, identity_matrix = transform_inequalities(A1, B1, C1, A2, B2, C2)
        
        print(f"\nПреобразованные прямые:")
        print(f"  1: {alpha}*x + {beta}*y = {gamma}")
        print(f"  2: -y = {beta2}")  # beta2 будет отрицательным, так как это коэффициент при y
        
        # Вывод матрицы преобразования
        print(f"\nМатрица преобразования:")
        print(identity_matrix)
        
        # 5. Запуск граем метода на преобразованных прямых
        print("\n" + "="*80)
        print("ПОЛУЧЕНИЕ ВЕРШИН УГЛОВОГО ПОЛИЭДРА МЕТОДОМ ГРЕХЕМА")
        print("="*80)
        
        a = int(alpha)
        b = int(beta)
        c = int(gamma)
        
        print(f"Вызываем граем метод с a={a}, b={b}, c={c}")
        print()
        
        # Метод Грэхема
        print("--- Метод Грэхема ---")
        vertices_grahem = Grahem(a, b, c)
        print("Вершины (Метод Грэхема):")
        print(vertices_grahem)
        
        # Метод AngleHull
        print("\n--- Метод AngleHull ---")
        P, H = AngleHull(a, b, c)
        print("Вершины (AngleHull):")
        print(P.T)
        print("\nПерспективные направления (H):")
        print(H)
        
        print("\n" + "="*80)
        print("ГОТОВО")
        print("="*80)
    else:
        print("Не удалось определить ограничения, дающие точку пересечения")
        return

if __name__ == "__main__":
    main()
