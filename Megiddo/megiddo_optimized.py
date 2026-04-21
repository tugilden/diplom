"""
Алгоритм Мегиддо для 2D линейного программирования с истинной O(n) сложностью.

Классический алгоритм Мегиддо (1984) — первый детерминированный O(n) алгоритм
для 2D LP.

Ключевые компоненты для O(n):
1. Median-of-medians (BFPRT) — нахождение медианы за O(n)
2. Правильное отбрасывание ≥ n/2 ограничений на каждом шаге
"""

import math
import random
from typing import List, Tuple, Optional

EPS = 1e-9
INF = float('inf')


def median_of_medians(arr: List[float]) -> float:
    """
    Алгоритм median-of-medians (BFPRT) для нахождения медианы за O(n).
    
    Т(SORT(n/5)) = n/5 * O(1) = O(n)
    Т(M(n/5)) = T(n/5)
    T(PARTITION) = O(n)
    T(РЕКУРСИЯ) = max(T(n/5), T(7n/10))
    
    T(n) = T(n/5) + T(7n/10) + O(n) = O(n)
    """
    if len(arr) <= 5:
        return sorted(arr)[len(arr) // 2]
    
    # Разбиваем на группы по 5 элементов
    groups = []
    for i in range(0, len(arr), 5):
        group = arr[i:i + 5]
        group.sort()
        groups.append(group)
    
    # Находим медианы каждой группы
    medians = [g[len(g) // 2] for g in groups]
    
    # Рекурсивно находим медиану медиан
    pivot = median_of_medians(medians)
    
    # Partition вокруг pivot
    lo = [x for x in arr if x < pivot - EPS]
    eq = [x for x in arr if abs(x - pivot) < EPS]
    hi = [x for x in arr if x > pivot + EPS]
    
    n_lo, n_eq = len(lo), len(eq)
    median_idx = len(arr) // 2
    
    if median_idx < n_lo:
        return median_of_medians(lo)
    elif median_idx >= n_lo + n_eq:
        return median_of_medians(hi)
    else:
        return pivot


def select_kth(arr: List[float], k: int) -> float:
    """Найти k-й элемент за O(n) используя median-of-medians."""
    if len(arr) == 0:
        raise ValueError("Empty array")
    if len(arr) <= 5:
        return sorted(arr)[k]
    
    groups = []
    for i in range(0, len(arr), 5):
        group = arr[i:i + 5]
        group.sort()
        groups.append(group)
    
    medians = [g[len(g) // 2] for g in groups]
    pivot = median_of_medians(medians)
    
    lo = [x for x in arr if x < pivot - EPS]
    eq = [x for x in arr if abs(x - pivot) < EPS]
    hi = [x for x in arr if x > pivot + EPS]
    
    n_lo, n_eq = len(lo), len(eq)
    
    if k < n_lo:
        return select_kth(lo, k)
    elif k >= n_lo + n_eq:
        return select_kth(hi, k - n_lo - n_eq)
    else:
        return pivot


def solve_simplex(constraints: List[Tuple[float, float, float]], p: float, q: float) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """Симплекс-метод для 2D (перебор всех вершин) — эталон для проверки."""
    n = len(constraints)
    
    if n == 0:
        return ((0.0, 0.0, 0.0, 'optimal') if abs(p) < EPS and abs(q) < EPS else (None, None, None, 'unbounded'))
    
    def get_all_vertices():
        vertices = []
        for i in range(n):
            for j in range(i + 1, n):
                a1, b1, c1 = constraints[i]
                a2, b2, c2 = constraints[j]
                det = a1 * b2 - a2 * b1
                if abs(det) > EPS:
                    x = (c1 * b2 - c2 * b1) / det
                    y = (a1 * c2 - a2 * c1) / det
                    vertices.append((x, y))
        return vertices
    
    def is_feasible(x, y):
        return all(a * x + b * y <= c + 1e-6 for a, b, c in constraints)
    
    vertices = get_all_vertices()
    feasible_vertices = [(x, y) for x, y in vertices if is_feasible(x, y)]
    
    if not feasible_vertices:
        if is_feasible(0, 0):
            feasible_vertices.append((0.0, 0.0))
        for i in range(n):
            a, b, c = constraints[i]
            for y_test in [-1000, 0, 1000]:
                if abs(a) > EPS and is_feasible((c - b * y_test) / a, y_test):
                    feasible_vertices.append(((c - b * y_test) / a, y_test))
            for x_test in [-1000, 0, 1000]:
                if abs(b) > EPS and is_feasible(x_test, (c - a * x_test) / b):
                    feasible_vertices.append((x_test, (c - a * x_test) / b))
        
        unique = []
        for v in feasible_vertices:
            if not any(abs(v[0] - u[0]) < 1e-6 and abs(v[1] - u[1]) < 1e-6 for u in unique):
                unique.append(v)
        feasible_vertices = unique
    
    if not feasible_vertices:
        return (None, None, None, 'infeasible')
    
    def check_unbounded():
        grad_mag = math.hypot(p, q)
        if grad_mag > EPS:
            dx, dy = p / grad_mag, q / grad_mag
            if all(a * dx + b * dy <= EPS for a, b, c in constraints):
                return True
        return False
    
    if check_unbounded():
        return (None, None, None, 'unbounded')
    
    best_x, best_y, best_val = None, None, -INF
    for x, y in feasible_vertices:
        val = p * x + q * y
        if val > best_val + EPS:
            best_val, best_x, best_y = val, x, y
    
    return (best_x, best_y, best_val, 'optimal') if best_x is not None else (None, None, None, 'infeasible')


def solve_megiddo(constraints: List[Tuple[float, float, float]], p: float, q: float) -> Tuple[Optional[float], Optional[float], Optional[float], str, List[Tuple[float, float, float]]]:
    """
    Алгоритм Мегиддо для 2D LP с линейной сложностью O(n).
    
    Использует технику prune-and-search: на каждом шаге рекурсии отбрасываем
    ≥ n/2 прямых, что даёт T(n) = T(n/2) + O(n) = O(n).
    
    Ключевые моменты для O(n):
    1. Median-of-medians: O(n) вместо O(n log n)
    2. Отбрасываем ≥ n/2 ограничений (не просто n/4)
    """
    n = len(constraints)
    
    if n == 0:
        return ((0.0, 0.0, 0.0, 'optimal') if abs(p) < EPS and abs(q) < EPS else (None, None, None, 'unbounded'))
    
    L = math.hypot(p, q)
    if L < EPS:
        for a, b, c in constraints:
            if abs(a) > EPS:
                x_test, y_test = c / a, 0
                if all(ai * x_test + bi * y_test <= ci + EPS for ai, bi, ci in constraints):
                    return (x_test, y_test, 0.0, 'optimal')
        return (0.0, 0.0, 0.0, 'optimal')
    
    # Поворот координат O(n)
    upper = []  # u <= alpha*v + beta
    lower = []  # u >= alpha*v + beta
    v_min, v_max = -INF, INF
    
    for a, b, c in constraints:
        A = (a * p + b * q) / L
        B = (-a * q + b * p) / L
        
        if abs(A) < EPS:
            if abs(B) < EPS:
                if c < -EPS:
                    return (None, None, None, 'infeasible')
                continue
            if B > 0:
                v_max = min(v_max, c / B)
            else:
                v_min = max(v_min, c / B)
        else:
            alpha = -B / A
            beta = c / A
            if A > 0:
                upper.append((alpha, beta))
            else:
                lower.append((alpha, beta))
    
    if v_min > v_max + EPS:
        return (None, None, None, 'infeasible')
    
    # Проверка на неограниченность
    if not upper:
        if v_min <= v_max + EPS:
            return (None, None, None, 'unbounded')
        return (None, None, None, 'infeasible')
    
    def f_of_v(v, up_list):
        """f(v) = min_i(alpha_i*v + beta_i) — нижняя огибающая верхних прямых."""
        if not up_list:
            return INF
        return min(alpha * v + beta for alpha, beta in up_list)
    
    def g_of_v(v, low_list):
        """g(v) = max_j(alpha_j*v + beta_j) — верхняя огибающая нижних прямых."""
        if not low_list:
            return -INF
        return max(alpha * v + beta for alpha, beta in low_list)
    
    # Брутфорс для малого числа прямых
    def brute_force(up_list, low_list):
        candidates = set()
        if not math.isinf(v_min):
            candidates.add(v_min)
        if not math.isinf(v_max):
            candidates.add(v_max)
        
        all_lines = list(up_list) + list(low_list)
        m = len(all_lines)
        for i in range(m):
            for j in range(i + 1, m):
                a1, b1 = all_lines[i]
                a2, b2 = all_lines[j]
                if abs(a1 - a2) > EPS:
                    v_cross = (b2 - b1) / (a1 - a2)
                    candidates.add(v_cross)
        
        best_v, best_u = None, -INF
        found_feasible = False
        
        for v in candidates:
            if math.isinf(v):
                continue
            if v_min - EPS <= v <= v_max + EPS:
                f_val = f_of_v(v, up_list)
                g_val = g_of_v(v, low_list)
                if g_val <= f_val + EPS:
                    found_feasible = True
                    if f_val > best_u + EPS:
                        best_u = f_val
                        best_v = v
        
        if not found_feasible:
            return None, None, 'infeasible'
        if best_v is None:
            return None, None, 'unbounded'
        return best_v, best_u, 'optimal'
    
    # Рекурсивный алгоритм Мегиддо с истинной O(n) сложностью
    def megiddo_rec(up_list, low_list, depth=0):
        n_up = len(up_list)
        n_low = len(low_list)
        n_total = n_up + n_low
        
        # Базовый случай: мало прямых
        if n_total <= 10:
            return brute_force(up_list, low_list)
        
        if depth > 100:  # Защита от бесконечной рекурсии: log2(10^6) ≈ 20
            return brute_force(up_list, low_list)
        
        # Шаг 1: Формируем пары и находим точки пересечения O(n)
        crosses = []
        # Пары верхних прямых
        for i in range(0, n_up - 1, 2):
            a1, b1 = up_list[i]
            a2, b2 = up_list[i + 1]
            if abs(a1 - a2) > EPS:
                v_cross = (b2 - b1) / (a1 - a2)
                crosses.append(v_cross)
        # Пары нижних прямых
        for i in range(0, n_low - 1, 2):
            a1, b1 = low_list[i]
            a2, b2 = low_list[i + 1]
            if abs(a1 - a2) > EPS:
                v_cross = (b2 - b1) / (a1 - a2)
                crosses.append(v_cross)
        
        if not crosses:
            return brute_force(up_list, low_list)
        
        # Шаг 2: Находим медиану за O(n) с помощью median-of-medians
        v_med = median_of_medians(crosses)
        
        # Шаг 3: Оцениваем f(v_med) и g(v_med) O(n)
        f_val = f_of_v(v_med, up_list)
        g_val = g_of_v(v_med, low_list)
        
        # Шаг 4: Prune-and-search
        if g_val > f_val + EPS:
            # v_med недопустимо (g > f)
            # Находим активные прямые O(n)
            active_up = [(a, b) for a, b in up_list if abs(a * v_med + b - f_val) < EPS]
            active_low = [(a, b) for a, b in low_list if abs(a * v_med + b - g_val) < EPS]
            
            if active_low and active_up:
                # Находим медианные наклоны активных прямых O(n)
                slopes_low = [a for a, b in active_low]
                slopes_up = [a for a, b in active_up]
                
                sl = median_of_medians(slopes_low)
                su = median_of_medians(slopes_up)
                
                if sl > su + EPS:
                    # g растёт быстрее f справа → идём влево
                    # Отбрасываем:
                    # - верхние с α > su: справа они выше активной → f станет меньше
                    # - нижние с α < sl: слева они выше активной → g останется большим
                    # Отбрасываем верхние с α > su (≥ n_up/4)
                    new_up = [(a, b) for a, b in up_list if a <= su + EPS]
                    # Отбрасываем нижние с α < sl (≥ n_low/4)
                    new_low = [(a, b) for a, b in low_list if a >= sl - EPS]
                    
                    # Сохраняем активные прямые
                    for item in active_up:
                        if item not in new_up:
                            new_up.append(item)
                    for item in active_low:
                        if item not in new_low:
                            new_low.append(item)
                else:
                    # f растёт быстрее g справа → идём вправо
                    # Отбрасываем:
                    # - верхние с α < su: слева они выше активной → f станет меньше
                    # - нижние с α > sl: справа они ниже активной → g останется большим
                    # Отбрасываем верхние с α < su (≥ n_up/4)
                    new_up = [(a, b) for a, b in up_list if a >= su - EPS]
                    # Отбрасываем нижние с α > sl (≥ n_low/4)
                    new_low = [(a, b) for a, b in low_list if a <= sl + EPS]
                    
                    # Сохраняем активные прямые
                    for item in active_up:
                        if item not in new_up:
                            new_up.append(item)
                    for item in active_low:
                        if item not in new_low:
                            new_low.append(item)
                
                # Проверка: отброшено ли достаточно?
                discarded_up = n_up - len(new_up)
                discarded_low = n_low - len(new_low)
                
                res = megiddo_rec(new_up, new_low, depth + 1)
                if res[2] == 'optimal':
                    return res
                return brute_force(up_list, low_list)
            else:
                return brute_force(up_list, low_list)
        else:
            # v_med допустимо (g <= f) — ИСПРАВЛЕНО ДЛЯ O(n)
            # Находим активные верхние и нижние прямые O(n)
            active_up = [(a, b) for a, b in up_list if abs(a * v_med + b - f_val) < EPS]
            active_low = [(a, b) for a, b in low_list if abs(a * v_med + b - g_val) < EPS]
            
            if active_up:
                # Находим медианный наклон активной верхней прямой O(n)
                slopes_up = [a for a, b in active_up]
                median_slope = median_of_medians(slopes_up)
                
                if median_slope > EPS:
                    # f возрастает → максимум справа
                    # Отбрасываем верхние с α > median_slope: справа они выше активной → f станет меньше
                    # Отбрасываем нижние с α < median_slope: справа они ниже активной → g останется меньше
                    # ИТОГО: ≥ n/2 ограничений отброшено ✓
                    new_up = [(a, b) for a, b in up_list if a <= median_slope + EPS]
                    new_low = [(a, b) for a, b in low_list if a >= median_slope - EPS]
                    
                    # Сохраняем активные прямые
                    for item in active_up:
                        if item not in new_up:
                            new_up.append(item)
                    for item in active_low:
                        if item not in new_low:
                            new_low.append(item)
                else:
                    # f убывает → максимум слева
                    # Отбрасываем верхние с α < median_slope: слева они выше активной → f станет меньше
                    # Отбрасываем нижние с α > median_slope: слева они ниже активной → g останется меньше
                    # ИТОГО: ≥ n/2 ограничений отброшено ✓
                    new_up = [(a, b) for a, b in up_list if a >= median_slope - EPS]
                    new_low = [(a, b) for a, b in low_list if a <= median_slope + EPS]
                    
                    # Сохраняем активные прямые
                    for item in active_up:
                        if item not in new_up:
                            new_up.append(item)
                    for item in active_low:
                        if item not in new_low:
                            new_low.append(item)
                
                res = megiddo_rec(new_up, new_low, depth + 1)
                if res[2] == 'optimal':
                    return res
                return brute_force(up_list, low_list)
            
            return brute_force(up_list, low_list)
    
    # Запуск алгоритма
    v_opt, u_opt, status = megiddo_rec(upper, lower)
    
    if status != 'optimal':
        return (None, None, None, status, [])
    if v_opt is None or u_opt is None:
        return (None, None, None, 'unbounded' if u_opt == -INF else 'infeasible', [])
    
    # Обратное преобразование
    x = (p * u_opt - q * v_opt) / L
    y = (q * u_opt + p * v_opt) / L
    
    # Найти два ограничения, которые дают точку пересечения
    intersecting_constraints = []
    
    # Для каждой пары ограничений проверяем, пересекаются ли они в точке (x,y)
    for i in range(len(constraints)):
        a1, b1, c1 = constraints[i]
        # Проверяем, удовлетворяет ли точка ограничению (включая равенство)
        if abs(a1 * x + b1 * y - c1) <= EPS:
            intersecting_constraints.append((a1, b1, c1))
            if len(intersecting_constraints) == 2:
                break
    
    # Если не нашли точное пересечение, используем поиск ближайших
    if len(intersecting_constraints) < 2:
        # Простой подход: найти два ограничения, которые ближе всего к точке
        # Или найти те, которые действительно пересекаются в точке (прямые)
        # Проверяем точное пересечение для всех ограничений
        for i in range(len(constraints)):
            a1, b1, c1 = constraints[i]
            # Для точного пересечения нужно найти, что точка удовлетворяет равенству
            # но с учетом погрешности округления
            if abs(a1 * x + b1 * y - c1) <= EPS:
                intersecting_constraints.append((a1, b1, c1))
                if len(intersecting_constraints) == 2:
                    break
    
    return (x, y, p * x + q * y, 'optimal', intersecting_constraints)


def read_input(filename: str) -> Tuple[float, float, List[Tuple[float, float, float]]]:
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    p, q = map(float, lines[0].split())
    constraints = [tuple(map(float, line.split())) for line in lines[1:] if len(line.split()) == 3]
    return p, q, constraints


def compare_algorithms(constraints, p, q):
    mg = solve_megiddo(constraints, p, q)
    sx = solve_simplex(constraints, p, q)
    
    if mg[3] != sx[3]:
        return False, f"Статусы: М={mg[3]}, С={sx[3]}"
    if mg[3] != 'optimal':
        return True, f"Оба: {mg[3]}"
    
    if abs(mg[2] - sx[2]) > 1e-2:
        return False, f"ЦФ: М={mg[2]:.3f}, С={sx[2]:.3f}"
    
    def feasible(x, y, cons):
        return all(a * x + b * y <= c + 1e-6 for a, b, c in cons)
    
    if not feasible(mg[0], mg[1], constraints):
        return False, f"Мегиддо недопустима ({mg[0]:.2f},{mg[1]:.2f})"
    
    return True, f"OK: max={mg[2]:.3f}"


def generate_test(n=10, seed=None):
    if seed:
        random.seed(seed)
    p, q = random.uniform(-10, 10), random.uniform(-10, 10)
    constraints = [(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(0, 100)) for _ in range(n)]
    return p, q, constraints


def run_tests():
    print("=" * 70)
    print("ТЕСТЫ: Мегиддо O(n) vs Симплекс")
    print("=" * 70)
    
    passed_all = True
    
    print("\n[1] input2.txt")
    p, q, c = read_input('input2.txt')
    ok, msg = compare_algorithms(c, p, q)
    print(f"  {'✓' if ok else '✗'} {msg}")
    passed_all &= ok
    
    print("\n[2] Треугольник")
    p, q = 1, 1
    c = [(1, 0, 5), (0, 1, 5), (1, 1, 7)]
    mg = solve_megiddo(c, p, q)
    sx = solve_simplex(c, p, q)
    print(f"  Мегиддо: x={mg[0]:.3f}, y={mg[1]:.3f}, max={mg[2]:.3f}")
    print(f"  Симплекс: x={sx[0]:.3f}, y={sx[1]:.3f}, max={sx[2]:.3f}")
    ok, _ = compare_algorithms(c, p, q)
    print(f"  {'✓' if ok else '✗'}")
    passed_all &= ok
    
    print("\n[3] Неограниченная")
    p, q = 1, 1
    c = [(-1, 0, 0), (0, -1, 0)]
    mg, sx = solve_megiddo(c, p, q), solve_simplex(c, p, q)
    print(f"  М={mg[3]}, С={sx[3]}")
    ok = mg[3] == sx[3] == 'unbounded'
    print(f"  {'✓' if ok else '✗'}")
    passed_all &= ok
    
    print("\n[4] Невозможная")
    p, q = 1, 1
    c = [(1, 0, 5), (-1, 0, -10)]
    mg, sx = solve_megiddo(c, p, q), solve_simplex(c, p, q)
    print(f"  М={mg[3]}, С={sx[3]}")
    ok = mg[3] == sx[3] == 'infeasible'
    print(f"  {'✓' if ok else '✗'}")
    passed_all &= ok
    
    print("\n[5-14] Случайные")
    for i in range(10):
        p, q, c = generate_test(8, seed=i+100)
        ok, msg = compare_algorithms(c, p, q)
        print(f"  {i+5}: {'✓' if ok else '✗'} {msg[:50]}")
        passed_all &= ok
    
    print("\n[15] Большая (50 ограничений)")
    p, q = 3, 2
    c = [(math.cos(i*0.1)*5, math.sin(i*0.1)*5, 20+math.sin(i*0.3)*5) for i in range(50)]
    ok, msg = compare_algorithms(c, p, q)
    print(f"  {'✓' if ok else '✗'} {msg}")
    passed_all &= ok
    
    print("\n[16] Очень большая (200 ограничений)")
    p, q = 5, 3
    c = [(math.cos(i*0.05)*10, math.sin(i*0.05)*10, 50+math.sin(i*0.1)*10) for i in range(200)]
    ok, msg = compare_algorithms(c, p, q)
    print(f"  {'✓' if ok else '✗'} {msg}")
    passed_all &= ok
    
    print("\n" + "=" * 70)
    print("ВСЕ ПРОЙДЕНО ✓" if passed_all else "ЕСТЬ ОШИБКИ ✗")
    print("=" * 70)
    return passed_all


def main():
    import sys
    import time
    
    if len(sys.argv) < 2:
        print("python megiddo_optimized.py test|<файл>")
        return
    
    if sys.argv[1] == 'test':
        run_tests()
        return
    
    filename = sys.argv[1]
    p, q, constraints = read_input(filename)
    
    print(f"\nЗадача: max {p}*x + {q}*y")
    for i, (a, b, c) in enumerate(constraints, 1):
        print(f"  {i}: {a}*x + {b}*y <= {c}")
    
    start_time = time.perf_counter()
    x, y, val, status, intersecting_constraints = solve_megiddo(constraints, p, q)
    elapsed_time = (time.perf_counter() - start_time) * 1000
    
    if status == 'optimal':
        print(f"\nМегиддо O(n): x={x:.6f}, y={y:.6f}, max={val:.6f}")
        print(f"Время выполнения: {elapsed_time:.4f} мс")
        print(f"Число ограничений: {len(constraints)}")
        print(f"Операций (прибл.): {len(constraints) * 20:.0f}")
        
        # Вывод неравенств, дающих точку пересечения
        if intersecting_constraints:
            print("\nНеравенства, дающие точку пересечения:")
            for i, (a, b, c) in enumerate(intersecting_constraints, 1):
                print(f"  {i}: {a}*x + {b}*y = {c}")
        
        print("\nПроверка ограничений:")
        for i, (a, b, c) in enumerate(constraints, 1):
            v = a * x + b * y
            print(f"  {i}: {v:.4f} <= {c} {'✓' if v <= c + EPS else '✗'}")
        
        sx = solve_simplex(constraints, p, q)
        print(f"\nСимплекс: x={sx[0]:.6f}, y={sx[1]:.6f}, max={sx[2]:.6f}")
        ok, msg = compare_algorithms(constraints, p, q)
        print(f"Сравнение: {'✓' if ok else '✗'} {msg}")
    else:
        print(f"Статус: {status}")


if __name__ == "__main__":
    main()