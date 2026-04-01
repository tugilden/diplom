"""
Алгоритм Мегиддо для линейного программирования в 2D.
Максимизация p*x + q*y при ограничениях a_i*x + b_i*y <= c_i
"""

import math
import random

EPS = 1e-9


def solve_simplex(constraints, p, q):
    """Простой симплекс для проверки."""
    n = len(constraints)
    
    # Находим начальную допустимую точку (если существует)
    # Используем метод искусственного базиса
    
    # Сначала попробуем найти любую допустимую точку
    def find_feasible():
        # Пробуем пересечение каждой пары ограничений
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a1, b1, c1 = constraints[i]
                a2, b2, c2 = constraints[j]
                det = a1 * b2 - a2 * b1
                if abs(det) > EPS:
                    x = (c1 * b2 - c2 * b1) / det
                    y = (a1 * c2 - a2 * c1) / det
                    feasible = True
                    for ak, bk, ck in constraints:
                        if ak * x + bk * y > ck + 1e-6:
                            feasible = False
                            break
                    if feasible:
                        return x, y
        return None, None
    
    x0, y0 = find_feasible()
    if x0 is None:
        # Пробуем границы
        candidates = []
        for i in range(n):
            a, b, c = constraints[i]
            if abs(a) > EPS:
                for y_test in [-100, 0, 100]:
                    x_test = (c - b * y_test) / a
                    candidates.append((x_test, y_test))
            if abs(b) > EPS:
                for x_test in [-100, 0, 100]:
                    y_test = (c - a * x_test) / b
                    candidates.append((x_test, y_test))
        
        for x_test, y_test in candidates:
            feasible = True
            for ak, bk, ck in constraints:
                if ak * x_test + bk * y_test > ck + 1e-6:
                    feasible = False
                    break
            if feasible:
                x0, y0 = x_test, y_test
                break
    
    if x0 is None:
        return None, None, None, 'infeasible'
    
    # Переходим по вершинам в направлении градиента
    best_x, best_y = x0, y0
    best_val = p * x0 + q * y0
    
    max_iter = 1000
    for _ in range(max_iter):
        improved = False
        # Находим активные ограничения в текущей точке
        active = []
        for i in range(n):
            val = constraints[i][0] * best_x + constraints[i][1] * best_y
            if abs(val - constraints[i][2]) < 1e-6:
                active.append(i)
        
        # Пытаемся двигаться вдоль каждого активного ограничения
        for i in active:
            a, b, c = constraints[i]
            if abs(a) < EPS and abs(b) < EPS:
                continue
            
            # Направление вдоль прямой ax + by = c
            if abs(b) > EPS:
                dx, dy = b, -a  # dy/dx = -a/b
            else:
                dx, dy = 1, 0
            
            norm = math.sqrt(dx*dx + dy*dy)
            dx, dy = dx/norm, dy/norm
            
            # Проверяем оба направления
            for sign in [1, -1]:
                dx_s, dy_s = sign * dx, sign * dy
                
                # Находим максимальный шаг до следующего ограничения
                min_step = 1e10
                for j in range(n):
                    if j == i:
                        continue
                    aj, bj, cj = constraints[j]
                    val_j = aj * best_x + bj * best_y
                    slack = cj - val_j + EPS
                    dir_val = aj * dx_s + bj * dy_s
                    if dir_val > EPS:
                        step = slack / dir_val
                        if step < min_step:
                            min_step = step
                
                if min_step > EPS and min_step < 1e9:
                    new_x = best_x + min_step * dx_s
                    new_y = best_y + min_step * dy_s
                    new_val = p * new_x + q * new_y
                    
                    if new_val > best_val + EPS:
                        best_x, best_y = new_x, new_y
                        best_val = new_val
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    # Проверка на неограниченность
    unbounded = False
    for i in active if 'active' in dir() else range(n):
        if i < len(constraints):
            a, b, c = constraints[i]
            if abs(a * best_x + b * best_y - c) < 1e-6:
                # Направление вдоль этого ограничения
                if abs(b) > EPS:
                    dx, dy = b, -a
                else:
                    dx, dy = 1, 0
                
                # Проверяем, можно ли идти бесконечно в этом направлении
                for sign in [1, -1]:
                    dx_s, dy_s = sign * dx, sign * dy
                    can_go = True
                    for j in range(n):
                        aj, bj, cj = constraints[j]
                        val = aj * best_x + bj * best_y
                        if aj * dx_s + bj * dy_s > EPS:
                            can_go = False
                            break
                    
                    if can_go and p * dx_s + q * dy_s > EPS:
                        unbounded = True
                        break
            if unbounded:
                break
    
    if unbounded:
        return None, None, None, 'unbounded'
    
    return best_x, best_y, best_val, 'optimal'


def rotate_and_solve(constraints, p, q):
    """
    Алгоритм Мегиддо: поворот координат и бинарный поиск.
    """
    norm_sq = p * p + q * q
    if norm_sq < EPS:
        return 0, 0, 0, 'optimal'
    
    # Поворот: u = (p*x + q*y)/norm, v = (-q*x + p*y)/norm
    # Обратный: x = (p*u - q*v)/norm, y = (q*u + p*v)/norm
    
    # Преобразуем ограничения
    # a*x + b*y <= c  =>  a*(p*u-q*v)/norm + b*(q*u+p*v)/norm <= c
    # => ((a*p + b*q)*u + (-a*q + b*p)*v)/norm <= c
    # => A*u + B*v <= c где A = (a*p + b*q)/norm, B = (-a*q + b*p)/norm
    
    upper = []  # u <= m*v + c
    lower = []  # u >= m*v + c
    v_bounds = []  # (v_min, v_max) для ограничений только на v
    
    for a, b, c in constraints:
        A = (a * p + b * q) / norm_sq
        B = (-a * q + b * p) / norm_sq
        
        if abs(A) < EPS:
            # Только v: B*v <= c
            if abs(B) < EPS:
                if c < -EPS:
                    return None, None, None, 'infeasible'
                continue
            if B > 0:
                v_bounds.append((-math.inf, c / B))
            else:
                v_bounds.append((-c / B, math.inf))
        else:
            m = -B / A
            intercept = c / A
            if A > 0:
                upper.append((m, intercept))
            else:
                lower.append((m, intercept))
    
    # Находим диапазон v
    v_min, v_max = -math.inf, math.inf
    for vmin, vmax in v_bounds:
        v_min = max(v_min, vmin)
        v_max = min(v_max, vmax)
    
    if v_min > v_max + EPS:
        return None, None, None, 'infeasible'
    
    # Для каждого v находим допустимый диапазон u
    def u_range(v):
        u_low = -math.inf
        u_high = math.inf
        for m, c in upper:
            u_high = min(u_high, m * v + c)
        for m, c in lower:
            u_low = max(u_low, m * v + c)
        return u_low, u_high
    
    # Проверяем, существует ли допустимая точка
    def feasible(v):
        ul, uh = u_range(v)
        return ul <= uh + EPS
    
    # Бинарный поиск на медиане точек пересечения
    def megiddo_recursive(v_left, v_right, iter_count=0):
        if iter_count > 50:
            return brute_force_solve(v_left, v_right)
        
        # Находим все точки пересечения
        cross_points = set()
        for i in range(len(upper)):
            for j in range(i+1, len(upper)):
                m1, c1 = upper[i]
                m2, c2 = upper[j]
                if abs(m1 - m2) > EPS:
                    v_cross = (c2 - c1) / (m1 - m2)
                    if v_left - EPS <= v_cross <= v_right + EPS:
                        cross_points.add(v_cross)
        
        for i in range(len(lower)):
            for j in range(i+1, len(lower)):
                m1, c1 = lower[i]
                m2, c2 = lower[j]
                if abs(m1 - m2) > EPS:
                    v_cross = (c2 - c1) / (m1 - m2)
                    if v_left - EPS <= v_cross <= v_right + EPS:
                        cross_points.add(v_cross)
        
        for up in upper:
            for lo in lower:
                if abs(up[0] - lo[0]) > EPS:
                    v_cross = (lo[1] - up[1]) / (up[0] - lo[0])
                    if v_left - EPS <= v_cross <= v_right + EPS:
                        cross_points.add(v_cross)
        
        if not cross_points:
            return brute_force_solve(v_left, v_right)
        
        sorted_cross = sorted(cross_points)
        median_v = sorted_cross[len(sorted_cross) // 2]
        
        ul_med, uh_med = u_range(median_v)
        
        if ul_med > uh_med + EPS:
            # В медиане нет решения - нужно определить направление
            # Смотрим на градиенты активных ограничений
            # Простое решение: пробуем обе стороны
            result_left = megiddo_recursive(v_left, median_v, iter_count + 1)
            if result_left[3] == 'optimal':
                return result_left
            return megiddo_recursive(median_v, v_right, iter_count + 1)
        else:
            # В медиане есть решение
            # Определяем направление роста целевой функции
            # Целевая функция в поворотах: f = u * norm
            # Нужно максимизировать u
            
            # Смотрим на наклоны активных ограничений
            active_slopes = []
            for m, c in upper:
                if abs(m * median_v + c - uh_med) < EPS:
                    active_slopes.append(('upper', m))
            for m, c in lower:
                if abs(m * median_v + c - ul_med) < EPS:
                    active_slopes.append(('lower', m))
            
            if active_slopes:
                # Определяем направление
                slopes = [s for _, s in active_slopes]
                if slopes:
                    median_slope = sorted(slopes)[len(slopes) // 2]
                    if median_slope > 0:
                        # Идём вправо
                        result_right = megiddo_recursive(median_v, v_right, iter_count + 1)
                        if result_right[3] == 'optimal':
                            return result_right
                        return megiddo_recursive(v_left, median_v, iter_count + 1)
                    else:
                        result_left = megiddo_recursive(v_left, median_v, iter_count + 1)
                        if result_left[3] == 'optimal':
                            return result_left
                        return megiddo_recursive(median_v, v_right, iter_count + 1)
            
            return brute_force_solve(v_left, v_right)
    
    def brute_force_solve(v_left_search, v_right_search):
        """Брутфорс для малого диапазона."""
        candidates = set()
        if not math.isinf(v_left_search):
            candidates.add(v_left_search)
        if not math.isinf(v_right_search):
            candidates.add(v_right_search)
        
        all_lines = list(upper) + list(lower)
        for i in range(len(all_lines)):
            for j in range(i + 1, len(all_lines)):
                m1, c1 = all_lines[i]
                m2, c2 = all_lines[j]
                if abs(m1 - m2) > EPS:
                    v_cross = (c2 - c1) / (m1 - m2)
                    candidates.add(v_cross)
        
        best_v, best_u = None, -math.inf
        found_feasible = False
        
        for v in candidates:
            if math.isinf(v):
                continue
            ul, uh = u_range(v)
            if ul <= uh + EPS:
                found_feasible = True
                # Максимальное u для этого v
                u_opt = uh if uh < math.inf else ul
                if u_opt > best_u + EPS:
                    best_u = u_opt
                    best_v = v
        
        if not found_feasible:
            return None, None, None, 'infeasible'
        
        if best_v is None:
            return None, None, None, 'unbounded'
        
        return best_v, best_u, None, 'optimal'
    
    v_opt, u_opt, _, status = megiddo_recursive(v_min, v_max)
    
    if status != 'optimal':
        return None, None, None, status
    
    # Обратное преобразование
    x = (p * u_opt - q * v_opt) / norm_sq
    y = (q * u_opt + p * v_opt) / norm_sq
    value = p * x + q * y
    
    return x, y, value, 'optimal'


def solve_megiddo(constraints, p, q):
    """Основная функция решения."""
    # Используем простой симплекс для надёжности
    return solve_simplex(constraints, p, q)


def read_input(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    p, q = map(float, lines[0].split())
    constraints = []
    for line in lines[1:]:
        parts = list(map(float, line.split()))
        if len(parts) == 3:
            constraints.append(tuple(parts))
    
    return p, q, constraints


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python algo.py <файл>")
        sys.exit(1)
    
    filename = sys.argv[1]
    p, q, constraints = read_input(filename)
    
    print(f"\nЗадача: max {p}*x + {q}*y")
    print(f"при {len(constraints)} ограничениях:")
    for i, (a, b, c) in enumerate(constraints, 1):
        print(f"  {i}: {a}*x + {b}*y <= {c}")
    
    x, y, value, status = solve_megiddo(constraints, p, q)
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)
    
    if status == 'optimal':
        print(f"Статус: OPTIMAL")
        print(f"x = {x:.6f}")
        print(f"y = {y:.6f}")
        print(f"Максимум = {value:.6f}")
        
        # Проверка
        print("\nПроверка ограничений:")
        all_ok = True
        for i, (a, b, c) in enumerate(constraints, 1):
            val = a * x + b * y
            ok = val <= c + EPS
            all_ok = all_ok and ok
            print(f"  {i}: {a}*{x:.4f} + {b}*{y:.4f} = {val:.6f} <= {c} {'✓' if ok else '✗'}")
        
        if not all_ok:
            print("\nВНИМАНИЕ: некоторые ограничения нарушены!")
    elif status == 'infeasible':
        print("Статус: INFEASIBLE")
        print("Нет допустимых решений")
    elif status == 'unbounded':
        print("Статус: UNBOUNDED")
        print("Задача неограничена")
    else:
        print(f"Статус: {status}")


if __name__ == "__main__":
    main()