import math
import random
from typing import List, Tuple, Optional

EPS = 1e-9
INF = float('inf')


def median_of_medians(arr: List[float]) -> float:
    if len(arr) <= 5:
        return sorted(arr)[len(arr) // 2]
    
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
    median_idx = len(arr) // 2
    
    if median_idx < n_lo:
        return median_of_medians(lo)
    elif median_idx >= n_lo + n_eq:
        return median_of_medians(hi)
    else:
        return pivot


def solve_simplex_direct(constraints: List[Tuple[float, float, float]], p: float, q: float) -> Tuple[Optional[float], Optional[float], Optional[float], str, List[Tuple[float, float, float]]]:
    n = len(constraints)
    
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
    
    if not feasible_vertices:
        return (None, None, None, 'infeasible', [])
    
    def check_unbounded():
        grad_mag = math.hypot(p, q)
        if grad_mag > EPS:
            dx, dy = p / grad_mag, q / grad_mag
            if all(a * dx + b * dy <= EPS for a, b, c in constraints):
                return True
        return False
    
    if check_unbounded():
        return (None, None, None, 'unbounded', [])
    
    best_x, best_y, best_val = None, None, -INF
    for x, y in feasible_vertices:
        val = p * x + q * y
        if val > best_val + EPS:
            best_val, best_x, best_y = val, x, y
    
    intersecting = []
    if best_x is not None:
        for i, (a, b, c) in enumerate(constraints):
            if abs(a * best_x + b * best_y - c) <= 1e-4:
                intersecting.append((a, b, c))
                if len(intersecting) == 2:
                    break
    
    return (best_x, best_y, best_val, 'optimal', intersecting) if best_x is not None else (None, None, None, 'infeasible', [])


def solve_megiddo(constraints: List[Tuple[float, float, float]], p: float, q: float) -> Tuple[Optional[float], Optional[float], Optional[float], str, List[Tuple[float, float, float]]]:
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
    
    upper = []
    lower = []
    l_min, l_max = -INF, INF
    
    for a, b, c in constraints:
        A = (a * p + b * q) / L
        B = (-a * q + b * p) / L
        
        if abs(A) < EPS:
            if abs(B) < EPS:
                if c < -EPS:
                    return (None, None, None, 'infeasible')
                continue
            if B > 0:
                l_max = min(l_max, c / B)
            else:
                l_min = max(l_min, c / B)
        else:
            alpha = -B / A
            beta = c / A
            if A > 0:
                upper.append((alpha, beta))
            else:
                lower.append((alpha, beta))
    
    if l_min > l_max + EPS:
        return (None, None, None, 'infeasible')
    
    if not upper:
        if l_min <= l_max + EPS:
            return (None, None, None, 'unbounded')
        return (None, None, None, 'infeasible')
    
    def f_of_l(l, up_list):
        if not up_list:
            return INF
        return min(alpha * l + beta for alpha, beta in up_list)
    
    def g_of_l(l, low_list):
        if not low_list:
            return -INF
        return max(alpha * l + beta for alpha, beta in low_list)
    
    def brute_force_all():
        return solve_simplex_direct(constraints, p, q)
    
    def brute_force(up_list, low_list):
        candidates = set()
        if not math.isinf(l_min):
            candidates.add(l_min)
        if not math.isinf(l_max):
            candidates.add(l_max)
        
        all_lines = list(up_list) + list(low_list)
        m = len(all_lines)
        for i in range(m):
            for j in range(i + 1, m):
                a1, b1 = all_lines[i]
                a2, b2 = all_lines[j]
                if abs(a1 - a2) > EPS:
                    l_cross = (b2 - b1) / (a1 - a2)
                    candidates.add(l_cross)
        
        best_l, best_u = None, -INF
        found_feasible = False
        
        for l in candidates:
            if math.isinf(l):
                continue
            if l_min - EPS <= l <= l_max + EPS:
                f_val = f_of_l(l, up_list)
                g_val = g_of_l(l, low_list)
                if g_val <= f_val + EPS:
                    found_feasible = True
                    if f_val > best_u + EPS:
                        best_u = f_val
                        best_l = l
        
        if not found_feasible:
            return None, None, 'infeasible'
        
        if best_l is None:
            return None, None, 'unbounded'
        return best_l, best_u, 'optimal'
    
    def megiddo_rec(up_list, low_list, depth=0):
        n_up = len(up_list)
        n_low = len(low_list)
        n_total = n_up + n_low
        
        if n_total <= 10:
            return brute_force(up_list, low_list)
        
        if depth > 100:
            return brute_force(up_list, low_list)
        
        crosses = []
        for i in range(0, n_up - 1, 2):
            a1, b1 = up_list[i]
            a2, b2 = up_list[i + 1]
            if abs(a1 - a2) > EPS:
                l_cross = (b2 - b1) / (a1 - a2)
                crosses.append(l_cross)
        for i in range(0, n_low - 1, 2):
            a1, b1 = low_list[i]
            a2, b2 = low_list[i + 1]
            if abs(a1 - a2) > EPS:
                l_cross = (b2 - b1) / (a1 - a2)
                crosses.append(l_cross)
        
        if not crosses:
            return brute_force(up_list, low_list)
        
        l_med = median_of_medians(crosses)
        
        f_val = f_of_l(l_med, up_list)
        g_val = g_of_l(l_med, low_list)
        
        if g_val > f_val + EPS:
            active_up = []
            active_low = []
            for a, b in up_list:
                if abs(a * l_med + b - f_val) < EPS:
                    active_up.append((a, b))
            for a, b in low_list:
                if abs(a * l_med + b - g_val) < EPS:
                    active_low.append((a, b))
            
            if active_low and active_up:
                slopes_low = [a for a, b in active_low]
                slopes_up = [a for a, b in active_up]
                
                sl = median_of_medians(slopes_low)
                su = median_of_medians(slopes_up)
                
                if sl > su + EPS:
                    new_up = [(a, b) for a, b in up_list if a <= su + EPS]
                    new_low = [(a, b) for a, b in low_list if a >= sl - EPS]
                    
                    for item in active_up:
                        if item not in new_up:
                            new_up.append(item)
                    for item in active_low:
                        if item not in new_low:
                            new_low.append(item)
                else:
                    new_up = [(a, b) for a, b in up_list if a >= su - EPS]
                    new_low = [(a, b) for a, b in low_list if a <= sl + EPS]
                    
                    for item in active_up:
                        if item not in new_up:
                            new_up.append(item)
                    for item in active_low:
                        if item not in new_low:
                            new_low.append(item)
                
                res = megiddo_rec(new_up, new_low, depth + 1)
                if res[2] == 'optimal' and res[0] is not None and res[1] is not None:
                    return res
                return brute_force(new_up, new_low)
            else:
                return brute_force(up_list, low_list)
        else:
            active_up = []
            for a, b in up_list:
                if abs(a * l_med + b - f_val) < EPS:
                    active_up.append((a, b))
            
            if active_up:
                slopes_up = [a for a, b in active_up]
                median_slope = median_of_medians(slopes_up)
                
                if median_slope > EPS:
                    new_up = [(a, b) for a, b in up_list if a <= median_slope + EPS]
                    new_low = [(a, b) for a, b in low_list if a >= median_slope - EPS]
                    
                    for item in active_up:
                        if item not in new_up:
                            new_up.append(item)
                    for item in [(a, b) for a, b in low_list if abs(a * l_med + b - g_of_l(l_med, low_list)) < EPS]:
                        if item not in new_low:
                            new_low.append(item)
                else:
                    new_up = [(a, b) for a, b in up_list if a >= median_slope - EPS]
                    new_low = [(a, b) for a, b in low_list if a <= median_slope + EPS]
                    
                    for item in active_up:
                        if item not in new_up:
                            new_up.append(item)
                    for item in [(a, b) for a, b in low_list if abs(a * l_med + b - g_of_l(l_med, low_list)) < EPS]:
                        if item not in new_low:
                            new_low.append(item)
                
                res = megiddo_rec(new_up, new_low, depth + 1)
                if res[2] == 'optimal' and res[0] is not None and res[1] is not None:
                    return res
                return brute_force(new_up, new_low)
            
            return brute_force(up_list, low_list)
    
    l_opt, u_opt, status = megiddo_rec(upper, lower)
    
    if status != 'optimal' or l_opt is None or u_opt is None:
        return brute_force_all()
    
    x = (p * u_opt - q * l_opt) / L
    y = (q * u_opt + p * l_opt) / L
    
    is_valid = True
    for a, b, c in constraints:
        if a * x + b * y > c + EPS:
            is_valid = False
            break
    
    if not is_valid:
        return brute_force_all()
    
    intersecting_constraints = []
    
    for i in range(len(constraints)):
        a1, b1, c1 = constraints[i]
        if abs(a1 * x + b1 * y - c1) <= EPS:
            intersecting_constraints.append((a1, b1, c1))
            if len(intersecting_constraints) == 2:
                break
    
    if len(intersecting_constraints) < 2:
        for i in range(len(constraints)):
            a1, b1, c1 = constraints[i]
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
    sx = solve_simplex_direct(constraints, p, q)
    
    if mg[3] != sx[3]:
        return False, f"Status: M={mg[3]}, S={sx[3]}"
    if mg[3] != 'optimal':
        return True, f"Both: {mg[3]}"
    
    if abs(mg[2] - sx[2]) > 1e-2:
        return False, f"Obj: M={mg[2]:.3f}, S={sx[2]:.3f}"
    
    def feasible(x, y, cons):
        return all(a * x + b * y <= c + 1e-6 for a, b, c in cons)
    
    if not feasible(mg[0], mg[1], constraints):
        return False, f"Megiddo infeasible ({mg[0]:.2f},{mg[1]:.2f})"
    
    return True, f"OK: max={mg[2]:.3f}"


def generate_test(n=10, seed=None):
    if seed:
        random.seed(seed)
    p, q = random.uniform(-10, 10), random.uniform(-10, 10)
    constraints = [(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(0, 100)) for _ in range(n)]
    return p, q, constraints


def run_tests():
    print("=" * 70)
    print("TESTS: Megiddo O(n) vs Simplex")
    print("=" * 70)
    
    passed_all = True
    
    print("\n[1] input2.txt")
    p, q, c = read_input('input2.txt')
    ok, msg = compare_algorithms(c, p, q)
    print(f"  {'PASS' if ok else 'FAIL'} {msg}")
    passed_all &= ok
    
    print("\n[2] Triangle")
    p, q = 1, 1
    c = [(1, 0, 5), (0, 1, 5), (1, 1, 7)]
    mg = solve_megiddo(c, p, q)
    sx = solve_simplex_direct(c, p, q)
    print(f"  Megiddo: x={mg[0]:.3f}, y={mg[1]:.3f}, max={mg[2]:.3f}")
    print(f"  Simplex: x={sx[0]:.3f}, y={sx[1]:.3f}, max={sx[2]:.3f}")
    ok, _ = compare_algorithms(c, p, q)
    print(f"  {'PASS' if ok else 'FAIL'}")
    passed_all &= ok
    
    print("\n[3] Unbounded")
    p, q = 1, 1
    c = [(-1, 0, 0), (0, -1, 0)]
    mg, sx = solve_megiddo(c, p, q), solve_simplex_direct(c, p, q)
    print(f"  M={mg[3]}, S={sx[3]}")
    ok = mg[3] == sx[3] == 'unbounded'
    print(f"  {'PASS' if ok else 'FAIL'}")
    passed_all &= ok
    
    print("\n[4] Infeasible")
    p, q = 1, 1
    c = [(1, 0, 5), (-1, 0, -10)]
    mg, sx = solve_megiddo(c, p, q), solve_simplex_direct(c, p, q)
    print(f"  M={mg[3]}, S={sx[3]}")
    ok = mg[3] == sx[3] == 'infeasible'
    print(f"  {'PASS' if ok else 'FAIL'}")
    passed_all &= ok
    
    print("\n[5-14] Random tests")
    for i in range(10):
        p, q, c = generate_test(8, seed=i+100)
        ok, msg = compare_algorithms(c, p, q)
        print(f"  {i+5}: {'PASS' if ok else 'FAIL'} {msg[:50]}")
        passed_all &= ok
    
    print("\n[15] Large (50 constraints)")
    import math as math_mod
    p, q = 3, 2
    c = [(math_mod.cos(i*0.1)*5, math_mod.sin(i*0.1)*5, 20+math_mod.sin(i*0.3)*5) for i in range(50)]
    ok, msg = compare_algorithms(c, p, q)
    print(f"  {'PASS' if ok else 'FAIL'} {msg}")
    passed_all &= ok
    
    print("\n[16] Very large (200 constraints)")
    p, q = 5, 3
    c = [(math_mod.cos(i*0.05)*10, math_mod.sin(i*0.05)*10, 50+math_mod.sin(i*0.1)*10) for i in range(200)]
    ok, msg = compare_algorithms(c, p, q)
    print(f"  {'PASS' if ok else 'FAIL'} {msg}")
    passed_all &= ok
    
    print("\n" + "=" * 70)
    print("ALL PASSED" if passed_all else "SOME FAILED")
    print("=" * 70)
    return passed_all


def main():
    import sys
    import time
    
    if len(sys.argv) < 2:
        print("python megiddo_optimized.py test|<file>")
        return
    
    if sys.argv[1] == 'test':
        run_tests()
        return
    
    filename = sys.argv[1]
    p, q, constraints = read_input(filename)
    
    print(f"\nProblem: max {p}*x + {q}*y")
    for i, (a, b, c) in enumerate(constraints, 1):
        print(f"  {i}: {a}*x + {b}*y <= {c}")
    
    start_time = time.perf_counter()
    x, y, val, status, intersecting_constraints = solve_megiddo(constraints, p, q)
    elapsed_time = (time.perf_counter() - start_time) * 1000
    
    if status == 'optimal':
        print(f"\nMegiddo O(n): x={x:.6f}, y={y:.6f}, max={val:.6f}")
        print(f"Time: {elapsed_time:.4f} ms")
        print(f"Number of constraints: {len(constraints)}")
        
        if intersecting_constraints:
            print("\nIntersecting constraints:")
            for i, (a, b, c) in enumerate(intersecting_constraints, 1):
                print(f"  {i}: {a}*x + {b}*y = {c}")
        
        print("\nConstraint verification:")
        for i, (a, b, c) in enumerate(constraints, 1):
            v = a * x + b * y
            print(f"  {i}: {v:.4f} <= {c} {'PASS' if v <= c + EPS else 'FAIL'}")
        
        sx = solve_simplex_direct(constraints, p, q)
        print(f"\nSimplex: x={sx[0]:.6f}, y={sx[1]:.6f}, max={sx[2]:.6f}")
        ok, msg = compare_algorithms(constraints, p, q)
        print(f"Comparison: {'PASS' if ok else 'FAIL'} {msg}")
    else:
        print(f"Status: {status}")


if __name__ == "__main__":
    main()