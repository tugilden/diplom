import numpy as np
import plotly.graph_objects as go
import os
import sys

from megiddo_optimized import solve_megiddo, read_input
from transform_new import transform_inequalities
from grahem import Grahem, AngleHull


def format_coef(val, name):
    """Форматирует коэффициент для отображения."""
    if abs(val) >= 1 or abs(val) < 0.01:
        return f"{val:.1f}"
    return f"{val:.2f}"


def format_inequality_text(A, B, C):
    """Создает текст неравенства для hover."""
    A_str = format_coef(A, "A")
    B_str = format_coef(B, "B")
    C_str = format_coef(C, "C")
    
    return f"{A_str}x + {abs(B):.1f}y ≤ {C_str}"


def create_line_trace(A, B, C, x_vals, y_min_plot, y_max_plot, name, color, width, hover_text, visible=False):
    """Создает трейс для прямой: обычную или вертикальную."""
    # Вертикальная линия (B ≈ 0): x = C/A
    if abs(B) < 1e-9:
        x_const = C / A if abs(A) > 1e-9 else 0
        trace = go.Scatter(
            x=[x_const, x_const], y=[y_min_plot, y_max_plot], mode='lines',
            name=name,
            line=dict(color=color, width=width),
            hoverinfo='text', hovertext=hover_text,
            showlegend=False, visible=visible
        )
    else:
        y_vals = (C - A * x_vals) / B
        trace = go.Scatter(
            x=x_vals, y=y_vals, mode='lines',
            name=name,
            line=dict(color=color, width=width),
            hoverinfo='text', hovertext=hover_text,
            showlegend=False, visible=visible
        )
    return trace


def create_all_lines_mode1(constraints, active_indices, x_vals, y_min_plot, y_max_plot):
    """Создает все прямые для режима 1: активные синим, остальные голубым."""
    traces = []
    for i, (A, B, C) in enumerate(constraints):
        is_active = i in active_indices
        color = 'blue' if is_active else 'lightblue'
        width = 3 if is_active else 1.5
        
        hover_text = format_inequality_text(A, B, C)
        if is_active:
            hover_text += " (активное)"
        
        trace = create_line_trace(A, B, C, x_vals, y_min_plot, y_max_plot,
                                  f'Прямая {i+1}', color, width, hover_text, visible=False)
        traces.append(trace)
    return traces


def create_all_lines_mode3(constraints, x_vals, y_min_plot, y_max_plot):
    """Создает все прямые для режима 3: все голубым."""
    traces = []
    for i, (A, B, C) in enumerate(constraints):
        hover_text = format_inequality_text(A, B, C)
        
        trace = create_line_trace(A, B, C, x_vals, y_min_plot, y_max_plot,
                                  f'Прямая {i+1}', 'lightblue', 1.5, hover_text, visible=False)
        traces.append(trace)
    return traces


def create_active_lines_mode2(constraints, active_indices, x_vals, y_min_plot, y_max_plot):
    """Создает только активные прямые для режима 2."""
    traces = []
    for i, (A, B, C) in enumerate(constraints):
        if i not in active_indices:
            continue
        
        hover_text = format_inequality_text(A, B, C) + " (активное)"
        
        trace = create_line_trace(A, B, C, x_vals, y_min_plot, y_max_plot,
                                  f'Активное {i+1}', 'blue', 3, hover_text, visible=False)
        traces.append(trace)
    return traces


def create_solution_marker(x, y, val):
    """Создает маркер точки решения."""
    return go.Scatter(
        x=[x], y=[y], mode='markers',
        name='Решение',
        marker=dict(size=16, color='red', symbol='x-open', line=dict(width=3, color='darkred')),
        text=f'Решение: ({x:.2f}, {y:.2f})',
        hoverinfo='text', hovertext=f'Решение: ({x:.2f}, {y:.2f})',
        zorder=10, showlegend=False, visible=False
    )


def create_integer_points(x_coords, y_coords, color='red', symbol='circle', size=14):
    """Создает маркеры целочисленных решений."""
    if not x_coords:
        return None
    return go.Scatter(
        x=x_coords, y=y_coords, mode='markers+text',
        name='Целочисленные решения',
        marker=dict(size=size, color=color, symbol=symbol, line=dict(width=2, color=color)),
        text=[f'({xi}, {yi})' for xi, yi in zip(x_coords, y_coords)],
        textposition='top center',
        textfont=dict(size=11, color=color, family='Arial'),
        hoverinfo='text',
        hovertext=[f'({xi}, {yi})' for xi, yi in zip(x_coords, y_coords)],
        showlegend=True, zorder=10, visible=False
    )


def check_integer_solutions_feasibility(points_x, points_y, constraints, p, q):
    """
    Проверяет, попадают ли целые точки в исходную область допустимых решений.
    Вычисляет значения целевой функции для допустимых точек и возвращает лучшее решение.
    
    Args:
        points_x: список целых x-координат
        points_y: список целых y-координат
        constraints: список ограничений (A, B, C)
        p, q: коэффициенты целевой функции
        
    Returns:
        кортеж (valid_points, best_point, all_evaluations)
        - valid_points: список допустимых точек с значениями функции
        - best_point: лучшая точка с максимальным значением
        - all_evaluations: все оценки для отображения
    """
    if not points_x or not points_y:
        return [], None, []
    
    valid_points = []
    all_evaluations = []
    
    for i, (x_val, y_val) in enumerate(zip(points_x, points_y)):
        # Проверяем все ограничения
        is_feasible = True
        violated = []
        
        for idx, (A, B, C) in enumerate(constraints):
            lhs = A * x_val + B * y_val
            if lhs > C + 1e-6:  # Нарушение ограничения
                is_feasible = False
                violated.append((idx, lhs, C))
        
        # Вычисляем значение целевой функции
        obj_value = p * x_val + q * y_val
        
        all_evaluations.append({
            'point': (x_val, y_val),
            'feasible': is_feasible,
            'obj_value': obj_value,
            'violated': violated
        })
        
        if is_feasible:
            valid_points.append((x_val, y_val, obj_value))
    
    # Находим лучшее решение (максимальное значение)
    best_point = None
    if valid_points:
        best_point = max(valid_points, key=lambda pt: pt[2])
    
    return valid_points, best_point, all_evaluations


def create_best_point_marker(x, y, obj_value, size=18):
    """Создает выделенный маркер для лучшего целочисленного решения."""
    return go.Scatter(
        x=[x], y=[y], mode='markers+text',
        name='Лучшее целочисленное решение',
        marker=dict(
            size=size,
            color='yellow',
            symbol='circle',
            line=dict(width=3, color='orange')
        ),
        text=[f'({x}, {y})\nf={obj_value:.2f}'],
        textposition='top center',
        textfont=dict(size=13, color='orange', family='Arial', weight='bold'),
        hoverinfo='text',
        hovertext=f'Лучшее решение: ({x}, {y}), f={obj_value:.2f}',
        showlegend=True, zorder=15, visible=False
    )


def visualize_megiddo_solution(filename=None):
    """
    Визуализирует решение задачи Мегиддо с одним графиком и кнопками переключения.
    4 режима отображения:
    1) Все неравенства голубым + активные (дающие решение) синим
    2) Только активные неравенства + целочисленные решения (зеленые точки)
    3) Все неравенства голубым + целочисленные решения (красные точки)
    4) Все неравенства голубым + ЛУЧШЕЕ целочисленное решение (золотая звезда)
    """
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "input2.txt")
    
    # Чтение данных
    p, q, constraints = read_input(filename)
    n_constraints = len(constraints)
    
    print(f"Загружено {n_constraints} ограничений из {filename}")
    print(f"Целевая функция: max {p}*x + {q}*y")
    
    # Получение решения
    x, y, val, status, intersecting_constraints_list = solve_megiddo(constraints, p, q)
    
    # Индексы активных ограничений
    active_indices = []
    if intersecting_constraints_list:
        for idx, (A, B, C) in enumerate(constraints):
            for (A2, B2, C2) in intersecting_constraints_list:
                if abs(A - A2) < 1e-3 and abs(B - B2) < 1e-3 and abs(C - C2) < 1e-3:
                    active_indices.append(idx)
                    break
    
    if status != 'optimal':
        print(f"Решение не найдено: {status}")
        return
    
    print(f"Решение: x={x:.4f}, y={y:.4f}, max={val:.4f}")
    
    # Целочисленные решения
    integer_points_x = []
    integer_points_y = []
    best_point_coords = None  # (bx, by) - лучшая точка
    
    if active_indices and len(active_indices) >= 2:
        A1, B1, C1 = constraints[active_indices[0]]
        A2, B2, C2 = constraints[active_indices[1]]
        
        alpha, beta, gamma, beta2, identity_matrix = transform_inequalities(A1, B1, C1, A2, B2, C2)
        a, b, c = int(alpha), int(beta), int(gamma)
        
        P, H = AngleHull(a, b, c)
        P_transposed = P.T
        
        T_lst = [np.array([P_transposed[i, 0], P_transposed[i, 1]]) for i in range(P_transposed.shape[0])]
        T_fin_lst = [identity_matrix.dot(np.array([vec[0], vec[1] - int(C2 // abs(beta2))])) for vec in T_lst]
        
        integer_points_x = [int(vec[0]) for vec in T_fin_lst]
        integer_points_y = [int(vec[1]) for vec in T_fin_lst]
        
        print(f"\nЦелочисленные решения (AngleHull): {len(T_fin_lst)} точек")
        for i, vec in enumerate(T_fin_lst):
            print(f"  ({int(vec[0])}, {int(vec[1])})")
        
        # Находим лучшую точку (максимальное значение среди допустимых)
        best_val = -float('inf')
        for i, (px, py) in enumerate(zip(integer_points_x, integer_points_y)):
            # Проверяем допустимость
            is_feasible = True
            for j, (a_c, b_c, c_c) in enumerate(constraints):
                if a_c * px + b_c * py > c_c + 1e-6:
                    is_feasible = False
                    break
            if is_feasible:
                obj_v = p * px + q * py
                if obj_v > best_val:
                    best_val = obj_v
                    best_point_coords = (px, py)
        
        if best_point_coords:
            print(f"\nЛУЧШАЯ ЦЕЛОЧИСЛЕННАЯ ТОЧКА: {best_point_coords} (f={best_val:.2f})")
    
    has_integer_points = len(integer_points_x) > 0
    
    # ========================================================================
    # ПРОВЕРКА ЦЕЛОЧИСЛЕННЫХ ТОЧЕК НА ПРИНАДЛЕЖНОСТЬ ИСХОДНОЙ ОБЛАСТИ
    # ========================================================================
    print(f"\n{'='*70}")
    print("ПРОВЕРКА ЦЕЛОЧИСЛЕННЫХ ТОЧЕК НА ПРИНАДЛЕЖНОСТЬ ОБЛАСТИ")
    print(f"{'='*70}")
    
    # Проверяем все целые точки на попадание в исходную область
    valid_points, best_point, all_evaluations = check_integer_solutions_feasibility(
        integer_points_x, integer_points_y, constraints, p, q
    )
    
    # Выводим информацию о каждой точке
    print(f"\nВсего точек найдено алгоритмом: {len(integer_points_x)}")
    print(f"Допустимых точек (в области): {len(valid_points)}")
    
    print(f"\n--- Детальная проверка каждой точки ---")
    for eval_data in all_evaluations:
        px, py = eval_data['point']
        obj_val = eval_data['obj_value']
        is_feas = eval_data['feasible']
        violations = eval_data['violated']
        
        status_icon = "+" if is_feas else "x"
        status_text = "ДОПУСТИМА" if is_feas else "НЕДОПУСТИМА"
        
        print(f"  Точка ({px}, {py}): f(x,y) = {obj_val:.2f} [{status_icon} {status_text}]")
        
        if violations:
            print(f"    Нарушенные ограничения:")
            for idx, lhs, rhs in violations:
                print(f"      #{idx+1}: {lhs:.2f} > {rhs:.2f}")
    
    # Информация о лучшем решении
    if best_point:
        bx, by, bval = best_point
        print(f"\n{'='*60}")
        print("* ЛУЧШЕЕ ЦЕЛОЧИСЛЕННОЕ РЕШЕНИЕ *")
        print(f"{'='*60}")
        print(f"  x* = {bx}")
        print(f"  y* = {by}")
        print(f"  f(x*, y*) = {bval:.2f}")
        print(f"\n  Сравнение с вещественным оптимумом:")
        print(f"    Вещественный: f = {val:.2f}")
        print(f"    Целочисленное: f = {bval:.2f}")
        if val > 0:
            gap = ((val - bval) / val) * 100
        else:
            gap = 0
        print(f"    Потеря (gap): {gap:.2f}%")
        
        # Показываем все допустимые точки с их значениями
        print(f"\n--- Все допустимые точки (сортировка по значению f) ---")
        sorted_valid = sorted(valid_points, key=lambda pt: pt[2], reverse=True)
        for i, (vx, vy, vval) in enumerate(sorted_valid):
            marker = " *" if (vx, vy, vval) == best_point else ""
            print(f"  {i+1}. ({vx}, {vy}): f = {vval:.2f}{marker}")
    else:
        print(f"\n[!] Ни одна точка не попала в область допустимых решений!")
    
    # Флаг для использования лучшего решения в визуализации
    has_best_point = best_point is not None
    if has_best_point:
        best_x, best_y, best_val = best_point
    else:
        best_x, best_y, best_val = 0, 0, 0
    
    # Границы графика
    all_intersections = []
    for i in range(n_constraints):
        for j in range(i + 1, n_constraints):
            a1, b1, c1 = constraints[i]
            a2, b2, c2 = constraints[j]
            det = a1 * b2 - a2 * b1
            if abs(det) > 1e-9:
                ix = (c1 * b2 - c2 * b1) / det
                iy = (a1 * c2 - a2 * c1) / det
                all_intersections.append((ix, iy))
    
    if all_intersections:
        x_min = max(min(pt[0] for pt in all_intersections) - 5, x - 30)
        x_max = min(max(pt[0] for pt in all_intersections) + 5, x + 30)
        y_min = max(min(pt[1] for pt in all_intersections) - 5, y - 30)
        y_max = min(max(pt[1] for pt in all_intersections) + 5, y + 30)
    else:
        x_min, x_max = x - 30, x + 30
        y_min, y_max = y - 30, y + 30
    
    x_vals = np.linspace(x_min - 10, x_max + 10, 300)
    
    # ========================================================================
    # СОЗДАЕМ ТРЕЙСЫ ДЛЯ ВСЕХ РЕЖИМОВ
    # ========================================================================
    
    # Режим 1: все прямые (активные синим) + решение
    m1_lines = create_all_lines_mode1(constraints, active_indices, x_vals, y_min, y_max)
    m1_solution = create_solution_marker(x, y, val)
    
    # Режим 2: только активные прямые + решение + целочисленные (зеленые)
    m2_lines = create_active_lines_mode2(constraints, active_indices, x_vals, y_min, y_max)
    m2_solution = create_solution_marker(x, y, val)
    m2_integer = create_integer_points(integer_points_x, integer_points_y, color='green', size=14)
    
    # Режим 3: все прямые (голубым) + целочисленные (красные)
    m3_lines = create_all_lines_mode3(constraints, x_vals, y_min, y_max)
    m3_integer = create_integer_points(integer_points_x, integer_points_y, color='red', size=16)
    
    # Режим 4: все прямые (голубым) + ЛУЧШЕЕ целочисленное решение (золотая звезда)
    m4_lines = create_all_lines_mode3(constraints, x_vals, y_min, y_max)
    m4_best = None
    if best_point_coords:
        bx, by = best_point_coords
        m4_best = create_best_point_marker(bx, by, best_val)
    
    # ========================================================================
    # СОБИРАЕМ ВСЕ ТРЕЙСЫ В ОДИН СПИСОК
    # ========================================================================
    # Порядок: m1_lines + m1_solution + m2_lines + m2_solution + m2_integer + m3_lines + m3_integer + m4_lines + m4_best
    
    all_traces = []
    
    # Индексы для режима 1
    m1_lines_actual = len(m1_lines)
    m1_solution_idx = m1_lines_actual
    
    # Добавляем трейсы режима 1
    all_traces.extend(m1_lines)
    all_traces.append(m1_solution)
    
    # Индексы для режима 2
    m2_lines_start = len(all_traces)
    all_traces.extend(m2_lines)
    m2_solution_idx = len(all_traces)
    all_traces.append(m2_solution)
    m2_integer_idx = None
    if m2_integer:
        m2_integer_idx = len(all_traces)
        all_traces.append(m2_integer)
    
    # Индексы для режима 3
    m3_lines_start = len(all_traces)
    all_traces.extend(m3_lines)
    m3_integer_idx = None
    if m3_integer:
        m3_integer_idx = len(all_traces)
        all_traces.append(m3_integer)
    
    # Индексы для режима 4
    m4_lines_start = len(all_traces)
    m4_lines_actual = len(m4_lines)
    all_traces.extend(m4_lines)
    m4_best_idx = None
    if m4_best:
        m4_best_idx = len(all_traces)
        all_traces.append(m4_best)
    
    n_total = len(all_traces)
    
    # ========================================================================
    # ОПРЕДЕЛЯЕМ ВИДИМОСТЬ ДЛЯ КАЖДОГО РЕЖИМА
    # ========================================================================
    
    m1_lines_actual = len(m1_lines)
    m2_lines_actual = len(m2_lines)
    m3_lines_actual = len(m3_lines)
    
    # Режим 1: m1_lines + m1_solution
    vis_mode1 = [False] * n_total
    for i in range(m1_lines_actual):
        vis_mode1[i] = True
    vis_mode1[m1_solution_idx] = True
    
    # Режим 2: m2_lines + m2_solution + m2_integer
    vis_mode2 = [False] * n_total
    for i in range(m2_lines_actual):
        vis_mode2[m2_lines_start + i] = True
    vis_mode2[m2_solution_idx] = True
    if m2_integer_idx is not None:
        vis_mode2[m2_integer_idx] = True
    
    # Режим 3: m3_lines + m3_integer
    vis_mode3 = [False] * n_total
    for i in range(m3_lines_actual):
        vis_mode3[m3_lines_start + i] = True
    if m3_integer_idx is not None:
        vis_mode3[m3_integer_idx] = True
    
    # Режим 4: m4_lines + m4_best
    vis_mode4 = [False] * n_total
    for i in range(m4_lines_actual):
        vis_mode4[m4_lines_start + i] = True
    if m4_best_idx is not None:
        vis_mode4[m4_best_idx] = True
    
    # ========================================================================
    # СОЗДАЕМ ГРАФИК
    # ========================================================================
    fig = go.Figure(data=all_traces)
    
    fig.update_layout(
        title_text=f"Линейное программирование: max {p}*x + {q}*y<br>"
                   f"Решение: x={x:.2f}, y={y:.2f}, max={val:.2f} | "
                   f"Целочисленных решений: {len(integer_points_x)}",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(range=[x_min - 5, x_max + 5], title="x"),
        yaxis=dict(range=[y_min - 5, y_max + 5], title="y"),
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        label="Режим 1: Все + активные (синий)",
                        method="update",
                        args=[
                            {"visible": vis_mode1},
                            {"title": f"Режим 1: Все неравенства (голубой) + активные (синий)<br>"
                                      f"max {p}*x + {q}*y | Решение: ({x:.2f}, {y:.2f})"}
                        ]
                    ),
                    dict(
                        label="Режим 2: Активные + целочисленные",
                        method="update",
                        args=[
                            {"visible": vis_mode2},
                            {"title": f"Режим 2: Активные неравенства + целочисленные решения<br>"
                                      f"Активных: {len(active_indices)} | Целочисленных: {len(integer_points_x)}"}
                        ]
                    ),
                    dict(
                        label="Режим 3: Все + целочисленные (красные)",
                        method="update",
                        args=[
                            {"visible": vis_mode3},
                            {"title": f"Режим 3: Все неравенства + целочисленные решения (красные)<br>"
                                      f"Всего: {n_constraints} | Целочисленных: {len(integer_points_x)}"}
                        ]
                    ),
                    dict(
                        label="Режим 4: ЛУЧШЕЕ решение (*)",
                        method="update",
                        args=[
                            {"visible": vis_mode4},
                            {"title": f"Режим 4: Все неравенства + ЛУЧШЕЕ целочисленное решение (*) золотая звезда<br>"
                                      f"Всего: {n_constraints} | Лучшее: {best_point_coords if best_point_coords else 'N/A'}"}
                        ]
                    ),
                ]),
                direction="down",
                showactive=True,
                x=0.78,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        template="plotly_white",
        height=700,
        width=1000,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    # Сохраняем график в HTML файл
    output_path = os.path.join(os.path.dirname(__file__), "megiddo_graph.html")
    fig.write_html(output_path)
    print(f"\nГрафик сохранен: {output_path}")
    
    fig.show()


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_megiddo_solution(filename)