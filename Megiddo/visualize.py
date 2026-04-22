#!/usr/bin/env python3
"""
Визуализация ограничений и решения задачи линейного программирования.
Один график с кнопками для переключения между 3 режимами отображения:
1) Все неравенства голубым + активные (дающие решение) синим
2) Только активные неравенства + целочисленные решения (зеленые точки)
3) Все неравенства голубым + целочисленные решения (красные точки)
"""

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


def create_all_lines_mode1(constraints, active_indices, x_vals):
    """Создает все прямые для режима 1: активные синим, остальные голубым."""
    traces = []
    for i, (A, B, C) in enumerate(constraints):
        if abs(B) > 1e-9:
            y_vals = (C - A * x_vals) / B
        else:
            continue
        
        is_active = i in active_indices
        color = 'blue' if is_active else 'lightblue'
        width = 3 if is_active else 1.5
        
        hover_text = format_inequality_text(A, B, C)
        if is_active:
            hover_text += " (активное)"
        
        trace = go.Scatter(
            x=x_vals, y=y_vals, mode='lines',
            name=f'Прямая {i+1}',
            line=dict(color=color, width=width),
            hoverinfo='text', hovertext=hover_text,
            showlegend=False, visible=False
        )
        traces.append(trace)
    return traces


def create_all_lines_mode3(constraints, x_vals):
    """Создает все прямые для режима 3: все голубым."""
    traces = []
    for i, (A, B, C) in enumerate(constraints):
        if abs(B) > 1e-9:
            y_vals = (C - A * x_vals) / B
        else:
            continue
        
        hover_text = format_inequality_text(A, B, C)
        
        trace = go.Scatter(
            x=x_vals, y=y_vals, mode='lines',
            name=f'Прямая {i+1}',
            line=dict(color='lightblue', width=1.5),
            hoverinfo='text', hovertext=hover_text,
            showlegend=False, visible=False
        )
        traces.append(trace)
    return traces


def create_active_lines_mode2(constraints, active_indices, x_vals):
    """Создает только активные прямые для режима 2."""
    traces = []
    for i, (A, B, C) in enumerate(constraints):
        if abs(B) > 1e-9:
            y_vals = (C - A * x_vals) / B
        else:
            continue
        
        if i not in active_indices:
            continue
        
        hover_text = format_inequality_text(A, B, C) + " (активное)"
        
        trace = go.Scatter(
            x=x_vals, y=y_vals, mode='lines',
            name=f'Активное {i+1}',
            line=dict(color='blue', width=3),
            hoverinfo='text', hovertext=hover_text,
            showlegend=False, visible=False
        )
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


def visualize_megiddo_solution(filename=None):
    """
    Визуализирует решение задачи Мегиддо с одним графиком и кнопками переключения.
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
    
    has_integer_points = len(integer_points_x) > 0
    
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
    # СОЗДАЕМ ТРЕЙСЫ ДЛЯ ВСЕХ ТРЕХ РЕЖИМОВ
    # ========================================================================
    
    # Режим 1: все прямые (активные синим) + решение
    m1_lines = create_all_lines_mode1(constraints, active_indices, x_vals)
    m1_solution = create_solution_marker(x, y, val)
    
    # Режим 2: только активные прямые + решение + целочисленные (зеленые)
    m2_lines = create_active_lines_mode2(constraints, active_indices, x_vals)
    m2_solution = create_solution_marker(x, y, val)
    m2_integer = create_integer_points(integer_points_x, integer_points_y, color='green', size=14)
    
    # Режим 3: все прямые (голубым) + целочисленные (красные)
    m3_lines = create_all_lines_mode3(constraints, x_vals)
    m3_integer = create_integer_points(integer_points_x, integer_points_y, color='red', size=16)
    
    # ========================================================================
    # СОБИРАЕМ ВСЕ ТРЕЙСЫ В ОДИН СПИСОК
    # ========================================================================
    # Порядок: m1_lines + m1_solution + m2_lines + m2_solution + m2_integer + m3_lines + m3_integer
    
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
                ]),
                direction="down",
                showactive=True,
                x=0.02,
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