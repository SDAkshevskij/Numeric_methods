import cmath
import math
import sys
from pathlib import Path

import numpy as np


MAX_ITERATIONS = 10000


def read_input(filename):
    lines = [line for line in Path(filename).read_text().splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError("Во входном файле недостаточно данных.")

    n, eps = int(lines[0]), float(lines[1])
    if n <= 0 or len(lines) < n + 4:
        raise ValueError("Неверные входные данные.")

    A = np.array([list(map(float, line.split())) for line in lines[2:n + 2]])
    m = int(lines[n + 2])
    example = np.array([list(map(float, line.split())) for line in lines[n + 3:]])

    if m <= 0 or len(lines) != n + m + 3:
        raise ValueError("Неверное количество строк.")
    if A.shape != (n, n) or example.shape != (m, m):
        raise ValueError("Матрицы должны быть квадратными.")
    if not np.isfinite(eps) or eps <= 0 or not np.all(np.isfinite(A)) or not np.all(np.isfinite(example)):
        raise ValueError("Неверные входные данные.")
    return A, eps, example


def householder_qr(A):
    n = len(A)
    Q, R = np.eye(n), A.copy()

    for k in range(n - 1):
        x = R[k:, k].copy()
        length = math.hypot(*x)
        if length == 0:
            continue

        x[0] += math.copysign(length, x[0])
        v = x / math.hypot(*x)
        R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v, v)

    R[abs(R) < 1e-12 * np.max(np.abs(A))] = 0
    return Q, R


def hessenberg(A):
    H = A.copy()

    for k in range(len(A) - 2):
        x = H[k + 1:, k].copy()
        length = math.hypot(*x)
        if length == 0:
            continue

        x[0] += math.copysign(length, x[0])
        v = x / math.hypot(*x)
        H[k + 1:, k:] -= 2 * np.outer(v, v @ H[k + 1:, k:])
        H[:, k + 1:] -= 2 * np.outer(H[:, k + 1:] @ v, v)
        H[k + 2:, k] = 0

    return H


def eigenvalues_2x2(A, i):
    a, b = A[i, i], A[i, i + 1]
    c, d = A[i + 1, i], A[i + 1, i + 1]
    scale = max(abs(a), abs(b), abs(c), abs(d))
    if scale == 0:
        return 0j, 0j

    a, b, c, d = a / scale, b / scale, c / scale, d / scale
    trace, determinant = a + d, a * d - b * c
    root = cmath.sqrt(trace ** 2 - 4 * determinant)
    return scale * (trace + root) / 2, scale * (trace - root) / 2


def qr_eigenvalues(A, eps):
    current = hessenberg(A)
    values = []
    active = len(A)
    iterations = 0

    while active:
        block = current[:active, :active]

        if active == 1 or math.hypot(*block[-1, :-1]) <= eps:
            values.insert(0, complex(block[-1, -1]))
            active -= 1
            continue

        pair = eigenvalues_2x2(block, active - 2)
        lower_left = block[-2:, :-2]
        pair_is_separated = active == 2 or math.hypot(*lower_left.ravel()) <= eps

        complex_pair = abs(pair[0].imag) > 1e-12 * abs(pair[0])
        if pair_is_separated and complex_pair:
            values[0:0] = pair
            active -= 2
            continue

        if iterations == MAX_ITERATIONS:
            raise RuntimeError("QR-алгоритм не сошелся.")

        if not complex_pair:
            shift = min(pair, key=lambda value: abs(value.real - block[-1, -1])).real
            Q, R = householder_qr(block - shift * np.eye(active))
            current[:active, :active] = R @ Q + shift * np.eye(active)
        else:
            normalized = block / np.max(np.abs(block))
            a, b = normalized[-2, -2], normalized[-2, -1]
            c, d = normalized[-1, -2], normalized[-1, -1]
            polynomial = normalized @ normalized - (a + d) * normalized + (a * d - b * c) * np.eye(active)
            Q, _ = householder_qr(polynomial)
            current[:active, :active] = Q.T @ block @ Q

        iterations += 1

    return values, current, iterations


def show_matrix(name, value):
    print(f"{name}:\n{np.array2string(value, precision=6, suppress_small=True)}\n")


def show_values(name, values):
    print(name + ":")
    for i, value in enumerate(values, 1):
        if abs(value.imag) <= 1e-12 * abs(value):
            print(f"lambda_{i} = {value.real:.6f}")
        else:
            print(f"lambda_{i} = {value.real:.6f} {value.imag:+.6f}i")
    print()


def eigenvalue_error(values, reference):
    first = max(min(abs(value - other) for other in reference) for value in values)
    second = max(min(abs(value - other) for other in values) for value in reference)
    return max(first, second)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("input.txt")
    A, eps, example = read_input(filename)
    Q, R = householder_qr(A)
    values, final, iterations = qr_eigenvalues(A, eps)

    show_matrix("Исходная матрица A", A)
    show_matrix("Матрица Q", Q)
    show_matrix("Матрица R", R)
    show_matrix("Q * R", Q @ R)
    print("Ошибка A = Q * R:", np.max(np.abs(A - Q @ R)))
    print("Количество QR-итераций:", iterations, "\n")
    show_values("Собственные значения", values)

    numpy_Q, numpy_R = np.linalg.qr(A)
    numpy_values = np.linalg.eigvals(A)
    print("Сравнение с NumPy:")
    print("Ошибка A = Q * R у NumPy:", np.max(np.abs(A - numpy_Q @ numpy_R)))
    print("Ошибка собственных значений:", eigenvalue_error(values, numpy_values), "\n")

    example_values, _, example_iterations = qr_eigenvalues(example, eps)
    show_matrix("Пример с комплексно-сопряженными значениями", example)
    show_values("Значения примера", example_values)
    numpy_example = np.linalg.eigvals(example)
    print("Проверка NumPy:", numpy_example)
    print("Ошибка значений примера:", eigenvalue_error(example_values, numpy_example))
    print("Количество QR-итераций примера:", example_iterations)


if __name__ == "__main__":
    main()
