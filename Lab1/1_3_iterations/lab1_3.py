import sys
from pathlib import Path

import numpy as np


MAX_ITERATIONS = 100000


def read_input(filename):
    lines = [line for line in Path(filename).read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Во входном файле недостаточно данных.")

    n, eps = int(lines[0]), float(lines[1])
    if n <= 0 or len(lines) != n + 4 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Неверные входные данные.")

    A = np.array([list(map(float, lines[i].split())) for i in range(2, n + 2)])
    b = np.fromstring(lines[n + 2], sep=" ")
    exact = np.fromstring(lines[n + 3], sep=" ")
    data = np.concatenate((A.ravel(), b, exact))
    if A.shape != (n, n) or len(b) != n or len(exact) != n or not np.all(np.isfinite(data)):
        raise ValueError("Неверные размеры матрицы или векторов.")
    return A, b, eps, exact


def iteration_form(A, b):
    diagonal = np.diag(A)
    row_scale = np.max(np.abs(A), axis=1)
    if np.any(row_scale == 0) or np.any(abs(diagonal) <= 1e-12 * row_scale):
        raise ValueError("На главной диагонали есть ноль.")
    alpha = -A / diagonal[:, None]
    np.fill_diagonal(alpha, 0)
    return alpha, b / diagonal


def norm(x):
    return np.max(np.abs(x))


def simple_iterations(alpha, beta, eps):
    q = max(sum(abs(value) for value in row) for row in alpha)
    previous = beta.copy()

    for k in range(1, MAX_ITERATIONS + 1):
        current = alpha @ previous + beta
        if not np.all(np.isfinite(current)):
            raise RuntimeError("Метод простых итераций расходится.")
        error = norm(current - previous)
        error = q / (1 - q) * error if q < 1 else error
        if error <= eps:
            return current, k, error
        previous = current
    raise RuntimeError("Метод простых итераций не сошелся.")


def seidel(alpha, beta, eps):
    previous = beta.copy()

    for k in range(1, MAX_ITERATIONS + 1):
        current = previous.copy()
        for i in range(len(beta)):
            current[i] = beta[i] + alpha[i, :i] @ current[:i] + alpha[i, i + 1:] @ previous[i + 1:]
        if not np.all(np.isfinite(current)):
            raise RuntimeError("Метод Зейделя расходится.")
        error = norm(current - previous)
        if error <= eps:
            return current, k, error
        previous = current
    raise RuntimeError("Метод Зейделя не сошелся.")


def show_method(name, result, exact):
    x, iterations, estimate = result
    print(f"\n{name}")
    print("Решение:", np.array2string(x, precision=6, suppress_small=True))
    print("Количество итераций:", iterations)
    print("Оценка погрешности:", estimate)
    print("Фактическая ошибка:", norm(x - exact))


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("input.txt")
    A, b, eps, exact = read_input(filename)
    alpha, beta = iteration_form(A, b)
    simple = simple_iterations(alpha, beta, eps)
    seidel_result = seidel(alpha, beta, eps)

    print("Точность:", eps)
    print("Эталонное решение:", exact)
    show_method("Метод простых итераций", simple, exact)
    show_method("Метод Зейделя", seidel_result, exact)

    first, second = simple[1], seidel_result[1]
    if first == second:
        print("\nМетоды потребовали одинаковое число итераций.")
    else:
        faster = "простых итераций" if first < second else "Зейделя"
        print(f"\nМетод {faster} сошелся быстрее: {min(first, second)} против {max(first, second)}.")

    numpy_x = np.linalg.solve(A, b)
    print("\nСравнение с NumPy:")
    print("Ошибка простых итераций:", norm(simple[0] - numpy_x))
    print("Ошибка метода Зейделя:", norm(seidel_result[0] - numpy_x))


if __name__ == "__main__":
    main()
