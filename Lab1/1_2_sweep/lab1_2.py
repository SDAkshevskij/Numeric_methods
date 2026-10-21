import sys
from pathlib import Path

import numpy as np


EPS = 1e-12


def read_input(filename):
    lines = [line for line in Path(filename).read_text().splitlines() if line.strip()]
    if len(lines) != 5:
        raise ValueError("Во входном файле должно быть 5 строк.")

    n = int(lines[0])
    diagonals = [np.array(list(map(float, line.split()))) for line in lines[1:]]
    lower, main, upper, rhs = diagonals
    if n < 2 or tuple(map(len, diagonals)) != (n - 1, n, n - 1, n):
        raise ValueError("Неверное количество элементов диагоналей или правой части.")
    if not all(np.all(np.isfinite(values)) for values in diagonals):
        raise ValueError("Все входные данные должны быть конечными числами.")
    return lower, main, upper, rhs


def sweep(lower, main, upper, rhs):
    n = len(main)
    P, Q = np.zeros(n), np.zeros(n)

    for i in range(n):
        denominator = main[i] + (lower[i - 1] * P[i - 1] if i else 0)
        scale = max(abs(main[i]), abs(lower[i - 1]) if i else 0, abs(upper[i]) if i < n - 1 else 0)
        if scale == 0 or abs(denominator) <= EPS * scale:
            raise ValueError(f"Нулевой знаменатель на шаге {i + 1}.")
        P[i] = -upper[i] / denominator if i < n - 1 else 0
        Q[i] = (rhs[i] - (lower[i - 1] * Q[i - 1] if i else 0)) / denominator

    x = np.zeros(n)
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x, P, Q


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("input.txt")
    lower, main, upper, rhs = read_input(filename)
    x, P, Q = sweep(lower, main, upper, rhs)

    print("Прогоночные коэффициенты:")
    for i, (p, q) in enumerate(zip(P, Q), 1):
        print(f"i = {i}: P = {p:.6f}, Q = {q:.6f}")
    print("\nРешение:", np.array2string(x, precision=6, suppress_small=True))

    A = np.diag(main) + np.diag(lower, -1) + np.diag(upper, 1)
    numpy_x = np.linalg.solve(A, rhs)
    print("Сравнение с NumPy:", np.max(np.abs(x - numpy_x)))


if __name__ == "__main__":
    main()
