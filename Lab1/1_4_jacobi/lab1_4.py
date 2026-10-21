import math
import sys
from pathlib import Path

import numpy as np


MAX_ITERATIONS = 100000


def read_input(filename):
    lines = [line for line in Path(filename).read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Во входном файле недостаточно данных.")

    n, eps = int(lines[0]), float(lines[1])
    A = np.array([list(map(float, line.split())) for line in lines[2:]])
    if n <= 0 or A.shape != (n, n) or not np.isfinite(eps) or eps <= 0 or not np.all(np.isfinite(A)):
        raise ValueError("Неверные входные данные.")
    return A, eps


def off_diagonal_norm(A):
    return math.hypot(*(A[i, j] for i in range(len(A)) for j in range(i + 1, len(A))))


def max_position(A):
    return max((abs(A[i, j]), i, j) for i in range(len(A)) for j in range(i + 1, len(A)))[1:]


def jacobi(A, eps):
    n = len(A)
    current, V = A.copy(), np.eye(n)

    for k in range(MAX_ITERATIONS + 1):
        error = off_diagonal_norm(current)
        if error < eps:
            return np.diag(current).copy(), V, k, error
        if k == MAX_ITERATIONS:
            break

        p, q = max_position(current)
        difference = current[p, p] - current[q, q]
        angle = math.pi / 4 if abs(difference) < 1e-15 else 0.5 * math.atan(2 * current[p, q] / difference)
        c, s = math.cos(angle), math.sin(angle)
        H = np.eye(n)
        H[p, p] = H[q, q] = c
        H[p, q], H[q, p] = -s, s
        current = H.T @ current @ H
        V = V @ H

    raise RuntimeError("Метод вращений Якоби не сошелся.")


def show(name, value):
    print(f"{name}:\n{np.array2string(value, precision=6, suppress_small=True)}\n")


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("input.txt")
    A, eps = read_input(filename)
    if np.max(np.abs(A - A.T)) > 1e-10:
        raise ValueError("Матрица должна быть симметричной.")

    eigenvalues, V, iterations, error = jacobi(A, eps)
    Lambda = np.diag(eigenvalues)

    print("Точность:", eps)
    print("Количество вращений:", iterations)
    print("Итоговая внедиагональная норма:", error, "\n")
    show("Собственные значения", eigenvalues)
    for i in range(len(V)):
        show(f"Собственный вектор v{i + 1}", V[:, i])
    show("Матрица собственных векторов V", V)
    show("Матрица Lambda", Lambda)
    show("A * V", A @ V)
    show("V * Lambda", V @ Lambda)
    print("Ошибка A * V = V * Lambda:", np.max(np.abs(A @ V - V @ Lambda)))

    numpy_values = np.linalg.eigvalsh(A)
    print("\nСравнение с NumPy:")
    print("Собственные значения NumPy:", numpy_values)
    print("Максимальное отличие:", np.max(np.abs(np.sort(eigenvalues) - numpy_values)))


if __name__ == "__main__":
    main()
