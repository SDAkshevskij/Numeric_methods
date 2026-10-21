import sys
from pathlib import Path

import numpy as np


EPS = 1e-12


def read_input(filename):
    data = Path(filename).read_text(encoding="utf-8").split()
    if not data:
        raise ValueError("Входной файл пуст.")

    n = int(data[0])
    values = np.array(data[1:], dtype=float)

    if n <= 0 or len(values) != n * n + n or not np.all(np.isfinite(values)):
        raise ValueError("Неверное количество элементов A и b.")

    return values[:n * n].reshape(n, n), values[n * n:]


def lu_decomposition(A):
    n = len(A)
    L, U, P = np.eye(n), A.copy(), np.eye(n)
    scale = np.max(np.abs(A))
    swaps = 0

    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(U[i, k]))
        if scale == 0 or abs(U[pivot, k]) <= EPS * scale:
            raise ValueError("Матрица вырождена.")

        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            L[[k, pivot], :k] = L[[pivot, k], :k]
            swaps += 1

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U, swaps


def forward_substitution(L, b):
    y = np.zeros(len(b))
    for i in range(len(b)):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))
    return y


def backward_substitution(U, b):
    x = np.zeros(len(b))
    for i in range(len(b) - 1, -1, -1):
        total = sum(U[i, j] * x[j] for j in range(i + 1, len(b)))
        x[i] = (b[i] - total) / U[i, i]
    return x


def solve(P, L, U, b):
    return backward_substitution(U, forward_substitution(L, P @ b))


def inverse(P, L, U):
    return np.column_stack([solve(P, L, U, e) for e in np.eye(len(L))])


def show(name, value):
    print(f"{name}:\n{np.array2string(value, precision=6, suppress_small=True)}\n")


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("input.txt")
    A, b = read_input(filename)
    P, L, U, swaps = lu_decomposition(A)
    x = solve(P, L, U, b)
    A_inverse = inverse(P, L, U)
    LU, PA = L @ U, P @ A

    show("Матрица L", L)
    show("Матрица U", U)
    show("L * U", LU)
    show("Решение Ax = b", x)
    show("Обратная матрица A^(-1)", A_inverse)
    print("Определитель:", (-1) ** swaps * np.prod(np.diag(U)), "\n")
    show("Проверка A * A^(-1)", A @ A_inverse)
    show("P * A", PA)
    print("Ошибка L * U = P * A:", np.max(np.abs(LU - PA)))

    numpy_x = np.linalg.solve(A, b)
    numpy_inverse = np.linalg.inv(A)
    print("\nСравнение с NumPy:")
    print("Ошибка решения:", np.max(np.abs(x - numpy_x)))
    print("Ошибка обратной матрицы:", np.max(np.abs(A_inverse - numpy_inverse)))
    print("Ошибка определителя:", abs((-1) ** swaps * np.prod(np.diag(U)) - np.linalg.det(A)))


if __name__ == "__main__":
    main()
