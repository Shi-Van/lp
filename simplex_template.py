import numpy as np
import argparse

eps = 0.00001

def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    m, n_total = A.shape
    n = n_total - m

    if basis is None or nbasis is None:
        basis = list(range(n, n_total))
        nbasis = list(range(0, n))

    max_iterations = 100000
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        B = A[:, basis]
        N = A[:, nbasis]
        c_B = c[basis]
        c_N = c[nbasis]

        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        y = np.linalg.solve(B.T, c_B)
        reduced_cost = c_N - N.T @ y

        if np.all(reduced_cost <= eps):
            x = np.zeros(n_total)
            x[basis] = B_inv @ b
            x[nbasis] = 0
            obj = c @ x
            return "optimal", x, obj

        entering_candidates = np.where(reduced_cost > eps)[0]

        if len(entering_candidates) == 0:
            x = np.zeros(n_total)
            x[basis] = B_inv @ b
            x[nbasis] = 0
            obj = c @ x
            return "optimal", x, obj

        j_local = entering_candidates[np.argmax(reduced_cost[entering_candidates])]
        entering_var = nbasis[j_local]
        d = B_inv @ A[:, entering_var]

        if np.all(d <= eps):
            return "unbounded", None, None

        x_B = B_inv @ b
        ratios = []
        ratio_indices = []
        for i in range(m):
            if d[i] > eps:
                ratio = x_B[i] / d[i]
                ratios.append(ratio)
                ratio_indices.append(i)

        if len(ratios) == 0:
            return "unbounded", None, None

        min_ratio = min(ratios)
        candidates = [ratio_indices[i] for i in range(len(ratios)) if abs(ratios[i] - min_ratio) < eps]
        leaving_local = min(candidates, key=lambda i: basis[i])
        leaving_var = basis[leaving_local]
        basis[leaving_local] = entering_var
        nbasis[j_local] = leaving_var

    print("Warning: Maximum iterations reached")

    x = np.zeros(n_total)
    B = A[:, basis]
    B_inv = np.linalg.inv(B)
    x[basis] = B_inv @ b
    x[nbasis] = 0
    obj = c @ x
    return "optimal", x, obj


def DualSimplex(c, A, b, basis=None, nbasis=None):
    m, n_total = A.shape
    n = n_total - m

    if basis is None or nbasis is None:
        basis = list(range(n, n_total))
        nbasis = list(range(0, n))

    d = np.ones(len(nbasis))
    max_iterations = 100000
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        B = A[:, basis]
        N = A[:, nbasis]
        c_B = c[basis]
        c_N = c[nbasis]

        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        x_B = B_inv @ b
        y = np.linalg.solve(B.T, c_B)
        reduced_cost = c_N - N.T @ y
        r = d * reduced_cost

        if np.all(x_B >= -eps) and np.all(r <= eps):
            x = np.zeros(n_total)
            x[basis] = x_B
            x[nbasis] = 0
            obj = c @ x
            return "optimal", x, obj

        if np.all(x_B >= -eps):
            return "need_primal", None, None

        violations = np.where(x_B < -eps)[0]

        if len(violations) == 0:
            x = np.zeros(n_total)
            x[basis] = x_B
            x[nbasis] = 0
            obj = c @ x
            return "optimal", x, obj

        i = violations[np.argmin(x_B[violations])]
        leaving_local = i
        leaving_var = basis[i]
        B_inv_N = B_inv @ N
        a_hat_i = B_inv_N[i, :]
        sigma = -a_hat_i * d

        if np.all(sigma <= eps):
            return "infeasible", None, None

        ratios = []
        ratio_indices = []

        for j in range(len(nbasis)):
            if sigma[j] > eps:
                ratio = -r[j] / sigma[j]
                ratios.append(ratio)
                ratio_indices.append(j)

        if len(ratios) == 0:
            return "infeasible", None, None

        min_ratio = min(ratios)
        candidates = [ratio_indices[idx] for idx in range(len(ratios)) if abs(ratios[idx] - min_ratio) < eps]
        j_local = min(candidates, key=lambda j: nbasis[j])
        entering_var = nbasis[j_local]
        t = min_ratio
        r = r + sigma * t
        basis[leaving_local] = entering_var
        nbasis[j_local] = leaving_var
        d_new = np.ones(len(nbasis))
        d = d_new

    print("Warning: Maximum iterations reached")
    x = np.zeros(n_total)
    B = A[:, basis]
    B_inv = np.linalg.inv(B)
    x[basis] = B_inv @ b
    x[nbasis] = 0
    obj = c @ x
    return "optimal", x, obj


def Phase1(c, A, b):
    m, n_original = A.shape
    new_c = np.zeros(n_original + m)
    new_c[n_original:] = -1
    new_A = np.hstack([A, np.eye(m)])
    basis = list(range(n_original, n_original + m))
    nbasis = list(range(0, n_original))
    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)

    if status != "optimal" or obj < -eps:
        return "infeasible", None, None

    final_basis = []
    final_nbasis = list(range(0, n_original))

    for var in basis:
        if var < n_original:
            final_basis.append(var)
            if var in final_nbasis:
                final_nbasis.remove(var)

    while len(final_basis) < m:
        for var in final_nbasis:
            test_basis = final_basis + [var]
            if len(test_basis) <= m:
                test_matrix = A[:, test_basis]
                if len(test_basis) == m:
                    try:
                        np.linalg.inv(test_matrix)
                        final_basis.append(var)
                        final_nbasis.remove(var)
                        break
                    except np.linalg.LinAlgError:
                        continue
                else:
                    final_basis.append(var)
                    final_nbasis.remove(var)
                    break
        else:
            break

    if len(final_basis) != m:
        return "infeasible", None, None

    restored_c = c
    restored_A = A
    return PrimalSimplex(restored_c, restored_A, b, final_basis, final_nbasis)


def Solve(c, A, b):
    m, n = A.shape
    A_extended = np.hstack([A, np.eye(m)])
    c_extended = np.hstack([c, np.zeros(m)])
    basis = list(range(n, n + m))
    nbasis = list(range(0, n))
    is_dual_feasible = np.all(c <= eps)

    if is_dual_feasible:
        status, x_extended, obj = DualSimplex(c_extended, A_extended, b, basis, nbasis)
        if status == "need_primal":
            if np.all(b >= -eps):
                status, x_extended, obj = PrimalSimplex(c_extended, A_extended, b, basis, nbasis)
            else:
                status, x_extended, obj = Phase1(c_extended, A_extended, b)
    else:
        if np.all(b >= -eps):
            status, x_extended, obj = PrimalSimplex(c_extended, A_extended, b, basis, nbasis)
        else:
            status, x_extended, obj = Phase1(c_extended, A_extended, b)


    if status == "optimal":
        return status, x_extended[:n], obj
    else:
        return status, None, None


def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Dual Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    return parser.parse_args()


def main():
    args = proc_cmd()
    with open(args.filename, 'r', encoding='utf-8') as f:
        n, m = map(int, f.readline().split())
        c = np.array(list(map(float, f.readline().split())))
        A = []
        b = []
        for _ in range(m):
            *row, bi = map(float, f.readline().split())
            A.append(row)
            b.append(bi)
        A = np.array(A)
        b = np.array(b)
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)
    print("Solving the linear program using the Dual Simplex method...\n")
    status, solution, objective = Solve(c, A, b)
    print("\nResult:")
    print("Status:", status)
    
    if status == "optimal":
        print("Optimal solution x* =", solution)
        print("Optimal value =", objective)
    elif status == "unbounded":
        print("The problem is unbounded.")
    else:
        print("No solution found.")


if __name__ == '__main__':
    main()