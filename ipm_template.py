import numpy as np
import argparse
from scipy.linalg import lu_factor, lu_solve



def compute_max_step(x, dx):
    x = np.asarray(x).flatten()
    dx = np.asarray(dx).flatten()
    eps = 1e-25
    
    negative_mask = dx < -eps
    if not np.any(negative_mask):
        return np.inf
    
    ratios = -x[negative_mask] / dx[negative_mask]
    return np.min(ratios)


def max_abs(x):
    return np.max(np.abs(x))


def solve_kkt_full(A, x, z, y, w, sigma, rho, mu, xz_corr=None, yw_corr=None):
    m, n = A.shape
    x = x.flatten()
    z = z.flatten()
    y = y.flatten()
    w = w.flatten()
    sigma = sigma.flatten()
    rho = rho.flatten()
    
    M = np.zeros((n + m, n + m))
    M[:n, :n] = np.diag(z / x)
    M[:n, n:] = A.T
    M[n:, :n] = A
    M[n:, n:] = -np.diag(w / y)
    
    if xz_corr is None:
        xz_corr = np.zeros(n)
    if yw_corr is None:
        yw_corr = np.zeros(m)
    
    r_xz = mu - x * z + xz_corr
    r_yw = mu - y * w + yw_corr
    rhs = np.concatenate([
        sigma + r_xz / x,
        rho - r_yw / y
    ])
    
    LU, pivot = lu_factor(M)
    delta = lu_solve((LU, pivot), rhs)
    
    dx = delta[:n]
    dy = delta[n:]
    dz = A.T @ dy - sigma
    dw = rho - A @ dx
    
    return dx, dy, dz, dw


def Solve(c, A, b):
    m, n_total = A.shape
    n = n_total - m
    
    c_orig = c[:n]
    A_orig = A[:, :n]
    b_orig = b
    
    x = np.ones(n)
    z = np.ones(n)
    y = np.ones(m)
    w = np.ones(m)
    
    max_iter = 200
    
    for step in range(max_iter):
        rho = b_orig - A_orig @ x - w
        sigma = c_orig - A_orig.T @ y + z
        
        gamma = (x * z).sum() + (y * w).sum()
        mu_hat = gamma / (m + n)
        
        is_rho_optimal = max_abs(rho) < 1e-8 * (1 + max_abs(b_orig))
        is_sigma_optimal = max_abs(sigma) < 1e-8 * (1 + max_abs(c_orig))
        is_mu_optimal = mu_hat < 1e-8
        
        if is_rho_optimal and is_sigma_optimal and is_mu_optimal:
            obj = float(c[:n] @ x)
            return "optimal", x, obj
         
        dx_aff, dy_aff, dz_aff, dw_aff = solve_kkt_full(
            A_orig, x, z, y, w, sigma, rho, mu=0.0
        )
        
        
        max_aff = compute_max_step(x, dx_aff)
        max_aff = min(max_aff, compute_max_step(y, dy_aff))
        max_aff = min(max_aff, compute_max_step(z, dz_aff))
        max_aff = min(max_aff, compute_max_step(w, dw_aff))
        theta_aff = min(1.0, max_aff)
        
        
        x_pred = x + theta_aff * dx_aff
        z_pred = z + theta_aff * dz_aff
        y_pred = y + theta_aff * dy_aff
        w_pred = w + theta_aff * dw_aff
        
        gamma_pred = (x_pred * z_pred).sum() + (y_pred * w_pred).sum()
        mu_hat_pred = gamma_pred / (m + n)
        
        correction = (mu_hat_pred / mu_hat) ** 2.5 if mu_hat > 1e-30 else 0.0
        mu_hat = correction * mu_hat
        mu = 0.15 * mu_hat
        
        xz_corr = -dx_aff * dz_aff
        yw_corr = -dy_aff * dw_aff
        
        dx, dy, dz, dw = solve_kkt_full(
            A_orig, x, z, y, w, sigma, rho, mu,
            xz_corr=xz_corr, yw_corr=yw_corr
        )
        
        max_step = compute_max_step(x, dx)
        max_step = min(max_step, compute_max_step(y, dy))
        max_step = min(max_step, compute_max_step(z, dz))
        max_step = min(max_step, compute_max_step(w, dw))
        theta = min(1.0, 0.99 * max_step)
        
        x = x + theta * dx
        y = y + theta * dy
        z = z + theta * dz
        w = w + theta * dw
        
        eps = 1e-25
        if not ((x > eps).all() and (z > eps).all() and (y > eps).all() and (w > eps).all()):
            return "unbounded", None, None
        
        if max_abs(x) > 1e50 or max_abs(z) > 1e50 or \
           max_abs(y) > 1e50 or max_abs(w) > 1e50:
            return "unbounded", None, None
    
    
    obj = float(c[:n] @ x)
    return "infeasible", x, obj


def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    return parser.parse_args()

def main():
    # boilerplate for reading input data
    args = proc_cmd()
    try:
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
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found.")
        return

    A_full = np.hstack([A, np.eye(m)])
    c_full = np.concatenate([c, np.zeros(m)])
    
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)

    print("Solving the linear program using Mehrotra Predictor-Corrector IPM...\n")
    status, solution, objective = Solve(c_full, A_full, b)
    
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