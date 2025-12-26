import numpy as np
import argparse

eps = 0.00001


def solve_kkt_system(A, D2, r_p, r_d, r_sz, x_safe, term_sub):
    m = A.shape[0]
    M = (A * D2) @ A.T
    
    term = term_sub
    rhs = r_p + A @ (D2 * term)
    
    L = None
    reg = 1e-10
    for _ in range(6):
        try:
            M_reg = M + reg * np.eye(m)
            L = np.linalg.cholesky(M_reg)
            break
        except np.linalg.LinAlgError:
            reg *= 10
            
    if L is None:
        raise np.linalg.LinAlgError("Matrix not positive definite")
        
    t = np.linalg.solve(L, rhs)
    dy = np.linalg.solve(L.T, t)
    dx = D2 * (A.T @ dy - term)
    
    s_approx = x_safe / D2
    ds = (r_sz - s_approx * dx) / x_safe
    
    return dx, dy, ds


def get_step_length(v, dv, eta=0.99):
    mask = dv < 0
    if np.any(mask):
        return min(1.0, eta * np.min(-v[mask] / dv[mask]))
    return 1.0


def Solve(c, A, b, method="mehrotra"):
    m, n = A.shape
    c_min = -c
    

    x = np.ones(n)
    s = np.ones(n)
    y = np.zeros(m)
    
    max_iter = 100
    
    for k in range(max_iter):
        r_p = b - A @ x
        r_d = c_min - A.T @ y - s
        mu = np.dot(x, s) / n
        
        norm_rp = np.linalg.norm(r_p)
        norm_rd = np.linalg.norm(r_d)
        
        if norm_rp < eps and norm_rd < eps and mu < eps:
            return "optimal", x, c @ x
            
        if np.max(np.abs(x)) > 1e12:
            return "unbounded", None, None

        x_safe = np.maximum(x, 1e-16)
        s_safe = np.maximum(s, 1e-16)
        D2 = x_safe / s_safe
        
        try:
            if method == "simple":
                sigma = 0.5
                r_sz = -x * s + sigma * mu
                term = r_d - (1.0 / x_safe) * r_sz
                
                dx, dy, ds = solve_kkt_system(A, D2, r_p, r_d, r_sz, x_safe, term)
                
                alpha_p = get_step_length(x, dx, eta=0.99)
                alpha_d = get_step_length(s, ds, eta=0.99)
                
                x = x + alpha_p * dx
                y = y + alpha_d * dy
                s = s + alpha_d * ds
                
            else:
                r_sz_aff = -x * s
                term_aff = r_d - (1.0 / x_safe) * r_sz_aff
                dx_aff, dy_aff, ds_aff = solve_kkt_system(A, D2, r_p, r_d, r_sz_aff, x_safe, term_aff)
                
                alpha_aff_p = get_step_length(x, dx_aff, eta=1.0)
                alpha_aff_d = get_step_length(s, ds_aff, eta=1.0)
                
                mu_aff = np.dot(x + alpha_aff_p * dx_aff, s + alpha_aff_d * ds_aff) / n
                sigma = (mu_aff / mu)**3 if mu > 1e-16 else 0
                
                r_sz_corr = -x * s + sigma * mu * np.ones(n) - dx_aff * ds_aff
                term_corr = r_d - (1.0 / x_safe) * r_sz_corr
                
                dx, dy, ds = solve_kkt_system(A, D2, r_p, r_d, r_sz_corr, x_safe, term_corr)
                
                alpha_p = get_step_length(x, dx, eta=0.99)
                alpha_d = get_step_length(s, ds, eta=0.99)
                
                x = x + alpha_p * dx
                y = y + alpha_d * dy
                s = s + alpha_d * ds

        except np.linalg.LinAlgError:
            return "infeasible", None, None

    return "infeasible", None, None


def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    parser.add_argument("--method", choices=["simple", "mehrotra"], default="mehrotra", 
                        help="Method to use: 'simple' or 'mehrotra' (default).")
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

    method_name = "Mehrotra Predictor-Corrector" if args.method == "mehrotra" else "Basic Primal-Dual"
    print(f"Solving the linear program using {method_name} IPM...\n")
    
    status, solution, objective = Solve(c_full, A_full, b, method=args.method)
    
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