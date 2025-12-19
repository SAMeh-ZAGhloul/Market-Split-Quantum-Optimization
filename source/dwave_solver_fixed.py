# D-Wave Quantum Annealing Solver for Market Split Problem
# Maps MSP to QUBO and solves on quantum annealer with improved formulation

import numpy as np

try:
    import dwave_binary_quadratic_model as dqm
    import dimod
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False

def create_qubo_matrix(A, b, penalty=1000.0):
    """
    Create QUBO matrix for Market Split Problem with improved formulation
    
    The QUBO formulation minimizes:
    P * sum_i (sum_j A[i,j] * x[j] + slack_minus[i] - slack_plus[i] - b[i])^2 + sum_i (slack_plus[i] + slack_minus[i])
    
    Args:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        penalty: Penalty coefficient for constraints (should be large enough)
        
    Returns:
        Q: QUBO matrix (n + 2*m x n + 2*m)
        c: Linear coefficient vector
    """
    m, n = A.shape
    size = n + 2 * m  # Variables: x[0:n], slack_plus[n:n+m], slack_minus[n+m:n+2*m]
    
    Q = np.zeros((size, size))
    c = np.zeros(size)
    
    # Auto-tune penalty coefficient based on problem size
    max_coeff = np.max(np.abs(A)) if A.size > 0 else 1
    max_b = np.max(np.abs(b)) if b.size > 0 else 1
    estimated_max_objective = m * max_b * max_coeff
    
    # Ensure penalty is sufficiently large
    min_penalty = estimated_max_objective * 10 + 100
    if penalty < min_penalty:
        penalty = min_penalty
    
    # Objective: minimize sum of slack variables (encourages small slack)
    for i in range(n, n + m):
        c[i] = 1.0  # slack_plus coefficient
    for i in range(n + m, n + 2 * m):
        c[i] = 1.0  # slack_minus coefficient
    
    # Constraint penalties: P * (sum(A[i,j] * x[j]) + slack_minus[i] - slack_plus[i] - b[i])^2
    for i in range(m):
        # Expand the squared constraint: (Ax + slack_minus - slack_plus - b)^2
        # = (Ax)^2 + slack_minus^2 + slack_plus^2 + b^2 + 2*Ax*slack_minus - 2*Ax*slack_plus - 2*b*Ax - 2*b*slack_minus + 2*b*slack_plus
        
        # Linear terms from constraint expansion
        for j in range(n):
            c[j] += penalty * (-2 * b[i] * A[i, j])
        
        # Slack linear terms
        c[n + i] += penalty * (-2 * b[i] * (-1))  # slack_minus
        c[n + m + i] += penalty * (-2 * b[i] * 1)  # slack_plus
        
        # Quadratic terms for x variables (Ax)^2
        for j1 in range(n):
            for j2 in range(j1, n):
                if j1 == j2:
                    Q[j1, j2] += penalty * (A[i, j1] * A[i, j2])
                else:
                    Q[j1, j2] += 2 * penalty * (A[i, j1] * A[i, j2])
        
        # Quadratic terms for slack variables
        Q[n + i, n + i] += penalty * ((-1) * (-1))  # slack_minus squared
        Q[n + m + i, n + m + i] += penalty * (1 * 1)  # slack_plus squared
        
        # Cross terms between x and slack variables
        for j in range(n):
            # x_j * slack_minus_i term
            Q[j, n + i] += 2 * penalty * (A[i, j] * (-1))
            # x_j * slack_plus_i term  
            Q[j, n + m + i] += 2 * penalty * (A[i, j] * 1)
        
        # Cross term between slack variables
        Q[n + i, n + m + i] += 2 * penalty * ((-1) * 1)  # slack_minus_i * slack_plus_i
    
    return Q, c

def solve_dwave_quantum_annealing(A, b, num_reads=1000, penalty=None):
    """
    Solve Market Split Problem using D-Wave quantum annealing
    
    Args:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        num_reads: Number of quantum annealing runs
        penalty: Penalty coefficient for constraints (auto-tuned if None)
        
    Returns:
        Solution dictionary with x values and slack total
    """
    if not DWAVE_AVAILABLE:
        raise ImportError("D-Wave libraries not available. Install with: pip install dwave-ocean-sdk")
    
    # Auto-tune penalty if not provided
    if penalty is None:
        m, n = A.shape
        max_coeff = np.max(np.abs(A)) if A.size > 0 else 1
        max_b = np.max(np.abs(b)) if b.size > 0 else 1
        estimated_max = m * max_b * max_coeff
        penalty = max(1000, estimated_max * 10)
    
    # Create QUBO matrix with proper penalty
    Q, c = create_qubo_matrix(A, b, penalty)
    
    # Create binary quadratic model
    # Note: dimod expects the Q matrix in upper triangular form
    bqm = dimod.BinaryQuadraticModel(Q, c, 0.0, dimod.BINARY)
    
    # Solve using simulated annealing (for testing without quantum hardware)
    sampler = dimod.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    
    # Get the best solution (lowest energy)
    best_sample = min(response.samples(), key=lambda x: x.energy)
    
    # Extract solution
    n = A.shape[1]
    m = A.shape[0]
    x_solution = [int(best_sample[j]) for j in range(n)]
    
    # Apply bit-flip local search post-processing
    x_improved = bit_flip_local_search(x_solution, A, b)
    if x_improved:
        x_solution = x_improved
    
    # Calculate slack
    slack_total = 0
    for i in range(m):
        actual = sum(A[i, j] * x_solution[j] for j in range(n))
        slack = abs(actual - b[i])
        slack_total += slack
    
    return {'x': x_solution, 'slack_total': slack_total}

def bit_flip_local_search(solution, A, b, max_flips=None):
    """
    Improve solution using bit-flip local search
    
    Args:
        solution: Initial binary solution
        A: Constraint matrix
        b: Target vector
        max_flips: Maximum number of bit flips to try
        
    Returns:
        Improved solution or None if no improvement found
    """
    n = len(solution)
    if max_flips is None:
        max_flips = min(n, 15)  # Limit search for quantum methods
    
    current_solution = solution.copy()
    current_slack = np.sum(np.abs(A.dot(current_solution) - b))
    
    improved = True
    flips_attempted = 0
    
    while improved and flips_attempted < max_flips:
        improved = False
        best_flip = None
        best_slack = current_slack
        
        # Try flipping each bit
        for i in range(n):
            if flips_attempted >= max_flips:
                break
                
            test_solution = current_solution.copy()
            test_solution[i] = 1 - test_solution[i]  # Flip bit
            test_slack = np.sum(np.abs(A.dot(test_solution) - b))
            
            if test_slack < best_slack:
                best_slack = test_slack
                best_flip = i
                improved = True
            
            flips_attempted += 1
        
        # Apply the best improvement found
        if improved and best_flip is not None:
            current_solution[best_flip] = 1 - current_solution[best_flip]
            current_slack = best_slack
    
    # Return improved solution only if we actually improved
    final_slack = np.sum(np.abs(A.dot(current_solution) - b))
    if final_slack < np.sum(np.abs(A.dot(solution) - b)):
        return current_solution.tolist()
    else:
        return None

class DWaveMarketSplitSolver:
    """D-Wave quantum annealing solver for Market Split Problem"""
    
    def __init__(self, penalty=None, num_reads=1000):
        self.penalty = penalty
        self.num_reads = num_reads
        
    def solve_market_split(self, A, b, time_limit=None):
        """Solve MSP using D-Wave quantum annealing with post-processing"""
        import time
        start_time = time.time()
        
        try:
            solution = solve_dwave_quantum_annealing(A, b, self.num_reads, self.penalty)
            return solution, time.time() - start_time
        except Exception as e:
            print(f"D-Wave solver error: {e}")
            return {'x': [0] * A.shape[1], 'slack_total': float('inf')}, time.time() - start_time

# Test function
def test_dwave_solver():
    """Test the D-Wave solver with a simple known instance"""
    # Create a simple test problem where we know the solution
    A = np.array([[1, 2, 1], [2, 1, 2]])
    # Solution should be x = [1, 1, 0] giving b = [3, 3]
    true_x = [1, 1, 0]
    b = A.dot(true_x)
    
    solver = DWaveMarketSplitSolver(penalty=100.0, num_reads=100)
    solution, solve_time = solver.solve_market_split(A, b)
    
    print(f"Test problem: A = \n{A}")
    print(f"Target b = {b}")
    print(f"Expected solution: {true_x}")
    print(f"Found solution: {solution['x']}")
    print(f"Solve time: {solve_time:.3f}s")
    print(f"Slack: {solution['slack_total']}")
    
    # Verify the solution
    if solution['x'] == true_x:
        print("✓ Correct solution found!")
    else:
        residual = A.dot(solution['x']) - b
        print(f"✗ Solution found. Residual: {residual}")

if __name__ == "__main__":
    test_dwave_solver()
