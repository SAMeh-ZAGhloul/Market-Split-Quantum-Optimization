# Qiskit VQE/QAOA Solver for Market Split Problem
# Variational quantum algorithms on gate-based hardware

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.algorithms import VQE, QAOA
    from qiskit_optimization import QuadraticProgram
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.circuit.library import TwoLocal
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

def extract_solution_from_result(result, num_binary_vars):
    """
    Extract binary solution from VQE/QAOA result
    
    Args:
        result: VQE or QAOA result object
        num_binary_vars: Number of binary variables in the problem
        
    Returns:
        List of binary values
    """
    if hasattr(result, 'eigenstate') and result.eigenstate is not None:
        # Get the most probable bitstring
        if hasattr(result.eigenstate, 'bitstrings'):
            # Sample from the quantum state
            bitstrings = result.eigenstate.bitstrings()
            if bitstrings:
                # Get the most frequent bitstring
                from collections import Counter
                bitstring_counts = Counter(bitstrings)
                most_frequent_bitstring = bitstring_counts.most_common(1)[0][0]
                # Convert to binary list (take only first num_binary_vars)
                return [int(bit) for bit in most_frequent_bitstring[:num_binary_vars]]
        elif hasattr(result.eigenstate, 'to_dict'):
            # Get the state vector and find the most probable state
            state_dict = result.eigenstate.to_dict()
            if state_dict:
                # Find the state with highest probability
                max_prob = 0
                best_state = None
                for state_str, prob in state_dict.items():
                    if prob > max_prob:
                        max_prob = prob
                        best_state = state_str
                
                if best_state:
                    # Extract binary values (reverse order for Qiskit convention)
                    return [int(bit) for bit in reversed(best_state[-num_binary_vars:])]
    
    # Fallback: use the optimal parameters if available
    if hasattr(result, 'optimal_parameters') and result.optimal_parameters:
        # Convert parameters to binary values
        x_solution = []
        for i in range(num_binary_vars):
            param_key = f'x_{i}'
            if param_key in result.optimal_parameters:
                # Convert parameter value to binary (0 or 1)
                val = float(result.optimal_parameters[param_key])
                x_solution.append(1 if val > 0.5 else 0)
            else:
                x_solution.append(0)
        return x_solution
    
    # Last resort: return all zeros
    return [0] * num_binary_vars

def solve_vqe(A, b, max_iterations=1000):
    """
    Solve Market Split Problem using VQE (Variational Quantum Eigensolver)
    
    Args:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        max_iterations: Maximum VQE iterations
        
    Returns:
        Solution dictionary with x values and slack total
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit libraries not available. Install with: pip install qiskit qiskit-optimization")
    
    m, n = A.shape
    
    # Create quadratic program
    qp = QuadraticProgram()
    
    # Binary variables for retailer selection
    for i in range(n):
        qp.binary_var(f'x_{i}')
    
    # Integer variables for slack (bounded)
    slack_upper = 1000  # Upper bound for slack variables
    for i in range(m):
        qp.integer_var(0, slack_upper, f'slack_plus_{i}')
        qp.integer_var(0, slack_upper, f'slack_minus_{i}')
    
    # Objective: minimize total slack
    slack_obj = {}
    for i in range(m):
        slack_obj[(f'slack_plus_{i}', f'slack_plus_{i}')] = 1.0
        slack_obj[(f'slack_minus_{i}', f'slack_minus_{i}')] = 1.0
    
    qp.minimize(quadratic=slack_obj)
    
    # Add constraints: sum(A[i,j] * x[j]) + slack_minus[i] - slack_plus[i] = b[i]
    for i in range(m):
        linear_constraint = {}
        for j in range(n):
            linear_constraint[f'x_{j}'] = A[i, j]
        linear_constraint[f'slack_minus_{i}'] = -1
        linear_constraint[f'slack_plus_{i}'] = 1
        qp.linear_constraint(linear=linear_constraint, sense='==', rhs=b[i])
    
    # Convert to Ising Hamiltonian
    ising_operator, offset = qp.to_ising()
    
    # Set up VQE
    ansatz = TwoLocal(qp.get_num_binary_vars(), 'ry', 'cz', reps=1, entanglement='linear')
    optimizer = COBYLA(maxiter=max_iterations)
    
    vqe = VQE(ansatz=ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(ising_operator)
    
    # Extract solution properly
    x_solution = extract_solution_from_result(result, n)
    
    # Calculate slack
    slack_total = 0
    for i in range(m):
        actual = sum(A[i, j] * x_solution[j] for j in range(n))
        slack = abs(actual - b[i])
        slack_total += slack
    
    return {'x': x_solution, 'slack_total': slack_total}

def solve_qaoa(A, b, p=1, max_iterations=1000):
    """
    Solve Market Split Problem using QAOA (Quantum Approximate Optimization Algorithm)
    
    Args:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        p: Number of QAOA layers
        max_iterations: Maximum optimization iterations
        
    Returns:
        Solution dictionary with x values and slack total
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit libraries not available. Install with: pip install qiskit qiskit-optimization")
    
    m, n = A.shape
    
    # Create quadratic program (same as VQE)
    qp = QuadraticProgram()
    
    # Binary variables for retailer selection
    for i in range(n):
        qp.binary_var(f'x_{i}')
    
    # Integer variables for slack (bounded)
    slack_upper = 1000
    for i in range(m):
        qp.integer_var(0, slack_upper, f'slack_plus_{i}')
        qp.integer_var(0, slack_upper, f'slack_minus_{i}')
    
    # Objective: minimize total slack
    slack_obj = {}
    for i in range(m):
        slack_obj[(f'slack_plus_{i}', f'slack_plus_{i}')] = 1.0
        slack_obj[(f'slack_minus_{i}', f'slack_minus_{i}')] = 1.0
    
    qp.minimize(quadratic=slack_obj)
    
    # Add constraints
    for i in range(m):
        linear_constraint = {}
        for j in range(n):
            linear_constraint[f'x_{j}'] = A[i, j]
        linear_constraint[f'slack_minus_{i}'] = -1
        linear_constraint[f'slack_plus_{i}'] = 1
        qp.linear_constraint(linear=linear_constraint, sense='==', rhs=b[i])
    
    # Set up QAOA
    optimizer = COBYLA(maxiter=max_iterations)
    qaoa = QAOA(optimizer=optimizer, reps=p)
    
    result = qaoa.compute_minimum_eigenvalue(qp.to_ising()[0])
    
    # Extract solution properly
    x_solution = extract_solution_from_result(result, n)
    
    # Calculate slack
    slack_total = 0
    for i in range(m):
        actual = sum(A[i, j] * x_solution[j] for j in range(n))
        slack = abs(actual - b[i])
        slack_total += slack
    
    return {'x': x_solution, 'slack_total': slack_total}

class QiskitMarketSplitSolver:
    """Qiskit VQE/QAOA solver for Market Split Problem"""
    
    def __init__(self, method='vqe', p=1, max_iterations=1000):
        self.method = method
        self.p = p
        self.max_iterations = max_iterations
        
    def solve_market_split(self, A, b, time_limit=None):
        """Solve MSP using Qiskit VQE or QAOA"""
        import time
        start_time = time.time()
        
        try:
            if self.method.lower() == 'vqe':
                solution = solve_vqe(A, b, self.max_iterations)
            elif self.method.lower() == 'qaoa':
                solution = solve_qaoa(A, b, self.p, self.max_iterations)
            else:
                raise ValueError(f"Unknown method: {self.method}. Use 'vqe' or 'qaoa'")
            
            return solution, time.time() - start_time
        except Exception as e:
            print(f"Qiskit solver error: {e}")
            return {'x': [0] * A.shape[1], 'slack_total': float('inf')}, time.time() - start_time

# Test function
def test_qiskit_solver():
    """Test the Qiskit solver with a simple known instance"""
    # Create a simple test problem where we know the solution
    A = np.array([[1, 2, 3], [2, 1, 1]])
    # Solution should be x = [1, 1, 0] giving b = [3, 3]
    true_x = [1, 1, 0]
    b = A.dot(true_x)
    
    solver = QiskitMarketSplitSolver(method='vqe', max_iterations=100)
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
    test_qiskit_solver()
