# Lattice-Based Methods using solvediophant (LLL/BKZ Reduction)
# This file demonstrates the lattice-based approach that transforms 
# the Market Split Problem into a Shortest Vector Problem

# Pseudo-code for solvediophant approach
# Uses fpylll for LLL reduction

try:
    from fpylll import IntegerMatrix, LLL, BKZ
    FPYLLL_AVAILABLE = True
except ImportError:
    FPYLLL_AVAILABLE = False

import numpy as np

def solve_diophant_lattice(A, b, lambda_factor=100):
    """
    Transform MSP to Shortest Vector Problem using lattice reduction
    
    Args:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        lambda_factor: Scaling factor for lattice construction
        
    Returns:
        Solution vector or None if no solution found
    """
    if not FPYLLL_AVAILABLE:
        raise ImportError("fpylll library not available. Install with: pip install fpylll")
    
    m, n = A.shape
    L = IntegerMatrix(n + 1, n + m)
    
    # Construct lattice matrix for the system A*x = b
    # We want to find x such that A*x - b = 0
    # The lattice is constructed so that short vectors correspond to solutions
    
    # Identity matrix for x variables
    for i in range(n):
        L[i, i] = 1
    
    # Constraint matrix scaled by lambda_factor
    for i in range(n):
        for j in range(m):
            L[i, n + j] = lambda_factor * A[j, i]
    
    # Last row represents the target b
    for j in range(m):
        L[n, n + j] = -lambda_factor * b[j]

    # Apply LLL reduction
    LLL.reduction(L)
    
    # For a more aggressive reduction, we can also use BKZ
    # BKZ.reduction(L, BKZ.EasyParam(20))
    
    # Search for short vectors that correspond to solutions
    # In the reduced basis, short vectors that have a specific pattern
    # correspond to solutions of the original system
    
    # Extract the reduced basis vectors
    basis_vectors = []
    for i in range(n + 1):
        vector = [L[i, j] for j in range(n + m)]
        basis_vectors.append(vector)
    
    # Look for vectors where the first n components form a solution
    # A solution vector x should satisfy: L*x ≈ 0 (modulo the lattice structure)
    
    # Strategy: Look for vectors where the last m components are small
    # and the first n components form a valid binary solution
    
    solutions = []
    
    # Check each basis vector
    for vector in basis_vectors:
        # Extract x part (first n components)
        x_candidate = vector[:n]
        
        # Check if this is a reasonable candidate
        # A valid solution should have small values in the constraint part
        constraint_part = vector[n:n+m]
        
        # Calculate the residual A*x - b
        residual = A.dot(x_candidate) - b
        
        # Check if this is close to zero (allowing for scaling)
        if np.linalg.norm(residual) < 1e-6:
            # Check if x_candidate is binary or close to binary
            binary_score = sum(1 for val in x_candidate if abs(val) < 0.5 or abs(val - 1) < 0.5)
            if binary_score == n:
                # This looks like a valid solution
                # Round to nearest binary values
                x_solution = [1 if val > 0.5 else 0 for val in x_candidate]
                solutions.append(x_solution)
    
    # If we found solutions, return the best one
    if solutions:
        # Return the solution with smallest norm or first found
        return solutions[0]
    
    # If no exact solution found, try a different approach
    # Look for the shortest vector and use it to construct a solution
    
    # Find the shortest vector in the reduced basis
    min_norm = float('inf')
    best_vector = None
    
    for vector in basis_vectors:
        norm = np.linalg.norm(vector)
        if norm < min_norm:
            min_norm = norm
            best_vector = vector
    
    if best_vector is not None:
        # Try to extract a solution from the shortest vector
        x_candidate = best_vector[:n]
        
        # Round to binary values
        x_rounded = [1 if val > 0.5 else 0 for val in x_candidate]
        
        # Verify this is actually a solution
        residual = A.dot(x_rounded) - b
        if np.linalg.norm(residual) < 1e-6:
            return x_rounded
    
    # If no solution found, return None
    return None

class LatticeBasedSolver:
    """Lattice-based solver for Market Split Problem"""
    
    def __init__(self, lambda_factor=100):
        self.lambda_factor = lambda_factor
        
    def solve_market_split(self, A, b, time_limit=None):
        """Solve MSP using lattice reduction with improved post-processing"""
        import time
        start_time = time.time()
        
        try:
            solution = solve_diophant_lattice(A, b, self.lambda_factor)
            if solution is not None:
                # Apply bit-flip local search post-processing to improve solution
                improved_solution = self._bit_flip_local_search(solution, A, b)
                if improved_solution:
                    slack = np.sum(np.abs(A.dot(improved_solution) - b))
                    return {'x': improved_solution, 'slack_total': slack}, time.time() - start_time
                else:
                    slack = np.sum(np.abs(A.dot(solution) - b))
                    return {'x': solution, 'slack_total': slack}, time.time() - start_time
            else:
                # If no exact solution found, return minimal slack solution
                # Try a heuristic approach: find the best approximate solution
                n = A.shape[1]
                best_solution = None
                best_slack = float('inf')
                
                # Try a few random solutions as fallback
                np.random.seed(42)  # For reproducibility
                for _ in range(100):
                    x_candidate = np.random.randint(0, 2, n)
                    slack = np.sum(np.abs(A.dot(x_candidate) - b))
                    if slack < best_slack:
                        best_slack = slack
                        best_solution = x_candidate.tolist()
                
                # Apply local search to the best random solution
                if best_solution:
                    improved_solution = self._bit_flip_local_search(best_solution, A, b)
                    if improved_solution:
                        improved_slack = np.sum(np.abs(A.dot(improved_solution) - b))
                        if improved_slack < best_slack:
                            best_slack = improved_slack
                            best_solution = improved_solution
                
                return {'x': best_solution if best_solution else [0] * n, 
                       'slack_total': best_slack}, time.time() - start_time
        except Exception as e:
            print(f"Lattice solver error: {e}")
            return {'x': [0] * A.shape[1], 'slack_total': float('inf')}, time.time() - start_time
    
    def _bit_flip_local_search(self, solution, A, b, max_flips=None):
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
            max_flips = min(n, 20)  # Limit search to avoid exponential blowup
        
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

# Test function
def test_lattice_solver():
    """Test the lattice solver with a simple known instance"""
    # Create a simple test problem where we know the solution
    A = np.array([[1, 2, 3, 4], [2, 1, 1, 3]])
    # Solution should be x = [1, 0, 1, 1] giving b = [8, 6]
    true_x = [1, 0, 1, 1]
    b = A.dot(true_x)
    
    solver = LatticeBasedSolver()
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
        print(f"✗ Incorrect solution. Residual: {residual}")

if __name__ == "__main__":
    test_lattice_solver()
