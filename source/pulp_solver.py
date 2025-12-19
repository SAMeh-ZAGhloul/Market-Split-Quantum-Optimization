"""
PuLP Market Split Solver
Gracefully handles PuLP availability with fallback mechanisms
"""

import time
import numpy as np

# Try to import PuLP components
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError as e:
    pulp = None
    PULP_AVAILABLE = False
    PULP_IMPORT_ERROR = str(e)

class PuLPMarketSplitSolver:
    """
    Market Split Problem solver using PuLP linear programming
    
    Gracefully handles cases where PuLP is not available
    """
    
    def __init__(self):
        if not PULP_AVAILABLE:
            self.available = False
        else:
            self.available = True
    
    def solve_market_split(self, A, b, time_limit=60):
        """
        Solve Market Split Problem using PuLP
        
        Args:
            A: numpy array of shape (m, n) - constraint matrix
            b: numpy array of shape (m,) - target vector
            time_limit: time limit in seconds
            
        Returns:
            tuple: (solution_dict, solve_time)
        """
        if not self.available:
            # Fallback when PuLP is not available
            start_time = time.time()
            n = A.shape[1]
            # Return all-zeros solution with high slack (simulates solver failure)
            return {
                'x': [0] * n,
                'slack_total': float('inf'),
                'error': f"PuLP not available: {PULP_IMPORT_ERROR}",
                'fallback': True
            }, time.time() - start_time
        
        start_time = time.time()
        
        try:
            m, n = A.shape
            
            # Create the linear programming problem
            prob = pulp.LpProblem("Market_Split_Problem", pulp.LpMinimize)
            
            # Decision variables
            x = [pulp.LpVariable(f'x_{j}', cat='Binary') for j in range(n)]
            slack_plus = [pulp.LpVariable(f'slack_plus_{i}', lowBound=0) for i in range(m)]
            slack_minus = [pulp.LpVariable(f'slack_minus_{i}', lowBound=0) for i in range(m)]
            
            # Objective function: minimize total slack
            total_slack = pulp.lpSum(slack_plus[i] + slack_minus[i] for i in range(m))
            prob += total_slack
            
            # Constraints: A @ x + slack_minus - slack_plus = b
            for i in range(m):
                constraint = pulp.lpSum(A[i, j] * x[j] for j in range(n)) + slack_minus[i] - slack_plus[i] == b[i]
                prob += constraint
            
            # Solve the problem
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
            
            # Extract solution
            x_solution = [int(pulp.value(x[j])) for j in range(n)]
            slack_total = pulp.value(prob.objective)
            
            return {
                'x': x_solution, 
                'slack_total': slack_total,
                'fallback': False
            }, time.time() - start_time
            
        except Exception as e:
            # Handle runtime errors gracefully
            n = A.shape[1]
            return {
                'x': [0] * n,
                'slack_total': float('inf'),
                'error': str(e),
                'fallback': True
            }, time.time() - start_time
    
    @staticmethod
    def get_availability_info():
        """Get information about PuLP availability"""
        if PULP_AVAILABLE:
            return {
                'available': True,
                'message': 'PuLP is available and ready to use'
            }
        else:
            return {
                'available': False,
                'message': f'PuLP not available: {PULP_IMPORT_ERROR}',
                'installation_hint': 'Install with: pip install pulp',
                'python_version_note': 'Python 3.14 compatible'
            }

# Test function
def test_solver():
    """Test the PuLP solver"""
    print("Testing PuLP Market Split Solver")
    print("=" * 40)
    
    solver = PuLPMarketSplitSolver()
    availability = solver.get_availability_info()
    print(f"Availability: {availability['available']}")
    print(f"Message: {availability['message']}")
    
    # Create a simple test problem
    A = np.array([[1, 2, 3], [2, 1, 1]])
    b = np.array([5, 3])
    
    solution, solve_time = solver.solve_market_split(A, b, time_limit=10)
    print(f"Solve time: {solve_time:.3f}s")
    print(f"Solution: {solution}")
    
    return availability['available']

if __name__ == "__main__":
    test_solver()
