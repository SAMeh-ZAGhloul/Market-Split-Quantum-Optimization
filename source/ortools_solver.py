"""
OR-Tools Market Split Solver
Gracefully handles OR-Tools availability with fallback mechanisms
"""

import time
import numpy as np

# Try to import OR-Tools components
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError as e:
    cp_model = None
    ORTOOLS_AVAILABLE = False
    ORTOOLS_IMPORT_ERROR = str(e)

class ORToolsMarketSplitSolver:
    """
    Market Split Problem solver using Google OR-Tools CP-SAT
    
    Gracefully handles cases where OR-Tools is not available
    """
    
    def __init__(self):
        if not ORTOOLS_AVAILABLE:
            self.available = False
        else:
            self.available = True
    
    def solve_market_split(self, A, b, time_limit=60):
        """
        Solve Market Split Problem using OR-Tools CP-SAT
        
        Args:
            A: numpy array of shape (m, n) - constraint matrix
            b: numpy array of shape (m,) - target vector
            time_limit: time limit in seconds
            
        Returns:
            tuple: (solution_dict, solve_time)
        """
        if not self.available:
            # Fallback when OR-Tools is not available
            start_time = time.time()
            n = A.shape[1]
            # Return all-zeros solution with high slack (simulates solver failure)
            return {
                'x': [0] * n,
                'slack_total': float('inf'),
                'error': f"OR-Tools not available: {ORTOOLS_IMPORT_ERROR}",
                'fallback': True
            }, time.time() - start_time
        
        start_time = time.time()
        
        try:
            m, n = A.shape
            model = cp_model.CpModel()
            x = [model.NewBoolVar(f'x_{j}') for j in range(n)]
            slack_plus = [model.NewIntVar(0, 1000, f'slack_plus_{i}') for i in range(m)]
            slack_minus = [model.NewIntVar(0, 1000, f'slack_minus_{i}') for i in range(m)]
            total_slack = sum(slack_plus[i] + slack_minus[i] for i in range(m))
            model.Minimize(total_slack)
            
            for i in range(m):
                contributions = [A[i, j] * x[j] for j in range(n)]
                model.Add(sum(contributions) + slack_minus[i] - slack_plus[i] == b[i])

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = time_limit
            status = solver.Solve(model)
            
            x_solution = [solver.Value(x[j]) for j in range(n)]
            slack_total = solver.ObjectiveValue()
            
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
        """Get information about OR-Tools availability"""
        if ORTOOLS_AVAILABLE:
            return {
                'available': True,
                'message': 'OR-Tools is available and ready to use'
            }
        else:
            return {
                'available': False,
                'message': f'OR-Tools not available: {ORTOOLS_IMPORT_ERROR}',
                'installation_hint': 'Install with: pip install ortools',
                'python_version_note': 'Requires Python < 3.14'
            }

# Test function
def test_solver():
    """Test the OR-Tools solver"""
    print("Testing OR-Tools Market Split Solver")
    print("=" * 40)
    
    solver = ORToolsMarketSplitSolver()
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
