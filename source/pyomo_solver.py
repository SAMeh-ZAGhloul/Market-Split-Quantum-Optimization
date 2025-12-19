"""
Pyomo Market Split Solver
Gracefully handles Pyomo availability with fallback mechanisms
"""

import time
import numpy as np

# Try to import Pyomo components
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    PYOMO_AVAILABLE = True
except ImportError as e:
    pyo = None
    SolverFactory = None
    PYOMO_AVAILABLE = False
    PYOMO_IMPORT_ERROR = str(e)

class PyomoMarketSplitSolver:
    """
    Market Split Problem solver using Pyomo optimization framework
    
    Gracefully handles cases where Pyomo is not available
    """
    
    def __init__(self):
        if not PYOMO_AVAILABLE:
            self.available = False
        else:
            self.available = True
    
    def solve_market_split(self, A, b, time_limit=None):
        """
        Solve Market Split Problem using Pyomo optimization
        
        Args:
            A: numpy array of shape (m, n) - constraint matrix
            b: numpy array of shape (m,) - target vector
            time_limit: time limit in seconds (optional)
            
        Returns:
            tuple: (solution_dict, solve_time)
        """
        if not self.available:
            # Fallback when Pyomo is not available
            start_time = time.time()
            n = A.shape[1]
            # Return all-zeros solution with high slack (simulates solver failure)
            return {
                'x': [0] * n,
                'slack_total': float('inf'),
                'error': f"Pyomo not available: {PYOMO_IMPORT_ERROR}",
                'fallback': True
            }, time.time() - start_time
        
        start_time = time.time()
        
        try:
            m, n = A.shape
            model = pyo.ConcreteModel()
            model.I = pyo.Set(initialize=range(m))
            model.J = pyo.Set(initialize=range(n))
            model.A = pyo.Param(model.I, model.J, initialize={(i,j): A[i,j] for i in range(m) for j in range(n)})
            model.b = pyo.Param(model.I, initialize={i: b[i] for i in range(m)})
            model.x = pyo.Var(model.J, domain=pyo.Binary)
            model.slack_plus = pyo.Var(model.I, domain=pyo.NonNegativeReals)
            model.slack_minus = pyo.Var(model.I, domain=pyo.NonNegativeReals)
            model.objective = pyo.Objective(rule=lambda model: sum(model.slack_plus[i] + model.slack_minus[i] for i in model.I), sense=pyo.Minimize)
            model.balance_constraint = pyo.Constraint(model.I, rule=lambda model, i: sum(model.A[i,j] * model.x[j] for j in model.J) + model.slack_minus[i] - model.slack_plus[i] == model.b[i])

            solver = SolverFactory('gurobi')
            if time_limit:
                solver.options['TimeLimit'] = time_limit
            results = solver.solve(model, tee=False)

            x_solution = [int(pyo.value(model.x[j])) for j in range(n)]
            slack_total = pyo.value(model.objective)
            
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
        """Get information about Pyomo availability"""
        if PYOMO_AVAILABLE:
            return {
                'available': True,
                'message': 'Pyomo is available and ready to use'
            }
        else:
            return {
                'available': False,
                'message': f'Pyomo not available: {PYOMO_IMPORT_ERROR}',
                'installation_hint': 'Install with: pip install pyomo',
                'python_version_note': 'Requires Python < 3.14',
                'solver_note': 'Requires a solver like Gurobi, CPLEX, or CBC'
            }

# Test function
def test_solver():
    """Test the Pyomo solver"""
    print("Testing Pyomo Market Split Solver")
    print("=" * 40)
    
    solver = PyomoMarketSplitSolver()
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
