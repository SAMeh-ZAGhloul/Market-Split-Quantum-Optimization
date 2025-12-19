# Market Split Problem - Example Usage
# Demonstrates how to use the various solvers with graceful dependency handling

import numpy as np
import time
from typing import List, Dict, Any, Tuple

def generate_test_instance(seed=42, m=5, n=30):
    """
    Generate a test instance for Market Split Problem
    
    Args:
        seed: Random seed for reproducibility
        m: Number of products
        n: Number of retailers
        
    Returns:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        true_solution: The underlying solution (for verification)
    """
    rng = np.random.default_rng(seed)
    
    # Generate random matrix A (retailer demands for each product)
    A = rng.integers(1, 10, size=(m, n))
    
    # Generate true solution (which retailers to select)
    true_solution = rng.integers(0, 2, size=n)
    
    # Calculate target allocation b = A @ true_solution
    b = A @ true_solution
    
    return A, b, true_solution

def verify_solution(A, b, solution):
    """
    Verify if a solution is correct
    
    Args:
        A: Matrix of retailer demands (m x n)
        b: Target allocation vector (m,)
        solution: Solution dictionary with 'x' key
        
    Returns:
        Boolean indicating if solution is correct
    """
    x = solution['x']
    
    # Check if solution is binary
    if not all(val in [0, 1] for val in x):
        return False, "Solution is not binary"
    
    # Check if constraints are satisfied
    for i in range(A.shape[0]):
        actual = sum(A[i, j] * x[j] for j in range(len(x)))
        if actual != b[i]:
            return False, f"Constraint {i} violated: {actual} != {b[i]}"
    
    return True, "Solution is correct"

def get_available_solvers():
    """
    Get list of available solvers with graceful error handling
    
    Returns:
        tuple: (available_solvers_list, unavailable_solvers_dict)
    """
    available_solvers = []
    unavailable_solvers = {}
    
    # Try to import lattice solver (should always work)
    try:
        from lattice_solver import LatticeBasedSolver
        available_solvers.append(("Lattice-Based", LatticeBasedSolver()))
    except ImportError as e:
        unavailable_solvers['Lattice-Based'] = str(e)
    
    # Try to import OR-Tools solver
    try:
        from ortools_solver import ORToolsMarketSplitSolver
        ortools_solver = ORToolsMarketSplitSolver()
        if ortools_solver.available:
            available_solvers.append(("OR-Tools CP-SAT", ortools_solver))
        else:
            info = ortools_solver.get_availability_info()
            unavailable_solvers['OR-Tools CP-SAT'] = info['message']
    except ImportError as e:
        unavailable_solvers['OR-Tools CP-SAT'] = str(e)
    
    # Try to import Pyomo solver
    try:
        from pyomo_solver import PyomoMarketSplitSolver
        pyomo_solver = PyomoMarketSplitSolver()
        if pyomo_solver.available:
            available_solvers.append(("Pyomo + Gurobi", pyomo_solver))
        else:
            info = pyomo_solver.get_availability_info()
            unavailable_solvers['Pyomo + Gurobi'] = info['message']
    except ImportError as e:
        unavailable_solvers['Pyomo + Gurobi'] = str(e)
    
    # Try to import D-Wave solver
    try:
        from dwave_solver import DWaveMarketSplitSolver
        available_solvers.append(("D-Wave (SA)", DWaveMarketSplitSolver()))
    except ImportError as e:
        unavailable_solvers['D-Wave (SA)'] = str(e)
    
    # Try to import Qiskit solvers
    try:
        from qiskit_solver import QiskitMarketSplitSolver
        available_solvers.append(("Qiskit VQE", QiskitMarketSplitSolver(method='vqe')))
        available_solvers.append(("Qiskit QAOA", QiskitMarketSplitSolver(method='qaoa')))
    except ImportError as e:
        unavailable_solvers['Qiskit (VQE/QAOA)'] = str(e)
    
    return available_solvers, unavailable_solvers

def run_example():
    """Run example demonstrating different available solvers"""
    print("Market Split Problem - Example Usage")
    print("=" * 50)
    
    # Get available solvers
    available_solvers, unavailable_solvers = get_available_solvers()
    
    # Show solver availability
    print(f"Available solvers ({len(available_solvers)}):")
    for name, _ in available_solvers:
        print(f"  ✓ {name}")
    
    if unavailable_solvers:
        print(f"\nUnavailable solvers ({len(unavailable_solvers)}):")
        for name, reason in unavailable_solvers.items():
            print(f"  ✗ {name}: {reason}")
    
    if not available_solvers:
        print("Error: No solvers are available!")
        return []
    
    print()
    
    # Generate test instance
    print("Generating test instance...")
    A, b, true_solution = generate_test_instance(seed=42, m=5, n=30)
    print(f"Problem size: {A.shape[0]} products, {A.shape[1]} retailers")
    print(f"Target allocation: {b}")
    print()
    
    # Test available solvers
    results = []
    
    for solver_name, solver in available_solvers:
        print(f"Testing {solver_name}...")
        try:
            start_time = time.time()
            solution, solve_time = solver.solve_market_split(A, b, time_limit=60)
            total_time = time.time() - start_time
            
            # Check if this is a fallback solution
            is_fallback = solution.get('fallback', False)
            
            # Verify solution only if not a fallback
            if not is_fallback:
                is_valid, message = verify_solution(A, b, solution)
            else:
                is_valid = False
                message = f"Fallback solution (original error: {solution.get('error', 'Unknown')})"
            
            results.append({
                'solver': solver_name,
                'success': not is_fallback,
                'solve_time': solve_time,
                'total_time': total_time,
                'slack_total': solution.get('slack_total', float('inf')),
                'valid': is_valid,
                'message': message,
                'fallback': is_fallback
            })
            
            status = "✓ Success" if not is_fallback else "⚠ Fallback"
            print(f"  {status} in {solve_time:.3f}s")
            print(f"  Slack total: {solution.get('slack_total', 'N/A')}")
            print(f"  Valid: {is_valid} ({message})")
            
        except Exception as e:
            results.append({
                'solver': solver_name,
                'success': False,
                'error': str(e),
                'solve_time': float('inf'),
                'total_time': float('inf'),
                'slack_total': float('inf'),
                'valid': False,
                'message': "Solver failed",
                'fallback': False
            })
            
            print(f"  ✗ Failed: {e}")
        
        print()
    
    # Summary
    print("SUMMARY")
    print("-" * 50)
    print(f"{'Solver':<20} {'Status':<10} {'Time (s)':<10} {'Slack':<10} {'Valid':<8}")
    print("-" * 50)
    
    for result in results:
        if result['fallback']:
            status = "Fallback"
        elif result['success']:
            status = "Success"
        else:
            status = "Failed"
            
        time_str = f"{result['solve_time']:.3f}" if result['solve_time'] != float('inf') else "N/A"
        slack_str = f"{result['slack_total']:.3f}" if result['slack_total'] != float('inf') else "N/A"
        valid_str = "Yes" if result['valid'] else "No"
        
        print(f"{result['solver']:<20} {status:<10} {time_str:<10} {slack_str:<10} {valid_str:<8}")
    
    return results

def compare_solvers():
    """Compare performance of available solvers on multiple instances"""
    print("\nComparing available solvers on multiple instances...")
    print("=" * 60)
    
    # Get available solvers
    available_solvers, unavailable_solvers = get_available_solvers()
    
    # Filter to only classical solvers that should work well
    classical_solvers = [(name, solver) for name, solver in available_solvers 
                        if 'Lattice' in name or 'OR-Tools' in name or 'Pyomo' in name]
    
    if not classical_solvers:
        print("No classical solvers available for comparison.")
        return
    
    print(f"Comparing {len(classical_solvers)} classical solvers:")
    for name, _ in classical_solvers:
        print(f"  - {name}")
    print()
    
    # Generate multiple test instances
    instances = []
    instance_sizes = [(3, 15), (4, 20), (5, 25)]
    
    for i, (m, n) in enumerate(instance_sizes):
        A, b, true_solution = generate_test_instance(seed=42+i, m=m, n=n)
        instances.append((A, b))
        print(f"Instance {i+1}: {m}x{n} (products x retailers)")
    
    print()
    
    for solver_name, solver in classical_solvers:
        print(f"Testing {solver_name} on all instances...")
        total_time = 0
        successful = 0
        
        for i, (A, b) in enumerate(instances):
            try:
                start_time = time.time()
                solution, solve_time = solver.solve_market_split(A, b, time_limit=30)
                total_time += solve_time
                
                # Check if it's a valid solution (not fallback)
                if not solution.get('fallback', False):
                    is_valid, _ = verify_solution(A, b, solution)
                    if is_valid:
                        successful += 1
                    status = '✓' if is_valid else '✗'
                else:
                    status = '⚠'
                    
                print(f"  Instance {i+1}: {solve_time:.3f}s - {status}")
                
            except Exception as e:
                print(f"  Instance {i+1}: Failed - {e}")
        
        if successful > 0:
            avg_time = total_time/successful
            print(f"  Summary: {successful}/{len(instances)} successful, avg time: {avg_time:.3f}s")
        else:
            print(f"  Summary: 0/{len(instances)} successful")
        print()

def show_solver_info():
    """Show detailed information about solver availability"""
    print("\nSOLVER AVAILABILITY DETAILS")
    print("=" * 40)
    
    # Lattice solver
    try:
        from lattice_solver import LatticeBasedSolver
        print("✓ Lattice-Based Solver:")
        print("  - Uses fpylll for lattice reduction")
        print("  - Transforms MSP to Shortest Vector Problem")
        print("  - Very fast for small problems")
        print()
    except ImportError as e:
        print(f"✗ Lattice-Based Solver: {e}")
        print()
    
    # OR-Tools solver
    try:
        from ortools_solver import ORToolsMarketSplitSolver
        info = ORToolsMarketSplitSolver.get_availability_info()
        if info['available']:
            print("✓ OR-Tools CP-SAT Solver:")
            print("  - Uses Google OR-Tools constraint programming")
            print("  - Fast for constraint satisfaction problems")
            print("  - Requires Python < 3.14")
        else:
            print("✗ OR-Tools CP-SAT Solver:")
            print(f"  - {info['message']}")
            print(f"  - {info.get('installation_hint', '')}")
        print()
    except ImportError as e:
        print(f"✗ OR-Tools CP-SAT Solver: {e}")
        print()
    
    # Pyomo solver
    try:
        from pyomo_solver import PyomoMarketSplitSolver
        info = PyomoMarketSplitSolver.get_availability_info()
        if info['available']:
            print("✓ Pyomo + Gurobi Solver:")
            print("  - Uses Pyomo optimization modeling")
            print("  - Requires commercial Gurobi solver")
            print("  - Provides optimal solutions")
        else:
            print("✗ Pyomo + Gurobi Solver:")
            print(f"  - {info['message']}")
            print(f"  - {info.get('installation_hint', '')}")
        print()
    except ImportError as e:
        print(f"✗ Pyomo + Gurobi Solver: {e}")
        print()
    
    # Quantum solvers
    try:
        from dwave_solver import DWaveMarketSplitSolver
        print("✓ D-Wave Quantum Solver:")
        print("  - Uses D-Wave quantum annealer")
        print("  - Requires D-Wave API token for real hardware")
        print("  - Falls back to simulated annealing")
        print()
    except ImportError as e:
        print(f"✗ D-Wave Quantum Solver: {e}")
        print()
    
    try:
        from qiskit_solver import QiskitMarketSplitSolver
        print("✓ Qiskit Quantum Solvers:")
        print("  - VQE (Variational Quantum Eigensolver)")
        print("  - QAOA (Quantum Approximate Optimization)")
        print("  - Runs on simulators or IBM quantum hardware")
        print()
    except ImportError as e:
        print(f"✗ Qiskit Quantum Solvers: {e}")
        print()

if __name__ == "__main__":
    # Show detailed solver information
    show_solver_info()
    
    # Run the main example
    results = run_example()
    
    # Run comparison if we have multiple working solvers
    if len([r for r in results if r['success']]) > 1:
        compare_solvers()
    
    print("\nExample usage completed!")
    print("\nRecommendations:")
    print("- For best results, use Python 3.11 or 3.12")
    print("- Install missing dependencies as shown above")
    print("- The lattice-based solver works with current Python 3.14 setup")
