# Market Split Problem - Benchmarking Framework
# Comprehensive comparison of classical, lattice-based, and quantum optimization approaches

import time
import numpy as np
from typing import Dict, List, Tuple, Callable
import json

# Try to import all solvers with graceful fallbacks
def import_solvers():
    """Import all available solvers, handling missing dependencies"""
    solvers = {}
    import_errors = {}
    
    # Import lattice solver (should always work)
    try:
        from lattice_solver import LatticeBasedSolver
        solvers['Lattice_Based'] = LatticeBasedSolver()
    except ImportError as e:
        import_errors['Lattice_Based'] = str(e)
    
    # Import D-Wave solver (graceful fallback)
    try:
        from dwave_solver import DWaveMarketSplitSolver
        solvers['D-Wave_SA'] = DWaveMarketSplitSolver()
    except ImportError as e:
        import_errors['D-Wave_SA'] = str(e)
    
    # Import Qiskit solver (graceful fallback)
    try:
        from qiskit_solver import QiskitMarketSplitSolver
        solvers['VQE'] = QiskitMarketSplitSolver(method='vqe')
        solvers['QAOA'] = QiskitMarketSplitSolver(method='qaoa')
    except ImportError as e:
        import_errors['Qiskit_Solvers'] = str(e)
    
    # Import OR-Tools solver (graceful fallback)
    try:
        from ortools_solver import ORToolsMarketSplitSolver
        if ORToolsMarketSplitSolver.get_availability_info()['available']:
            solvers['OR-Tools'] = ORToolsMarketSplitSolver()
        else:
            import_errors['OR-Tools'] = ORToolsMarketSplitSolver.get_availability_info()['message']
    except ImportError as e:
        import_errors['OR-Tools'] = str(e)
    
    # Import Pyomo solver (graceful fallback)
    try:
        from pyomo_solver import PyomoMarketSplitSolver
        if PyomoMarketSplitSolver.get_availability_info()['available']:
            solvers['Pyomo_Gurobi'] = PyomoMarketSplitSolver()
        else:
            import_errors['Pyomo_Gurobi'] = PyomoMarketSplitSolver.get_availability_info()['message']
    except ImportError as e:
        import_errors['Pyomo_Gurobi'] = str(e)
    
    # Import PuLP solver (graceful fallback)
    try:
        from pulp_solver import PuLPMarketSplitSolver
        if PuLPMarketSplitSolver.get_availability_info()['available']:
            solvers['PuLP_CBC'] = PuLPMarketSplitSolver()
        else:
            import_errors['PuLP_CBC'] = PuLPMarketSplitSolver.get_availability_info()['message']
    except ImportError as e:
        import_errors['PuLP_CBC'] = str(e)
    
    return solvers, import_errors

class MarketSplitBenchmark:
    """
    Comprehensive benchmarking framework for Market Split Problem solvers
    
    Compares performance across:
    - Classical Optimization (Pyomo/Gurobi, OR-Tools CP-SAT)
    - Lattice-Based Methods (solvediophant with LLL/BKZ)
    - Quantum Optimization (D-Wave, Qiskit VQE/QAOA)
    
    Gracefully handles missing dependencies
    """
    
    def __init__(self):
        self.results = {}
        self.solvers, self.import_errors = import_solvers()
        
        if self.import_errors:
            print("Warning: Some solvers are not available:")
            for solver_name, error in self.import_errors.items():
                print(f"  - {solver_name}: {error}")
            print()
    
    def run_solver(self, solver_name: str, solver: Callable, A: np.ndarray, b: np.ndarray, 
                   time_limit: float = 60.0) -> Dict:
        """
        Run a single solver on the given problem instance
        
        Args:
            solver_name: Name of the solver
            solver: Solver function/class
            A: Problem matrix (m x n)
            b: Target vector (m,)
            time_limit: Time limit in seconds
            
        Returns:
            Dictionary with results and timing information
        """
        try:
            start_time = time.time()
            
            if hasattr(solver, 'solve_market_split'):
                # Solver is a class instance
                result, solve_time = solver.solve_market_split(A, b, time_limit=time_limit)
            else:
                # Solver is a function
                result = solver(A, b)
                solve_time = time.time() - start_time
            
            total_time = time.time() - start_time
            
            # Extract slack_total, handling different result formats
            slack_total = result.get('slack_total', float('inf'))
            if 'fallback' in result and result['fallback']:
                success = False
            else:
                success = True
            
            return {
                'success': success,
                'solution': result,
                'solve_time': solve_time,
                'total_time': total_time,
                'slack_total': slack_total,
                'error': result.get('error', None)
            }
            
        except Exception as e:
            return {
                'success': False,
                'solution': {'x': [0] * A.shape[1], 'slack_total': float('inf')},
                'solve_time': float('inf'),
                'total_time': float('inf'),
                'slack_total': float('inf'),
                'error': str(e)
            }
    
    def run_benchmark(self, instances: List[Tuple[np.ndarray, np.ndarray]], 
                     time_limit: float = 60.0, verbose: bool = True) -> Dict:
        """
        Run benchmark across all solvers and instances
        
        Args:
            instances: List of (A, b) tuples representing problem instances
            time_limit: Time limit for each solver run
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing benchmark results
        """
        if not self.solvers:
            print("Error: No solvers are available for benchmarking!")
            return {}
        
        benchmark_results = {}
        
        if verbose:
            print(f"Starting benchmark with {len(instances)} instances and {len(self.solvers)} solvers")
            print(f"Available solvers: {list(self.solvers.keys())}")
            print(f"Time limit per solver: {time_limit}s")
            if self.import_errors:
                print(f"Unavailable solvers: {list(self.import_errors.keys())}")
            print("-" * 60)
        
        for solver_name, solver in self.solvers.items():
            if verbose:
                print(f"Testing solver: {solver_name}")
            
            solver_results = {
                'instances_tested': 0,
                'successful_solutions': 0,
                'total_solve_time': 0.0,
                'avg_solve_time': 0.0,
                'avg_slack_total': 0.0,
                'success_rate': 0.0,
                'individual_results': []
            }
            
            for i, (A, b) in enumerate(instances):
                if verbose:
                    print(f"  Instance {i+1}/{len(instances)} (shape: {A.shape})", end=" ")
                
                result = self.run_solver(solver_name, solver, A, b, time_limit)
                
                solver_results['instances_tested'] += 1
                solver_results['individual_results'].append(result)
                
                if result['success']:
                    solver_results['successful_solutions'] += 1
                    solver_results['total_solve_time'] += result['solve_time']
                
                if verbose:
                    status = "✓" if result['success'] else "✗"
                    slack = result['slack_total'] if result['slack_total'] != float('inf') else 'inf'
                    error_note = f" ({result['error']})" if result['error'] else ""
                    print(f"- Time: {result['solve_time']:.3f}s, Slack: {slack} {status}{error_note}")
            
            # Calculate aggregate statistics
            if solver_results['successful_solutions'] > 0:
                solver_results['avg_solve_time'] = (solver_results['total_solve_time'] / 
                                                  solver_results['successful_solutions'])
            
            solver_results['success_rate'] = (solver_results['successful_solutions'] / 
                                            solver_results['instances_tested'])
            
            # Calculate average slack for successful solutions
            successful_slacks = [r['slack_total'] for r in solver_results['individual_results'] 
                               if r['success'] and r['slack_total'] != float('inf')]
            if successful_slacks:
                solver_results['avg_slack_total'] = np.mean(successful_slacks)
            else:
                solver_results['avg_slack_total'] = float('inf')
            
            benchmark_results[solver_name] = solver_results
            
            if verbose:
                print(f"  Summary: {solver_results['successful_solutions']}/{solver_results['instances_tested']} "
                      f"success ({solver_results['success_rate']:.1%}), "
                      f"avg time: {solver_results['avg_solve_time']:.3f}s")
                print()
        
        self.results = benchmark_results
        return benchmark_results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive benchmark report
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No benchmark results available. Run benchmark first."
        
        report = []
        report.append("=" * 80)
        report.append("MARKET SPLIT PROBLEM - BENCHMARK RESULTS")
        report.append("=" * 80)
        report.append("")
        
        # Summary table
        report.append("SOLVER PERFORMANCE SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Solver':<15} {'Success Rate':<12} {'Avg Time (s)':<12} {'Avg Slack':<12}")
        report.append("-" * 80)
        
        for solver_name, results in self.results.items():
            success_rate = f"{results['success_rate']:.1%}"
            avg_time = f"{results['avg_solve_time']:.3f}" if results['avg_solve_time'] != float('inf') else "N/A"
            avg_slack = f"{results['avg_slack_total']:.3f}" if results['avg_slack_total'] != float('inf') else "N/A"
            
            report.append(f"{solver_name:<15} {success_rate:<12} {avg_time:<12} {avg_slack:<12}")
        
        report.append("")
        
        # List unavailable solvers
        if self.import_errors:
            report.append("UNAVAILABLE SOLVERS")
            report.append("-" * 40)
            for solver_name, error in self.import_errors.items():
                report.append(f"• {solver_name}: {error}")
            report.append("")
        
        # Detailed analysis
        report.append("DETAILED ANALYSIS")
        report.append("-" * 40)
        
        # Find best performing solver
        best_solver = None
        best_score = float('inf')
        
        for solver_name, results in self.results.items():
            if results['success_rate'] > 0:
                # Score based on success rate (higher is better) and time (lower is better)
                if results['avg_solve_time'] != float('inf'):
                    score = (1 - results['success_rate']) * 100 + results['avg_solve_time']
                    if score < best_score:
                        best_score = score
                        best_solver = solver_name
        
        if best_solver:
            report.append(f"Best Overall Performance: {best_solver}")
            report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS:")
        report.append("• Classical MIP solvers (Pyomo/Gurobi): Scalable with optimal guarantees")
        report.append("  but exponential time on hard instances")
        report.append("• Lattice-based methods: Extremely fast for small problems,")
        report.append("  transforms MSP to Shortest Vector Problem")
        report.append("• Quantum approaches: Promising potential but limited by")
        report.append("  qubit count and noise on current hardware")
        report.append("• OR-Tools CP-SAT: Often fastest for constraint satisfaction")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """
        Save benchmark results to JSON file with proper NumPy type handling
        
        Args:
            filename: Output filename
        """
        
        def convert_to_json_serializable(obj):
            """Convert numpy types and complex objects to JSON-serializable format"""
            if obj is None:
                return None
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float)):
                return obj
            else:
                # Fallback for unknown types
                return str(obj)
        
        # Create serializable results
        serializable_results = {}
        
        for solver_name, results in self.results.items():
            solver_data = {
                'instances_tested': int(convert_to_json_serializable(results['instances_tested'])),
                'successful_solutions': int(convert_to_json_serializable(results['successful_solutions'])),
                'total_solve_time': float(convert_to_json_serializable(results['total_solve_time'])),
                'avg_solve_time': float(convert_to_json_serializable(results['avg_solve_time'])) if results['avg_solve_time'] != float('inf') else None,
                'avg_slack_total': float(convert_to_json_serializable(results['avg_slack_total'])) if results['avg_slack_total'] != float('inf') else None,
                'success_rate': float(convert_to_json_serializable(results['success_rate'])),
                'individual_results': []
            }
            
            # Process individual results
            for result in results['individual_results']:
                individual_result = {
                    'success': bool(convert_to_json_serializable(result['success'])),
                    'solve_time': float(convert_to_json_serializable(result['solve_time'])) if result['solve_time'] != float('inf') else None,
                    'total_time': float(convert_to_json_serializable(result['total_time'])) if result['total_time'] != float('inf') else None,
                    'slack_total': float(convert_to_json_serializable(result['slack_total'])) if result['slack_total'] != float('inf') else None,
                    'error': result['error']
                }
                
                # Handle solution data
                if 'solution' in result and result['solution']:
                    sol = result['solution']
                    x_data = sol.get('x', [])
                    if hasattr(x_data, 'tolist'):
                        x_data = x_data.tolist()
                    elif isinstance(x_data, (list, tuple)):
                        x_data = [convert_to_json_serializable(val) for val in x_data]
                    
                    slack_val = sol.get('slack_total', float('inf'))
                    slack_val = float(convert_to_json_serializable(slack_val)) if slack_val != float('inf') else None
                    
                    individual_result['solution'] = {
                        'x': x_data,
                        'slack_total': slack_val
                    }
                else:
                    individual_result['solution'] = None
                
                solver_data['individual_results'].append(individual_result)
            
            serializable_results[solver_name] = solver_data
        
        # Add import errors
        serializable_results['_import_errors'] = self.import_errors
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"Benchmark results saved to {filename}")

def generate_test_instances(num_instances: int = 5, sizes: List[Tuple[int, int]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate test instances for benchmarking
    
    Args:
        num_instances: Number of instances to generate
        sizes: List of (m, n) tuples for instance sizes
        
    Returns:
        List of (A, b) tuples
    """
    if sizes is None:
        sizes = [(3, 15), (5, 20), (6, 25), (4, 18), (5, 22)]
    
    instances = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    for i in range(num_instances):
        if i < len(sizes):
            m, n = sizes[i]
        else:
            # Generate random size
            m = rng.integers(3, 8)
            n = rng.integers(15, 30)
        
        # Generate random matrix A
        A = rng.integers(1, 10, size=(m, n))
        
        # Generate true solution
        true_solution = rng.integers(0, 2, size=n)
        
        # Calculate b = A @ true_solution
        b = A @ true_solution
        
        instances.append((A, b))
    
    return instances

# Example usage and demonstration
if __name__ == "__main__":
    print("Market Split Problem - Benchmark Framework")
    print("=" * 50)
    
    # Generate test instances
    test_instances = generate_test_instances(num_instances=3)
    
    # Create benchmark
    benchmark = MarketSplitBenchmark()
    
    # Run benchmark
    results = benchmark.run_benchmark(test_instances, time_limit=30.0, verbose=True)
    
    # Generate and print report
    print(benchmark.generate_report())
    
    # Save results
    benchmark.save_results("benchmark_results.json")
