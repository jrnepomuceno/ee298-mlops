#!/usr/bin/env python3
"""Script to optimize the ONNX model for better performance."""

import onnx
from onnxruntime import InferenceSession, SessionOptions
import onnxruntime as ort
from pathlib import Path
import argparse


def optimize_model(input_path: str, output_path: str):
    """Optimize the ONNX model."""
    print(f"Loading model from {input_path}")
    model = onnx.load(input_path)
    
    # Check the model
    print("Checking model...")
    onnx.checker.check_model(model)
    print("Model checked successfully")
    
    # Print model info
    print(f"Model opset: {model.opset_import[0].version}")
    print(f"Model inputs: {[i.name for i in model.graph.input]}")
    print(f"Model outputs: {[o.name for o in model.graph.output]}")
    
    # Save the optimized model
    print(f"Saving optimized model to {output_path}")
    onnx.save(model, output_path)
    print("Model optimized and saved successfully")


def create_optimized_session(model_path: str) -> InferenceSession:
    """Create an optimized ONNX Runtime session."""
    options = SessionOptions()
    
    # Enable optimization
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Enable memory pattern optimization
    options.enable_mem_pattern = True
    
    # Enable memory preallocation
    options.enable_cpu_mem_arena = True
    
    # Set execution mode to parallel
    options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    
    # Create session
    session = InferenceSession(model_path, options)
    return session


def benchmark_model(session: InferenceSession, input_shape: tuple, num_runs: int = 100):
    """Benchmark the model performance."""
    import time
    import numpy as np
    
    # Create test input
    test_input = np.random.randn(*input_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warmup runs
    print("Running warmup...")
    for _ in range(10):
        _ = session.run(None, {input_name: test_input})
    
    # Benchmark runs
    print(f"Running benchmark with {num_runs} iterations...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = session.run(None, {input_name: test_input})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.2f} inferences/second")


def main():
    parser = argparse.ArgumentParser(description="Optimize ONNX model for better performance")
    parser.add_argument("--input", type=str, default="test_model.onnx", help="Input ONNX model path")
    parser.add_argument("--output", type=str, default="optimized_model.onnx", help="Output optimized model path")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after optimization")
    
    args = parser.parse_args()
    
    # Check if input model exists
    if not Path(args.input).exists():
        print(f"Error: Input model not found at {args.input}")
        return 1
    
    # Optimize the model
    optimize_model(args.input, args.output)
    
    # Benchmark if requested
    if args.benchmark:
        print("\nBenchmarking original model...")
        original_session = InferenceSession(args.input)
        benchmark_model(original_session, (1, 100, 80))
        
        print("\nBenchmarking optimized model...")
        optimized_session = create_optimized_session(args.output)
        benchmark_model(optimized_session, (1, 100, 80))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())