import os
import re
import subprocess

def extract_cpp_code(text):
    """Parses the LLM's response to find and extract the C++ code block."""
    code_block_match = re.search(r'```(?:cpp|cuda)\n(.*?)```', text, re.DOTALL)
    if not code_block_match:
        print("Warning: Could not find a C++ or CUDA code block in the response.")
        return None
    
    code_content = code_block_match.group(1)
    
    start_match = re.search(r'__global__ void', code_content)
    if not start_match:
        print("Warning: Could not find a __global__ function in the code block.")
        return None
        
    code_from_kernel = code_content[start_match.start():]
    
    return code_from_kernel

def create_benchmark_harness(kernel_code):
    """
    Reads the C++ template file, injects the AI-generated kernel,
    and returns the complete, ready-to-compile C++ code as a string.
    """
    template_path = os.path.join('cpp_harness', 'benchmark_template.cpp')
    
    try:
        with open(template_path, 'r') as f:
            template_code = f.read()
    except FileNotFoundError:
        print(f"Error: The template file was not found at {template_path}")
        return None
    
    kernel_name_match = re.search(r'__global__ void\s+(\w+)\(', kernel_code)
    if not kernel_name_match:
        print("Warning: Could not find a __global__ function to rename.")
        return None

    full_code = template_code.replace('// {{GENERATED_KERNEL_CODE}}', kernel_code)
    full_code = full_code.replace('{kernel_name}', kernel_name_match.group(1))    
    return full_code


def compile_and_run(cpp_code, file_name):
    """
    Takes a string of C++ code, writes it to a file, compiles it with nvcc,
    and runs the resulting executable, printing the output.
    """
    if not cpp_code:
        print(f"Skipping compilation for {file_name} due to missing code.")
        return
    
    with open(file_name, "w") as f:
        f.write(cpp_code)

    executable_name = file_name.replace('.cu', '_exec')

    compile_command = (
        f"/usr/local/cuda-12.4/bin/nvcc {file_name} "
        f"-o {executable_name} "
        f"-gencode arch=compute_75,code=sm_75 "
        f"-lcublas --cudart static"
    )

    print(f"\nCompiling {file_name}...")
    compile_result = subprocess.run(
        compile_command,
        shell=True, capture_output=True, text=True
    )

    if compile_result.returncode != 0:
        print("--- COMPILATION FAILED ---")
        print("NVCC Error Output:")
        print(compile_result.stderr)
        return
    
    print(f"Compilation successful. Executable created: {executable_name}")

    print("\n--- Running Benchmark ---")
    run_result = subprocess.run(
        f"./{executable_name}",
        shell=True, capture_output=True, text=True
    )

    if run_result.returncode != 0:
        print(f"--- EXECUTION FAILED (Runtime Error) ---")
        print("Program Error Output:")
        print(run_result.stderr)
    else:
        print("Program Output:")
        print(run_result.stdout)
    return
