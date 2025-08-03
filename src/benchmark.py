import os
import re
import subprocess
import shutil

def extract_cpp_code(text):
    """Parses the LLM's response to find and extract the C++ code block."""
    match = re.search(r'```(?:cpp|cuda)\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        print("Warning: Could not find a valid C++ code block (```cpp or ```cuda) in the response.")
        return None

def clean_generated_code(code_content):
    """ 
    Removes boilerplate from the LLM's generated code that would conflict
    with the benchmark template.
    """
    if not code_content:
        return ""
    
    cleaned_code = re.sub(r'#include\s*<.*?>\n', '', code_content)
    cleaned_code = re.sub(r'int\s+main\s*\(.*\)\s*\{[\s\S]*\}', '', cleaned_code)
    
    # Remove CUDA_CHECK and CUBLAS_CHECK macro definitions to avoid redefinition
    cleaned_code = re.sub(r'#define\s+CUDA_CHECK\([\s\S]*?\}\s*while\s*\(0\)', '', cleaned_code)
    cleaned_code = re.sub(r'#define\s+CUBLAS_CHECK\([\s\S]*?\}\s*while\s*\(0\)', '', cleaned_code)
    
    return cleaned_code.strip()

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
    
    kernel_code = clean_generated_code(kernel_code)
    
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

    print("\nSearching for and removing conflicting CUDA toolkits...")
    conflicting_cuda_path = "/usr/local/cuda-12.5"
    if os.path.exists(conflicting_cuda_path):
        print(f"Found conflicting toolkit at {conflicting_cuda_path}. Removing it...")
        shutil.rmtree(conflicting_cuda_path)
        print("Conflicting toolkit removed.")
    else:
        print("No conflicting toolkits found.")

    compile_command = (
        f"/usr/local/cuda-12.4/bin/nvcc {file_name} "
        f"-o {executable_name} "
        f"-gencode arch=compute_75,code=sm_75 "
        f"--cudart static -lcublas_static -lcublasLt_static -lculibos"
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
