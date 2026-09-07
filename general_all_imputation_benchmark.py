import sys
import pandas as pd
import os
import argparse
import threading
import subprocess
import ast
import multiprocessing

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

base_working_directory = r'C:\Users\m-sad\OneDrive\Pulpit\data_imputation\data'
os.chdir(base_working_directory)
data_file_path = os.path.join(base_working_directory, 'data_imputation.xlsx')


def parse_numbers(input_str):
    if ':' in input_str:
        start, end = map(int, input_str.strip('[]').split(':'))
        return list(range(start, end + 1))
    else:
        return ast.literal_eval(input_str)


def run_script(g_number, task_id, num_tasks):
    script_name = "test1_imputation_benchmark.py"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    subprocess.run(["python", script_name, str(g_number)], env=env)


def Thred(g_numbers, data):
    threads = []
    for g_number in g_numbers:
        if g_number in data['g_number'].values:
            print(f"Starting thread for g_number {g_number}")
            thread = threading.Thread(target=run_script, args=(g_number, 0, 0))
            threads.append(thread)
            thread.start()
        else:
            print(f"g_number {g_number} not found in data.xlsx")

    for thread in threads:
        thread.join()


def Mult(g_numbers, data):
    processes = []
    for g_number in g_numbers:
        if g_number in data['g_number'].values:
            print(f"Starting process for g_number {g_number}")
            process = multiprocessing.Process(target=run_script, args=(g_number, 0, 0))
            processes.append(process)
            process.start()
        else:
            print(f"g_number {g_number} not found in data.xlsx")

    for process in processes:
        process.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='Run time-series clustering scripts based on a list of numbers.')
    parser.add_argument('numbers', type=str, help='The list of g_numbers to process (e.g., "[1,2,4]")')
    parser.add_argument('mode', type=int, choices=[1, 2], help='1 = thread, 2 = multiprocessing')
    args = parser.parse_args()

    g_numbers = parse_numbers(args.numbers)
    data = pd.read_excel(data_file_path)

    if args.mode == 1:
        print("Thred")
        Thred(g_numbers, data)
    else:
        print("Mult")
        Mult(g_numbers, data)