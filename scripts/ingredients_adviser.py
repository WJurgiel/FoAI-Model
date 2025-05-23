import argparse
import os
from termcolor import colored
import test_file
def check_file_exists(path_to_file):
    res = os.path.isfile(path_to_file)
    print(f"{path_to_file}... " + (colored("missing", 'red') if not res else colored("found", 'green')))
    return res
def run_file_checks(paths):
    print(colored("---------------RUNNING FILES CHECKS--------------", "cyan"))
    return all(check_file_exists(path) for path in paths)

def main():
    os.system('color')
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required="True", help="input file path containing ingredients")
    parser.add_argument("--output", "-o", required=False, default="output.txt", help="output file path, default ./output.txt")
    parser.add_argument("--test", "-t", nargs="?", required=False, default="test_file.py", help="test file to execute")
    args = parser.parse_args()
    
    manifest_vars = dict((key,vars(args)[key]) for key in ["input", "output", "test"])
    print(manifest_vars)

    current_dir_name = os.path.dirname(__file__)
    
    for key,value in manifest_vars.items():
        if value:
            manifest_vars[key] = os.path.join(current_dir_name,value)
    print(manifest_vars)

    if run_file_checks([manifest_vars[k] for k in ["input", "test"] if manifest_vars[k]]):
        test_file.run_model(manifest_vars)
    else:
        print(colored("not ok","red"))

if __name__=="__main__":
    main()