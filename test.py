import glob
import os
from timeit import default_timer as timer
import subprocess

files = glob.glob("in/input/*.txt")
res_path = "output.txt"
prog = "./k-clique"

print(f"running using {prog}")

start = timer()
start_sub = start
for i, f in enumerate(files):
    print(f'starting file {i}, {f}')
    subprocess.run([prog, f, "11", res_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = timer()
    ans_path = "in/output/" + f[9:]
    with open(ans_path) as ansf:
        ans = ansf.readlines()[0]
    with open(res_path) as resf:
        res = resf.readlines()[0]
    if(res != ans):
        print(f"FAIL: {i}")
    #print(f'finished in: {end-start_sub}')
    start_sub = end
print(f'total time: {end - start}')


"""

#!/usr/bin/env python3

import os
import subprocess
import sys

def main():
    dir = sys.argv[1]
    ok = 0
    cnt = 0

    for filename in os.listdir(dir + "/input"):
        test_name = filename[:-4]
        input_file = dir + "/input/" + test_name + ".txt"
        output_file = "output"
        subprocess.run(["./kcliques", input_file, "11", output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        expected_output = open(dir + "/output/" + test_name + ".txt", "r").readlines()[0].strip().split(" ")
        actual_output = open(output_file, "r").readlines()[0].strip().split(" ")
        cnt += 1
        if expected_output[:len(actual_output)] == actual_output:
            ok += 1
        else:
            print(test_name + " FAIL")
        
    print(f"Passesd {ok}/{cnt}")
        
if __name__ == "__main__":
    main()
"""