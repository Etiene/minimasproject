# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# usage: python generator.py number_of_samples max_int > file.txt
import random
import sys

def create_numbers(n, max_limit):
    all = {(-1,-1): True}
    for i in range(n):
        a, b = -1, -1
        # avoiding duplicates
        # could halt forever, go play the lottery if it does
        while(all.get((a,b), None) != None):
            a = random.randint(0, max_limit)
            b = random.randint(0, max_limit)
        all[(a,b)]=True
        print(a, b, a+b)

if __name__ == "__main__":
    random.seed(a=None, version=2)
    n_samples = int(sys.argv[1])
    max_limit = int(sys.argv[2]) or 1000
    create_numbers(n_samples, max_limit)
