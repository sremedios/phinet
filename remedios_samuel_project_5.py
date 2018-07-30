'''
Author: Samuel Remedios
CSCI 3038

Determines which nonces produce valid hashes for a given string.

Usage:
    python3 name_of_this_file.py hash_method num_nonce_digits num_leading_zeros target_string
'''
import hashlib
from multiprocessing import Process, Queue
from os import cpu_count
import sys
from time import time

def validate_hash(hash_method, nonce, init_string):
    h = hashlib.new(hash_method)
    h.update(str.encode(nonce))
    h.update(str.encode(init_string))
    return h.hexdigest()

def worker(hash_method, nonce_range, num_leading_zeros, init_string, q):
    cur_valid_hashes = []
    for n in nonce_range:
        cur_hash = validate_hash(hash_method, n, init_string)
        if cur_hash[:num_leading_zeros] == "0" * num_leading_zeros:
            cur_valid_hashes.append((n, cur_hash))
    q.put(cur_valid_hashes)

def nonce_val_generator(start_digit, end_digit, num_nonce_digits):
    for i in range(start_digit, end_digit):
        yield str(i).zfill(num_nonce_digits)

def gen_nonces(num_nonce_digits, num_cores):
    range_end = 10 ** num_nonce_digits
    remainder = range_end % num_cores
    sublist_len = range_end // num_cores
    cur = 0

    for _ in range(num_cores - remainder):
        yield nonce_val_generator(cur, cur + sublist_len, num_nonce_digits)
        cur += sublist_len
    for _ in range(remainder):
        if cur + sublist_len + 1 <= range_end:
            yield nonce_val_generator(cur, cur + sublist_len + 1, num_nonce_digits)
        else:
            yield nonce_val_generator(cur, range_end, num_nonce_digits)
        cur += sublist_len + 1

def display_results(init_string, valid_hashes):
    print(init_string)
    for i in range(len(valid_hashes)):
        print("    {} {}".format(valid_hashes[i][0], valid_hashes[i][1]))
    print(len(valid_hashes))

if __name__ == '__main__':
    hash_method = sys.argv[1]
    num_nonce_digits = int(sys.argv[2])
    num_leading_zeros = int(sys.argv[3])
    init_string = sys.argv[4]
    num_cores = cpu_count()
    if num_cores is None:
        num_cores = 8

    nonces = gen_nonces(num_nonce_digits, num_cores)
    nonces = (x for x in nonces)

    valid_hashes = []

    #start_time = time()

    # initialize processes, queue, etc
    q = Queue()
    procs = [Process(target=worker, args=(hash_method,
                                          n,
                                          num_leading_zeros,
                                          init_string,
                                          q)) for n in nonces]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    # collect results
    for _ in range(len(procs)):
        valid_hashes.extend(q.get())

    # sort results by nonce for easy grading
    valid_hashes = sorted(valid_hashes, key=lambda x: x[0])

    display_results(init_string, valid_hashes)
    #print("Elapsed time: {:.4f}".format(time() - start_time))
