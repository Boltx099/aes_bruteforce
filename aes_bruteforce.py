#!/usr/bin/env python3
"""
aes_bruteforce_mp.py

Multiprocessing/resumable brute-force for AES-Crypt .aes files using pyAesCrypt.
Usage:
  pip3 install pyAesCrypt
  python3 aes_bruteforce_mp.py --aes backup.aes --wordlist /usr/share/wordlists/rockyou.txt --out recovered_archive --workers 6 --checkpoint .aes_ckpt

Notes:
 - Make sure you have enough disk space for temporary files.
 - On success the decrypted file will be moved to the path you pass in --out
"""

import pyAesCrypt
import argparse
import os
import tempfile
import zipfile
import tarfile
import sys
from multiprocessing import Pool, Manager
import itertools
import signal
import time

BUFFER_SIZE = 64 * 1024  # pyAesCrypt buffer

# Globals set in initializer
_shared = None
_aes_file = None
_out_name = None

def looks_like_archive(path):
    try:
        if os.path.getsize(path) < 10:
            return False, None
        if zipfile.is_zipfile(path):
            return True, "zip"
        if tarfile.is_tarfile(path):
            return True, "tar"
    except Exception:
        pass
    # heuristics: not zero-length -> might be valid; caller must verify later
    try:
        if os.path.getsize(path) > 100:
            return True, "unknown"
    except Exception:
        pass
    return False, None

def worker_try(pw):
    """Worker: attempt decrypt using pw (string). Returns tuple (ok, pw, out_tmp, kind_or_error)"""
    global _shared, _aes_file, _out_name
    if _shared['found'].is_set():
        return (False, pw, None, "stopped")
    pw = pw.rstrip("\n")
    # create unique temp file per attempt
    fd, tmpname = tempfile.mkstemp(prefix="aes_try_", dir=".")
    os.close(fd)
    try:
        # pyAesCrypt.decryptFile raises on wrong password / integrity fail
        pyAesCrypt.decryptFile(_aes_file, tmpname, pw, BUFFER_SIZE)
        ok, kind = looks_like_archive(tmpname)
        if ok:
            # move to final out path atomically (if exists, add suffix)
            final_out = _out_name
            try:
                if os.path.exists(final_out):
                    base = final_out + ".found"
                    os.replace(tmpname, base)
                    final_out = base
                else:
                    os.replace(tmpname, final_out)
                _shared['found_pw'].value = pw
                _shared['found'].set()
                return (True, pw, final_out, kind)
            except Exception as e:
                # if move failed, leave tmp around
                return (True, pw, tmpname, f"moved_failed:{e}")
        else:
            os.remove(tmpname)
            return (False, pw, None, None)
    except Exception as e:
        # common: ValueError or IntegrityError for wrong password
        try:
            if os.path.exists(tmpname):
                os.remove(tmpname)
        except Exception:
            pass
        return (False, pw, None, str(e))

def init_worker(shared, aes_file, out_name):
    global _shared, _aes_file, _out_name
    _shared = shared
    _aes_file = aes_file
    _out_name = out_name
    # ignore SIGINT in worker; parent handles it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def chunked_iterator(iterator, chunk_size=1000):
    """Yield lists of chunk_size from iterator"""
    it = iter(iterator)
    while True:
        chunk = list(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

def main():
    parser = argparse.ArgumentParser(description="AES-Crypt .aes brute-force with multiprocessing and resume")
    parser.add_argument("--aes", required=True, help="path to .aes backup file")
    parser.add_argument("--wordlist", required=True, help="path to wordlist (rockyou)")
    parser.add_argument("--out", default="recovered_archive", help="path to write recovered archive on success")
    parser.add_argument("--workers", type=int, default=4, help="number of worker processes")
    parser.add_argument("--checkpoint", default=".aes_bruteforce.ckpt", help="checkpoint file (stores last line offset)")
    parser.add_argument("--chunk", type=int, default=500, help="how many passwords to send to pool at once")
    args = parser.parse_args()

    if not os.path.isfile(args.aes):
        print("AES file not found:", args.aes); sys.exit(1)
    if not os.path.isfile(args.wordlist):
        print("Wordlist not found:", args.wordlist); sys.exit(1)

    manager = Manager()
    shared = {
        'found': manager.Event(),
        'found_pw': manager.Value('u', "")  # unicode string (may be empty)
    }

    # read checkpoint: store byte offset (we'll use line count instead)
    start_line = 0
    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r") as ck:
                start_line = int(ck.read().strip() or "0")
            print(f"[+] Resuming from line {start_line}")
        except Exception:
            start_line = 0

    pool = Pool(processes=args.workers, initializer=init_worker, initargs=(shared, args.aes, args.out))

    try:
        with open(args.wordlist, "r", errors="ignore") as wl:
            # skip lines up to start_line
            for _ in range(start_line):
                if not wl.readline():
                    break
            line_no = start_line
            batch_size = args.chunk
            for chunk in chunked_iterator(wl, chunk_size=batch_size):
                if shared['found'].is_set():
                    break
                # submit chunk to pool using imap_unordered for results as they come
                results = pool.map_async(worker_try, chunk)
                # wait for completion of this chunk but check for early found
                while not results.ready():
                    if shared['found'].is_set():
                        # attempt to cancel (workers may still be decrypting)
                        try:
                            pool.terminate()
                        except Exception:
                            pass
                        break
                    time.sleep(0.2)
                # process results quickly
                try:
                    outs = results.get(timeout=0.1)
                except Exception:
                    outs = []
                for ok, pw, outfile, info in outs:
                    line_no += 1
                    # update checkpoint every result (approx)
                    try:
                        with open(args.checkpoint, "w") as ck:
                            ck.write(str(line_no))
                    except Exception:
                        pass
                    if ok:
                        print(f"[+] FOUND password: {pw}")
                        print(f"[+] Decrypted file: {outfile} (kind: {info})")
                        shared['found'].set()
                        shared['found_pw'].value = pw
                        break
                if shared['found'].is_set():
                    break
            # done reading
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        # close/terminate pool
        try:
            pool.close()
            pool.join(timeout=1)
        except Exception:
            pool.terminate()
            pool.join()

    if shared['found'].is_set():
        print("[*] Success. Password:", shared['found_pw'].value)
        print("[*] Remove checkpoint if you want to restart from top.")
    else:
        print("[-] No password found in provided wordlist (or stopped). Check checkpoint for resume line.")

if __name__ == "__main__":
    main()
