"""Minimal file-based job queue so two GPUs never run the same job twice.

jobs.txt: one shell command per line. Lines starting with '#' are ignored.
Each worker atomically claims the next unclaimed line, runs it on its GPU,
and records the outcome in jobs.state (claimed / done / failed).

    # terminal 1 (training queue)
    python jobqueue.py --gpu 0 --jobs train_jobs.txt
    # terminal 2 (eval queue; falls back to training jobs when eval is empty)
    python jobqueue.py --gpu 1 --jobs eval_jobs.txt --fallback train_jobs.txt

Append new lines to a jobs file at any time; workers pick them up.
Run inside tmux so jobs survive disconnects. Logs: logs/<jobfile>_<line>.log
"""
import argparse
import fcntl
import os
import subprocess
import time


def claim(jobs_path, gpu):
    state_path = jobs_path + ".state"
    with open(state_path, "a+") as st:
        fcntl.flock(st, fcntl.LOCK_EX)
        st.seek(0)
        claimed = {l.split("\t")[0] for l in st.read().splitlines() if l}
        with open(jobs_path) as f:
            lines = f.read().splitlines()
        for i, cmd in enumerate(lines):
            key = f"{i}"
            if cmd.strip() and not cmd.lstrip().startswith("#") and key not in claimed:
                st.write(f"{key}\tclaimed\tgpu{gpu}\t{time.strftime('%m-%d %H:%M')}\t{cmd}\n")
                st.flush()
                fcntl.flock(st, fcntl.LOCK_UN)
                return i, cmd
        fcntl.flock(st, fcntl.LOCK_UN)
    return None, None


def mark(jobs_path, i, status, gpu, dt):
    with open(jobs_path + ".state", "a") as st:
        fcntl.flock(st, fcntl.LOCK_EX)
        st.write(f"{i}-{status}\t{status}\tgpu{gpu}\t{time.strftime('%m-%d %H:%M')}\t{dt / 60:.1f} min\n")
        fcntl.flock(st, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--fallback", default=None)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))

    while True:
        src, (i, cmd) = args.jobs, claim(args.jobs, args.gpu)
        if cmd is None and args.fallback:
            src, (i, cmd) = args.fallback, claim(args.fallback, args.gpu)
        if cmd is None:
            time.sleep(args.poll)
            continue
        log = f"logs/{os.path.basename(src)}_{i}.log"
        print(f"[gpu{args.gpu}] start {src}:{i}  {cmd}")
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.call(cmd, shell=True, env=env, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        mark(src, i, "done" if rc == 0 else "failed", args.gpu, dt)
        print(f"[gpu{args.gpu}] {'done' if rc == 0 else 'FAILED'} {src}:{i} in {dt / 60:.1f} min (log {log})")


if __name__ == "__main__":
    main()
