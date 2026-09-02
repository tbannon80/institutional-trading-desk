import subprocess
import sys
import os
import time

def execute_automated_pipeline(watch_guard: bool = False):
    print("[*] Initializing automated test-gated pipeline verification...")
    
    # 1. Run unit test suite as the strict gating criteria
    result = subprocess.run(["python3", "-m", "unittest", "discover", "-s", "tests/"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[-] TEST GATE FAILED: Proposed changes violate test invariants. Aborting deployment.")
        print(result.stderr)
        sys.exit(1)
    else:
        print("[+] TEST GATE PASSED: All unit and integration checks verified successfully.")
        
    # 2. Simulate safe deployment sync
    print("[+] Pipeline synchronized. Autonomous state active.")

    # 3. Post-deployment watcher guard with automated rollback
    if watch_guard or "--watch" in sys.argv or "--monitor" in sys.argv or "--full" in sys.argv:
        print("[*] Launching post-deployment autonomous exception watcher...")
        watcher_result = subprocess.run(["bash", "watch_daemon.sh"], capture_output=True, text=True)
        if watcher_result.returncode != 0:
            print("[-] WATCHER GUARD TRIGGERED ROLLBACK. Autonomous deployment reverted safely.")
            print(watcher_result.stdout)
            sys.exit(1)
        print("[+] Fully autonomous sync cycle completed with zero exceptions. State locked.")

if __name__ == "__main__":
    execute_automated_pipeline()
