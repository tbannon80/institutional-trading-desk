import subprocess
import sys
import os
import time

def execute_automated_pipeline():
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

    # 3. Optional post-sync watcher invocation
    if "--watch" in sys.argv or "--monitor" in sys.argv:
        print("[*] Launching post-deployment autonomous exception watcher...")
        subprocess.run(["./watch_daemon.sh"], check=True)

if __name__ == "__main__":
    execute_automated_pipeline()
