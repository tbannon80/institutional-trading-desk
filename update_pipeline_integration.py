import subprocess
import sys
import os

def run_full_autonomous_cycle():
    print("[*] Starting fully autonomous pipeline sync cycle...")
    
    # 1. Run unit test gating
    test_result = subprocess.run(["python3", "-m", "unittest", "discover", "-s", "tests/"], capture_output=True, text=True)
    if test_result.returncode != 0:
        print("[-] TEST GATE FAILED: Aborting autonomous update.")
        print(test_result.stderr)
        sys.exit(1)
    print("[+] Test gate passed successfully.")
    
    # 2. Trigger post-deployment watcher guard
    watcher_result = subprocess.run(["bash", "watch_daemon.sh"], capture_output=True, text=True)
    if watcher_result.returncode != 0:
        print("[-] WATCHER GUARD TRIGGERED ROLLBACK. Autonomous deployment reverted safely.")
        print(watcher_result.stdout)
        sys.exit(1)
        
    print("[+] Fully autonomous sync cycle completed with zero exceptions. State locked.")

if __name__ == "__main__":
    run_full_autonomous_cycle()
