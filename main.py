"""
100% Offline Multilingual AI Receptionist — Main Launcher
Target: Xilinx PYNQ-Z2 FPGA & Fixed-Point ALU
Languages: English, Hindi, Tamil
Zero Browser | Zero Cloud Calls | 100% Local Execution
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    print("=" * 60)
    print("  100% OFFLINE MULTILINGUAL AI RECEPTIONIST LAUNCHER")
    print("=" * 60)
    print(" Select Launch Mode (Zero Browser Needed):")
    print("  [1] Native Desktop GUI Window (Tkinter + Animated Voice Orb)")
    print("  [2] Interactive Terminal Voice Console (CLI)")
    print("  [3] Run Automated 42-Test Suite (PyTest)")
    print("  [4] Run Live 3-Language Demo")
    print("=" * 60)
    
    choice = input("Select [1-4, default 1] > ").strip()
    
    if choice == "2":
        from src.cli_receptionist import main as cli_main
        cli_main()
    elif choice == "3":
        import pytest
        pytest.main(["-v"])
    elif choice == "4":
        from tests.test_live_demo import run_live_demonstrations
        run_live_demonstrations()
    else:
        from desktop_app import main as gui_main
        gui_main()

if __name__ == "__main__":
    main()
