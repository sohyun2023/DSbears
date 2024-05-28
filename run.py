# run.py

import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import my_4
import ds3

def main():
    # Execute the main functions from your modules
    print("Running the main script")
    my_4.main()
    ds3.main()

if __name__ == "__main__":
    main()