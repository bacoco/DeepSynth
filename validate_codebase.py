#!/usr/bin/env python3
"""
Wrapper entry point to run the codebase validator from tools/.
Keeps backwards compatibility with docs/Makefile calling this at repo root.
"""

from tools.validate_codebase import main

if __name__ == "__main__":
    main()

