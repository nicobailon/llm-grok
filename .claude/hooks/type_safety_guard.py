#!/usr/bin/env python3
"""
Type Safety Guard Hook - Detects and blocks 'any' type usage
Enforces CLAUDE.md type safety guidelines
"""

import json
import sys
import re
from typing import Dict, Any

def detect_any_usage(content: str) -> bool:
    """
    Detect 'any' type usage in code content
    Returns True if 'any' type is found
    """
    # Pattern to match 'any' as a type annotation
    patterns = [
        r'\b(?:typing\.)?[Aa]ny\b',  # any, Any, typing.any, typing.Any
        r':\s*any\b',               # : any (type annotation)
        r'->\s*any\b',              # -> any (return type)
        r'\[\s*any\s*\]',           # [any] (generic type)
    ]
    
    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    
    return False

def main():
    """Main hook execution"""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        
        # Only check Edit, Write, MultiEdit operations
        if tool_name not in ["Edit", "Write", "MultiEdit"]:
            return
            
        # Extract content based on tool type
        content = ""
        if tool_name == "Write":
            content = tool_input.get("content", "")
        elif tool_name == "Edit":
            content = tool_input.get("new_string", "")
        elif tool_name == "MultiEdit":
            edits = tool_input.get("edits", [])
            content = " ".join([edit.get("new_string", "") for edit in edits])
        
        # Check for 'any' usage
        if detect_any_usage(content):
            print("🚨 WARNING: CLAUDE.md VIOLATION - Detected Any/any usage!", file=sys.stderr)
            print("Type widening is FORBIDDEN. Use precise types (Protocol, TypeGuard, Union, Literal, etc.) instead.", file=sys.stderr)
            sys.exit(2)  # Exit with error code to block operation
            
    except Exception as e:
        print(f"Type safety guard error: {e}", file=sys.stderr)
        # Don't block operation on script errors
        sys.exit(0)

if __name__ == "__main__":
    main()