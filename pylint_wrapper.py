#!/usr/bin/env python3
import sys
import subprocess
import re

def preprocess_notes(notes_str):
    """Convert punctuation-only note tags to alphanumeric equivalents"""
    notes = notes_str.split(',')
    processed_notes = []
    
    for note in notes:
        if note and not any(c.isalnum() for c in note):
            # Replace punctuation-only tags with 'PUNC_' prefix
            processed_notes.append(f'PUNC_{note}')
        else:
            processed_notes.append(note)
    
    return ','.join(processed_notes)

def main():
    args = sys.argv[1:]
    
    try:
        notes_index = args.index('--notes')
        notes_value = args[notes_index + 1]
        args[notes_index + 1] = preprocess_notes(notes_value)
    except ValueError:
        pass  # --notes not found in arguments
    
    # Also handle --notes=value format
    for i, arg in enumerate(args):
        if arg.startswith('--notes='):
            _, notes_value = arg.split('=', 1)
            args[i] = f'--notes={preprocess_notes(notes_value)}'
    
    # Run pylint with modified arguments
    result = subprocess.run(['pylint'] + args, check=False)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()