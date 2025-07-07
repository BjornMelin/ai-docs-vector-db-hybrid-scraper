#!/usr/bin/env python3
"""Script to fix event loop issues in async tests."""

import re
from pathlib import Path
from typing import List, Tuple


def fix_event_loop_patterns(content: str) -> Tuple[str, List[str]]:
    """Fix event loop patterns in test files."""
    changes = []
    lines = content.split('\n')
    modified_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Pattern 1: loop = asyncio.new_event_loop() followed by asyncio.set_event_loop(loop)
        if 'asyncio.new_event_loop()' in line and 'loop' in line:
            # Skip this and the next few lines that use this pattern
            indent = re.match(r'^(\s*)', line).group(1)
            j = i + 1
            
            # Look for related lines to skip
            while j < len(lines) and (
                'asyncio.set_event_loop(' in lines[j] or
                'loop.run_until_complete(' in lines[j] or
                'loop.close()' in lines[j] or
                (lines[j].strip() == '' and j < i + 5) or
                (lines[j].startswith(indent + '    ') and 'finally:' not in lines[j])
            ):
                j += 1
            
            # If we find a run_until_complete, extract the async call
            async_call = None
            for k in range(i, j):
                match = re.search(r'loop\.run_until_complete\((.*?)\)', lines[k])
                if match:
                    async_call = match.group(1).strip()
                    break
            
            if async_call:
                # Replace with await
                modified_lines.append(f"{indent}result = await {async_call}")
                changes.append(f"Replaced event loop pattern with await {async_call}")
            
            # Skip to after the pattern
            i = j
            continue
            
        # Pattern 2: asyncio.run() in test functions
        if 'asyncio.run(' in line and 'def test_' in content[:content.find(line)]:
            match = re.search(r'asyncio\.run\((.*?)\)', line)
            if match:
                indent = re.match(r'^(\s*)', line).group(1)
                async_call = match.group(1).strip()
                modified_lines.append(f"{indent}await {async_call}")
                changes.append(f"Replaced asyncio.run() with await {async_call}")
                i += 1
                continue
        
        # Pattern 3: Simple get_event_loop() calls
        if 'get_event_loop()' in line and 'asyncio' in line:
            # Skip these lines as pytest-asyncio handles the event loop
            changes.append("Removed get_event_loop() call")
            i += 1
            continue
            
        modified_lines.append(line)
        i += 1
    
    return '\n'.join(modified_lines), changes


def process_file(file_path: Path) -> dict:
    """Process a single file to fix event loop issues."""
    try:
        content = file_path.read_text()
        
        # Fix event loop patterns
        new_content, changes = fix_event_loop_patterns(content)
        
        # Check if the function using event loops should be async
        if changes:
            # Find test functions that should be async
            lines = new_content.split('\n')
            modified_lines = []
            
            for i, line in enumerate(lines):
                # If this is a test function that uses await but isn't async
                if re.match(r'^(\s*)def\s+test_', line) and not 'async' in line:
                    # Check if any line in this function uses await
                    indent = re.match(r'^(\s*)', line).group(1)
                    function_end = i + 1
                    
                    while function_end < len(lines) and (
                        lines[function_end].startswith(indent + ' ') or 
                        lines[function_end].strip() == ''
                    ):
                        function_end += 1
                    
                    # Check if function body contains await
                    function_body = '\n'.join(lines[i:function_end])
                    if 'await ' in function_body:
                        # Make it async
                        modified_lines.append(line.replace('def ', 'async def '))
                        changes.append(f"Made {line.strip()} async")
                        continue
                
                modified_lines.append(line)
            
            new_content = '\n'.join(modified_lines)
        
        # Write back if changes were made
        if changes:
            file_path.write_text(new_content)
            return {
                'file': str(file_path),
                'changes': changes,
                'success': True
            }
        else:
            return {
                'file': str(file_path),
                'changes': [],
                'success': True
            }
            
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'success': False
        }


def main():
    """Main entry point."""
    # Files identified as having event loop issues
    files_to_fix = [
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/services/observability/test_tracking.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/services/vector_db/filters/test_composer.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/services/query_processing/test_federated.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/unit/services/query_processing/test_expansion.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/integration/test_observability_integration.py",
        "/workspace/repos/ai-docs-vector-db-hybrid-scraper/tests/integration/mcp_services/test_mcp_services_integration.py",
    ]
    
    print("🔧 Fixing event loop issues in test files...\n")
    
    total_fixed = 0
    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            continue
            
        result = process_file(path)
        
        if result.get('error'):
            print(f"❌ Error processing {result['file']}: {result['error']}")
        elif result['changes']:
            total_fixed += 1
            print(f"✅ Fixed {result['file']}:")
            for change in result['changes']:
                print(f"   - {change}")
        else:
            print(f"ℹ️  No changes needed in {result['file']}")
        print()
    
    print(f"\n📊 Summary: Fixed {total_fixed} files")


if __name__ == "__main__":
    main()