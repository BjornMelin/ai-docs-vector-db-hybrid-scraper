#!/bin/bash
# Example workflow for a parallel subagent

# Set agent ID and role
AGENT_ID="agent-naming-$(date +%s)"
AGENT_ROLE="naming-cleanup"

echo "Starting $AGENT_ROLE subagent with ID: $AGENT_ID"

# 1. Register with coordinator
echo "Registering with coordinator..."
python tests/scripts/parallel_coordinator.py register "$AGENT_ID" "$AGENT_ROLE"

# 2. Main work loop
while true; do
    # Get next task
    echo -e "\nGetting next task..."
    TASK=$(python tests/scripts/parallel_coordinator.py task "$AGENT_ID" "$AGENT_ROLE")
    
    if [[ "$TASK" == "No tasks available" ]]; then
        echo "No more tasks available. Work complete!"
        break
    fi
    
    # Extract file path from task (this is simplified - real agent would parse JSON)
    FILE_PATH=$(echo "$TASK" | grep -o '"file_path": "[^"]*"' | cut -d'"' -f4)
    
    if [[ -z "$FILE_PATH" ]]; then
        echo "Could not extract file path from task"
        continue
    fi
    
    echo "Processing: $FILE_PATH"
    
    # 3. Claim the file
    echo "Claiming file..."
    python tests/scripts/parallel_coordinator.py claim "$AGENT_ID" "$FILE_PATH"
    
    # 4. Validate current state
    echo "Validating current state..."
    ISSUES_BEFORE=$(python tests/scripts/validate_test_quality.py "$FILE_PATH" --json | grep -c '"severity": "error"')
    
    # 5. Perform the work (naming cleanup in this example)
    echo "Fixing naming issues..."
    python tests/scripts/rename_antipattern_files.py "$FILE_PATH" --execute
    
    # 6. Validate after fix
    echo "Validating after fix..."
    ISSUES_AFTER=$(python tests/scripts/validate_test_quality.py "$FILE_PATH" --json | grep -c '"severity": "error"')
    
    # 7. Calculate issues fixed
    ISSUES_FIXED=$((ISSUES_BEFORE - ISSUES_AFTER))
    
    # 8. Report progress
    echo "Reporting progress (fixed $ISSUES_FIXED naming issues)..."
    python tests/scripts/parallel_coordinator.py progress "$AGENT_ID" \
        --processed 1 \
        --fixed "naming:$ISSUES_FIXED"
    
    # 9. Release the file
    echo "Releasing file..."
    python tests/scripts/parallel_coordinator.py release "$AGENT_ID" "$FILE_PATH"
    
    echo "Task complete!"
done

# 10. Check final status
echo -e "\nFinal status:"
python tests/scripts/parallel_coordinator.py status

echo -e "\nSubagent $AGENT_ID work complete!"