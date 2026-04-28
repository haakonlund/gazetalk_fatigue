#!/bin/bash

# Configuration
SCRIPT_PATH="/server.py"
LOG_FILE= "./monitor_script.log"

# Loop to keep the script running
while true; do
    # Run the script and capture its exit code
    python3 $SCRIPT_PATH >> $LOG_FILE 2>&1
    EXIT_CODE=$?
    
    # Check if the script exited abnormally
    if [ $EXIT_CODE -ne 0 ]; then
        # Send notification email
        echo "Script crashed with exit code $EXIT_CODE at $(date). Restarting..." >> $LOG_FILE
        # Wait a bit before restarting to avoid rapid restart loops
        sleep 10
    else
        # Script exited normally, break the loop
        echo "Script exited normally at $(date)." >> $LOG_FILE
        break
    fi
done