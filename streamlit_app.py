import os
import sys

# Ensure relative folder imports are resolved correctly in cloud environments
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

# Launch the unified clinical diagnostic system cockpit
import app_clinical_system
