from google.cloud import storage
from google.auth import credentials
import subprocess
import sys

# Abre o browser para autenticar
subprocess.run([
    sys.executable, "-m", "google.auth.transport.requests"
])