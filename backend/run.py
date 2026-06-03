import sys
from pathlib import Path
import uvicorn

# Append parent directory to sys.path to resolve 'backend' imports properly
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

if __name__ == "__main__":
    print("Starting Uvicorn FastAPI server on http://localhost:8000...")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
