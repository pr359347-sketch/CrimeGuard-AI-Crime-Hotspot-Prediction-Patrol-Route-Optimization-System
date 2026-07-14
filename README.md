Crime Hotspot Optimizer — reorganized

Project reorganized into canonical layout for clarity and deployment.

Top-level folders:
- 01.app/ — apps, demo scripts and Streamlit UI
- 02.data/ — data and data ingestion scripts
- 03.notebook/ — analysis notebooks
- 04.models/ — trained models and checkpoints
- 05.src/ — canonical source code (use this in imports)
- 06.images/ — generated visual outputs (HTML, maps)

Quick next steps to finalize (run these locally in the repo root):

PowerShell (recommended on Windows):

```powershell
.
\scripts\finalize_reorg.ps1
```

Or run commands manually:

```powershell
git checkout -b reorganize-structure
git add -A
git commit -m "Reorganize: move files into 01.app..06.images and canonicalize 05.src/src; remove duplicates"
git push -u origin reorganize-structure
```

Package generated images (from repo root):

```bash
python scripts/package_images.py
```

Run examples (after installing requirements in `05.src/requirements.txt`):

```bash
python 01.app/examples/example_basic_usage.py
python 01.app/examples/example_route_optimization.py
python 01.app/examples/example_model_evaluation.py
```

If you want, I can continue: run the examples here, fix any remaining import issues, and finalize commits if you allow.
