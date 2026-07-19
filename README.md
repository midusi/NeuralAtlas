# NeuralAtlas

Generate attribution maps for image classification models and serve them through a React viewer.

## Backend (Python)

The backend is managed with [uv](https://docs.astral.sh/uv/). Dependencies and the
Python version are pinned in `pyproject.toml` / `uv.lock` / `.python-version` 

1. Install uv 
2. Create the local environment and install locked dependencies:
   ```bash
   uv sync
   ```
   This creates a project-local `.venv/` and installs the exact versions from `uv.lock`.
3. Run the attribution pipeline:
   ```bash
   uv run python main.py --help
   ```

## Frontend (React + Vite)

To run in development mode:

1. Clone the repository
2. Navigate to the `interpretability-viewer` directory
3. Install dependencies with `npm install`
4. Start the development server with `npm run dev`
