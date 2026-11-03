#!/usr/bin/env bash
# Create (or update) the shared virtual environment at <root>/.venv and install every project in this
# directory that has a pyproject.toml, in editable mode. Run it again after adding a new project.
#   ./setup_env.sh                 # all projects
#   EXTRAS="./vsca[eval]" ./setup_env.sh   # optional extras, e.g. lm-eval for vsca
set -euo pipefail
cd "$(dirname "$0")"
PYVER=${PYVER:-3.11}
[ -d .venv ] || uv venv --python "$PYVER" .venv
projects=()
for d in */; do
  [ -f "$d/pyproject.toml" ] && projects+=("-e" "./${d%/}")
done
[ ${#projects[@]} -gt 0 ] || { echo "no projects with pyproject.toml found"; exit 1; }
uv pip install --python .venv/bin/python "${projects[@]}"
if [ -n "${EXTRAS:-}" ]; then
  for e in $EXTRAS; do uv pip install --python .venv/bin/python -e "$e"; done
fi
echo "installed:${projects[*]}"
echo "activate with: source $(pwd)/.venv/bin/activate"
