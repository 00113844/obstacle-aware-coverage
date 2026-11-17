#!/usr/bin/env bash
# Build Fields2Cover v2.1.1 + OR-Tools v9.9 on a Nectar (OpenStack) Ubuntu 22.04 VM.
# Same recipe as Part 0 of the Colab notebook (tools/build_notebook.py), made native and reusable.
# Usage:  bash nectar_setup.sh            (run as a sudo-capable user, ~10-15 min)
set -euo pipefail

F2C_GIT_TAG="${F2C_GIT_TAG:-v2.1.1}"
ORTOOLS_URL="https://github.com/google/or-tools/releases/download/v9.9/or-tools_amd64_ubuntu-22.04_cpp_v9.9.3963.tar.gz"
VENV="${VENV:-$HOME/f2c-venv}"
WORK="${WORK:-$HOME/f2c-build}"

. /etc/os-release
echo "OS: $PRETTY_NAME   cores: $(nproc)   RAM: $(free -g | awk '/Mem/{print $2}') GB"
[ "$VERSION_ID" = "22.04" ] || echo "WARNING: the OR-Tools tarball targets Ubuntu 22.04; on $VERSION_ID pick the matching tarball."

# 1. System dependencies
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential cmake git wget python3-venv python3-dev swig \
  libgdal-dev libgeos-dev libeigen3-dev libtbb-dev libtinyxml2-dev nlohmann-json3-dev libboost-dev \
  tmux htop

# 2. OR-Tools C++ (prebuilt)
if [ ! -d /opt/or-tools/lib/cmake/ortools ]; then
  wget -q -O /tmp/ortools.tar.gz "$ORTOOLS_URL"
  sudo mkdir -p /opt/or-tools
  sudo tar -xzf /tmp/ortools.tar.gz -C /opt/or-tools --strip-components=1
  echo /opt/or-tools/lib | sudo tee /etc/ld.so.conf.d/ortools.conf >/dev/null
  sudo ldconfig
fi

# 3. Python environment
python3 -m venv "$VENV"
. "$VENV/bin/activate"
pip install -q --upgrade pip wheel
pip install -q "shapely>=2.0" numpy pandas matplotlib statsmodels scipy

# 4. Fields2Cover: source checkout (kept, so it can be patched and rebuilt) + wheel
mkdir -p "$WORK"
if [ ! -d "$WORK/Fields2Cover" ]; then
  git clone --branch "$F2C_GIT_TAG" --depth 1 https://github.com/Fields2Cover/Fields2Cover.git "$WORK/Fields2Cover"
fi
export CMAKE_PREFIX_PATH=/opt/or-tools
export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
pip wheel --no-deps -w "$WORK/wheels" "$WORK/Fields2Cover"
pip install --force-reinstall --no-deps "$WORK"/wheels/fields2cover-*.whl

# 5. Smoke test
python - <<'EOF'
import fields2cover as f2c
rand = f2c.Random(42)
robot = f2c.Robot(2.0, 6.0)
field = rand.generateRandField(1e4, 5)
cells = field.getField()
hl = f2c.HG_Const_gen().generateHeadlands(cells, 3.0 * robot.getWidth())
sw = f2c.SG_BruteForce().generateBestSwaths(f2c.OBJ_NSwath(), robot.getCovWidth(), hl.getGeometry(0))
print("Fields2Cover OK - swaths:", sw.size())
EOF
echo "Done. Activate with:  . $VENV/bin/activate"
