#!/usr/bin/env bash
# Short Vulkan diagnostic for the lab server: which NVIDIA libs / ICD manifests /
# loaders exist, and what `vulkaninfo --summary` says for each candidate pair.
#
#   bash scripts/diagnose_vulkan.sh
#
# Read-only apart from its own temp output. Expected on a healthy setup: the
# generated manifest + current loader shows the A100 with
# DRIVER_ID_NVIDIA_PROPRIETARY and PHYSICAL_DEVICE_TYPE_DISCRETE_GPU.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
hr() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

# Where setup_server.sh put things (~/.callosum, or the scratch dir); the env file knows.
if [ -f "$HOME/.callosum-env.sh" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.callosum-env.sh" 2>/dev/null || true
fi
WORK="${CALLOSUM_SCRATCH:-${CALLOSUM_DATA:-$HOME/.callosum}}"
VENV_PY="${UV_PROJECT_ENVIRONMENT:-$WORK/venv}/bin/python"
MESA="$WORK/mesa"
VKINFO="$(command -v vulkaninfo 2>/dev/null || true)"
[ -n "$VKINFO" ] || VKINFO="$MESA/bin/vulkaninfo"

join() { local o="" p; for p in "$@"; do [ -n "$p" ] && o="${o:+$o:}$p"; done; printf '%s' "$o"; }

hr "Environment of this shell"
echo "  VK_ICD_FILENAMES = ${VK_ICD_FILENAMES:-<unset>}"
echo "  LD_LIBRARY_PATH  = ${LD_LIBRARY_PATH:-<unset>}"
echo "  NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}  NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "  data root=$WORK  python=$VENV_PY"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'

hr "NVIDIA Vulkan user-space libraries"
LDCONFIG="$(command -v ldconfig 2>/dev/null || echo /sbin/ldconfig)"
for l in libGLX_nvidia.so.0 libEGL_nvidia.so.0 libnvidia-glvkspirv libnvidia-glcore libnvidia-eglcore libcuda.so.1; do
  hit="$("$LDCONFIG" -p 2>/dev/null | awk -v n="$l" 'index($1, n) == 1 {print $NF; exit}')"
  if [ -n "$hit" ]; then printf '  ok       %-24s %s\n' "$l" "$hit"; else printf '  MISSING  %s (not in ldconfig)\n' "$l"; fi
done

hr "ICD manifests"
for f in "$WORK"/vulkan/icd.d/*.json /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json "$MESA"/share/vulkan/icd.d/*.json; do
  [ -e "$f" ] || continue
  lib="$(grep -o '"library_path"[^,}]*' "$f" | sed 's/.*:[[:space:]]*"//; s/"$//')"
  api="$(grep -o '"api_version"[^,}]*' "$f" | sed 's/.*:[[:space:]]*"//; s/"$//')"
  printf '  %s\n      library_path=%s api_version=%s\n' "$f" "$lib" "$api"
done
[ -e "$WORK/vulkan/icd.d/nvidia_icd.json" ] || echo "  (no generated manifest: setup_server.sh has not found libGLX_nvidia.so.0 or has not run)"

hr "Vulkan loaders"
"$LDCONFIG" -p 2>/dev/null | grep 'libvulkan\.so' | sed 's/^[[:space:]]*/  system: /'
for f in "$WORK"/vklib/libvulkan.so.1 "$MESA"/lib/libvulkan.so.1; do
  [ -e "$f" ] && echo "  current: $f -> $(readlink -f "$f")"
done
[ -e "$MESA/lib/libvulkan.so.1" ] || echo "  (no conda loader in $MESA: conda-forge unreachable at setup time?)"

hr "vulkaninfo --summary per candidate (loader x manifest)"
if [ ! -x "$VKINFO" ]; then
  echo "  vulkaninfo not found (setup installs it with the conda loader into $MESA/bin)"
else
  show() {  # $1 manifest ("" = stock), $2 loader dir, $3 label
    echo "  -- $3"
    (
      unset VK_ICD_FILENAMES
      if [ -n "$1" ]; then export VK_ICD_FILENAMES="$1"; fi
      export LD_LIBRARY_PATH; LD_LIBRARY_PATH="$(join "$2" "")"
      [ -n "$LD_LIBRARY_PATH" ] || unset LD_LIBRARY_PATH
      timeout 60 "$VKINFO" --summary 2>&1 | grep -E 'deviceName|deviceType|driverID|apiVersion|ERROR|Found no drivers|Instance Version' | sort -u | head -n 8
    ) | sed 's/^[[:space:]]*/       /'
  }
  for f in "$WORK"/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json "$MESA"/share/vulkan/icd.d/*.json; do
    [ -e "$f" ] || continue
    [ -e "$WORK/vklib/libvulkan.so.1" ] && show "$f" "$WORK/vklib" "$(basename "$f") + current loader"
    show "$f" "" "$(basename "$f") + system loader"
  done
fi

hr "What SAPIEN sees with the env file's settings"
if [ -x "$VENV_PY" ]; then
  "$VENV_PY" - <<'PY' 2>&1 | head -n 15
import sapien
from sapien.render import RenderSystem, get_device_summary
print(get_device_summary())
try:
    RenderSystem("cuda:0")
    print("RenderSystem('cuda:0'): OK -> hardware Vulkan, render_backend='gpu' will work")
except Exception as e:
    print("RenderSystem('cuda:0'): FAILED ->", str(e).strip().splitlines()[0] if str(e).strip() else e)
PY
else
  echo "  venv python not found: $VENV_PY"
fi
echo
echo "To re-pick the best combination: bash scripts/setup_server.sh  (log of every probe: $WORK/vulkan-probe.log)"
