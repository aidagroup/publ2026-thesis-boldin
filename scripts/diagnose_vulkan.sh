#!/usr/bin/env bash
# Why can't SAPIEN find a rendering device?
#
# SAPIEN's renderer is Vulkan-only, and its URDF loader builds RenderMaterial
# objects unconditionally — so ManiSkill cannot create ANY environment without a
# working Vulkan device, even with render_backend="none" and state-only
# observations. This script decides whether that is fixable from user space or
# needs a change to how the container is launched.
#
#   bash scripts/diagnose_vulkan.sh
set -uo pipefail

hr() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

hr "Driver capabilities requested for this container"
echo "NVIDIA_DRIVER_CAPABILITIES = ${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES     = ${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "(compute,utility = maths only. Vulkan additionally needs 'graphics'.)"

hr "GPU is visible at all?"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -2

hr "NVIDIA graphics libraries (the decisive check)"
found=0
for lib in libGLX_nvidia.so.0 libnvidia-glcore.so libEGL_nvidia.so.0 libnvidia-eglcore.so; do
    hit=$(ldconfig -p 2>/dev/null | grep -m1 "$lib" | awk '{print $NF}')
    if [ -z "$hit" ]; then
        hit=$(find /usr/lib /usr/lib64 /usr/local -maxdepth 4 -name "${lib}*" 2>/dev/null | head -1)
    fi
    if [ -n "$hit" ]; then printf '  %-24s ✓ %s\n' "$lib" "$hit"; found=1
    else                   printf '  %-24s ✗ missing\n' "$lib"; fi
done

hr "Vulkan loader and ICD files"
ldconfig -p 2>/dev/null | grep -i 'libvulkan' | head -3 || echo "  libvulkan: not in ldconfig"
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
    if [ -d "$d" ]; then echo "  $d:"; ls -1 "$d" 2>/dev/null | sed 's/^/    /'
    else echo "  $d: does not exist"; fi
done
echo "  VK_ICD_FILENAMES = ${VK_ICD_FILENAMES:-<unset>}"

hr "vulkaninfo"
if command -v vulkaninfo > /dev/null 2>&1; then
    vulkaninfo --summary 2>&1 | head -20
else
    echo "  not installed (not required — the library check above is what matters)"
fi

hr "Verdict"
if [ "$found" = "1" ]; then
    cat <<TXT
  The NVIDIA graphics libraries ARE present. Vulkan should be reachable; the
  problem is likely just a missing ICD file. Try pointing the loader at the
  library explicitly (adjust the path printed above):

    mkdir -p ~/.local/share/vulkan/icd.d
    cat > ~/.local/share/vulkan/icd.d/nvidia_icd.json <<'JSON'
    { "file_format_version": "1.0.0",
      "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.242" } }
JSON
    export VK_ICD_FILENAMES=~/.local/share/vulkan/icd.d/nvidia_icd.json
    uv run python scripts/smoke_env.py
TXT
else
    cat <<TXT
  The NVIDIA graphics libraries are ABSENT. This cannot be fixed from user
  space: the libraries are injected into the container by the NVIDIA runtime
  only when the pod requests the 'graphics' capability, and this one did not.

  ManiSkill cannot create any environment without them — SAPIEN's URDF loader
  builds render materials unconditionally, so even state-only observations and
  render_backend="none" still need a Vulkan device.

  Ask the lab administrator to launch the image with:
      NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  (or 'all'), which is the standard setting for simulation workloads.
TXT
fi
