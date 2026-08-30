#!/usr/bin/env bash
# Why can't SAPIEN find a rendering device?
#
# SAPIEN's renderer is Vulkan-only, and its URDF loader builds RenderMaterial
# objects unconditionally — so ManiSkill cannot create ANY environment without a
# working Vulkan device, even with render_backend="none" and state-only
# observations.
#
#   bash scripts/diagnose_vulkan.sh
#
# Everything decisive is repeated in the SUMMARY block at the very end: that is
# the part to copy. The sections above it are the raw evidence behind it.
set -uo pipefail
# uv lives in ~/.local/bin, which is only on PATH once ~/.bashrc is sourced —
# and a non-interactive `bash scripts/...` does not source it. Without this the
# two most valuable probes below silently skip themselves.
export PATH="$HOME/.local/bin:$PATH"

hr() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

CAPS="${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
VISIBLE="${NVIDIA_VISIBLE_DEVICES:-<unset>}"
GPU=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)

hr "Container GPU configuration"
echo "  NVIDIA_DRIVER_CAPABILITIES = $CAPS"
echo "  NVIDIA_VISIBLE_DEVICES     = $VISIBLE"
echo "  GPU                        = ${GPU:-<nvidia-smi failed>}"

hr "NVIDIA graphics libraries"
LIBGLX=$(ldconfig -p 2>/dev/null | awk '/libGLX_nvidia\.so\.0/ {print $NF; exit}' || true)
[ -n "$LIBGLX" ] && [ -e "$LIBGLX" ] || LIBGLX=$(find /usr/lib /usr/lib64 -maxdepth 4 -name 'libGLX_nvidia.so.0' 2>/dev/null | head -1 || true)
for lib in libGLX_nvidia.so.0 libnvidia-glcore.so libEGL_nvidia.so.0 libvulkan.so.1; do
    hit=$(ldconfig -p 2>/dev/null | grep -m1 "$lib" | awk '{print $NF}' || true)
    [ -n "$hit" ] && printf '  %-22s ✓ %s\n' "$lib" "$hit" || printf '  %-22s ✗ missing\n' "$lib"
done

hr "Do the graphics libraries resolve their own dependencies?"
# libGLX_nvidia.so.0 needs libnvidia-glcore.so.<exact driver version>. If any
# dependency is missing the Vulkan loader silently skips the ICD, the instance
# is created with ZERO devices, and SAPIEN reports "failed to find a rendering device".
LDD_MISSING=""
if [ -n "$LIBGLX" ] && [ -e "$LIBGLX" ]; then
    LDD_MISSING=$(ldd "$LIBGLX" 2>&1 | grep -i 'not found' | awk '{print $1}' | tr '\n' ' ' || true)
    if [ -n "$LDD_MISSING" ]; then echo "  UNRESOLVED: $LDD_MISSING"
    else echo "  all dependencies of $LIBGLX resolve"; fi
else
    echo "  libGLX_nvidia.so.0 not present — nothing to check"
fi

hr "ICD manifests (a manifest existing proves nothing)"
ICD_REPORT=""
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
    if [ ! -d "$d" ]; then echo "  $d: does not exist"; continue; fi
    for f in "$d"/*.json; do
        [ -e "$f" ] || continue
        echo "  $f:"; sed 's/^/    /' "$f"
        l=$(grep -o '"library_path"[^,}]*' "$f" | sed 's/.*:[[:space:]]*"//; s/"$//' || true)
        case "$l" in
            "") verdict="(no library_path parsed)" ;;
            /*) [ -e "$l" ] && verdict="library exists" || verdict="LIBRARY MISSING: $l" ;;
            *)  r=$(ldconfig -p 2>/dev/null | grep -m1 "$l" | awk '{print $NF}' || true)
                [ -n "$r" ] && verdict="resolves to $r" || verdict="SONAME '$l' NOT RESOLVABLE" ;;
        esac
        echo "    -> $verdict"
        ICD_REPORT="$ICD_REPORT $f:$verdict;"
    done
done
echo "  VK_ICD_FILENAMES = ${VK_ICD_FILENAMES:-<unset>}"

hr "Driver device nodes"
DEV_NODES=$(ls -1 /dev/nvidia* 2>/dev/null | tr '\n' ' ' || true)
echo "  ${DEV_NODES:-<none present>}"

hr "What SAPIEN sees"
SAPIEN_OUT="(not probed — needs uv and the repo root)"
if command -v uv > /dev/null 2>&1 && [ -f pyproject.toml ]; then
    SAPIEN_OUT=$(uv run python - <<'PYEOF' 2>&1 | tail -6
import sapien
for alias in ("cuda", "cpu"):
    try:
        d = sapien.Device(alias)
        print(f"{alias}: name={d.name!r} can_render={d.can_render()} pci={d.pci_string}")
    except Exception as e:
        print(f"{alias}: {type(e).__name__}: {e}")
try:
    from sapien.render import RenderMaterial
    RenderMaterial()
    print("RenderMaterial: OK — a render device IS available")
except Exception as e:
    print(f"RenderMaterial: {type(e).__name__}: {e}")
PYEOF
)
fi
echo "$SAPIEN_OUT" | sed 's/^/  /'

hr "Vulkan loader trace"
TRACE="(not probed)"
if command -v uv > /dev/null 2>&1 && [ -f pyproject.toml ]; then
    TRACE=$(VK_LOADER_DEBUG=error,warn uv run python -c "
import sapien
from sapien.render import RenderMaterial
try: RenderMaterial()
except Exception as e: pass
" 2>&1 | grep -iE 'icd|driver|loader|manifest|libGLX' | head -12 || true)
    [ -n "$TRACE" ] || TRACE="(loader printed nothing)"
fi
echo "$TRACE" | sed 's/^/  /'

# ─────────────────────────────────────────────────────────────────────────────
printf '\n\033[1;33m%s\033[0m\n' "================ SUMMARY — copy from here ================"
echo "caps=$CAPS  visible=$VISIBLE"
echo "gpu=${GPU:-none}"
echo "libGLX=${LIBGLX:-MISSING}"
echo "unresolved_deps=${LDD_MISSING:-none}"
echo "dev_nodes=${DEV_NODES:-none}"
echo "icd:${ICD_REPORT:- none found}"
echo "vk_icd_filenames=${VK_ICD_FILENAMES:-unset}"
echo "--- sapien ---"
echo "$SAPIEN_OUT"
echo "--- loader trace ---"
echo "$TRACE"
printf '\033[1;33m%s\033[0m\n' "=========================================================="
