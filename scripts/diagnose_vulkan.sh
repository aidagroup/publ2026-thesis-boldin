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
    if [ -d "$d" ]; then
        echo "  $d:"
        for f in "$d"/*.json; do
            [ -e "$f" ] || continue
            echo "    $f:"
            sed 's/^/      /' "$f"
            # A manifest existing proves nothing: it may name a library that is
            # absent, in which case the loader skips it and reports zero devices.
            l=$(grep -o '"library_path"[^,}]*' "$f" | sed 's/.*:[[:space:]]*"//; s/"$//' || true)
            if [ -n "$l" ]; then
                case "$l" in
                    /*) [ -e "$l" ] && echo "      -> library EXISTS: $l" \
                                    || echo "      -> library MISSING: $l" ;;
                    *)  r=$(ldconfig -p 2>/dev/null | grep -m1 "$l" | awk '{print $NF}' || true)
                        [ -n "$r" ] && echo "      -> resolves to: $r" \
                                    || echo "      -> soname '$l' NOT resolvable by ldconfig" ;;
                esac
            fi
        done
    else echo "  $d: does not exist"; fi
done
echo "  VK_ICD_FILENAMES = ${VK_ICD_FILENAMES:-<unset>}"

hr "vulkaninfo"
if command -v vulkaninfo > /dev/null 2>&1; then
    vulkaninfo --summary 2>&1 | head -20
else
    echo "  not installed (not required — the library check above is what matters)"
fi

hr "Which ICD manifest is in effect"
echo "  VK_ICD_FILENAMES = ${VK_ICD_FILENAMES:-<unset>}"
MANIFEST="${VK_ICD_FILENAMES:-}"; MANIFEST="${MANIFEST%%:*}"
if [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ]; then
    sed 's/^/    /' "$MANIFEST"
    LIB=$(grep -o '"library_path"[^,}]*' "$MANIFEST" | sed 's/.*:[[:space:]]*"//; s/"$//' || true)
    echo "  library_path -> ${LIB:-<unparsed>}"
else
    echo "  (no manifest set; SAPIEN will fall back to its bundled one)"
    LIB=""
fi

hr "Can that library actually load? (the usual culprit)"
# libGLX_nvidia.so.0 needs libnvidia-glcore.so.<exact driver version>. Container
# runtimes sometimes inject an incomplete or mismatched set, and then the Vulkan
# loader silently skips the ICD -- instance creation succeeds with zero devices.
if [ -z "${LIB:-}" ] || [ ! -e "$LIB" ]; then
    LIB=$(ldconfig -p 2>/dev/null | awk '/libGLX_nvidia\.so\.0/ {print $NF; exit}' || true)
fi
if [ -n "${LIB:-}" ] && [ -e "$LIB" ]; then
    echo "  ldd $LIB"
    if ldd "$LIB" 2>&1 | grep -i 'not found'; then
        echo "  ^^^ UNRESOLVED dependencies — this is why no device is found"
    else
        echo "    all dependencies resolved"
    fi
else
    echo "  library not found at all"
fi

hr "Driver device nodes"
ls -1 /dev/nvidia* 2>/dev/null | sed 's/^/  /' || echo "  none present"

hr "What SAPIEN itself sees"
if command -v uv > /dev/null 2>&1 && [ -f pyproject.toml ]; then
    uv run python - <<'PYEOF' 2>&1 | sed 's/^/  /'
import sapien
for alias in ("cuda", "cuda:0", "cpu"):
    try:
        d = sapien.Device(alias)
        print(f"{alias:8} name={d.name!r} can_render={d.can_render()} "
              f"is_cuda={d.is_cuda()} pci={d.pci_string}")
    except Exception as e:
        print(f"{alias:8} {type(e).__name__}: {e}")
try:
    from sapien.render import RenderMaterial
    RenderMaterial()
    print("RenderMaterial(): ok — a render device IS available")
except Exception as e:
    print(f"RenderMaterial(): {type(e).__name__}: {e}")
PYEOF
else
    echo "  (run from the repo root with uv available to probe sapien)"
fi

hr "Vulkan loader trace (says exactly why an ICD is rejected)"
if command -v uv > /dev/null 2>&1 && [ -f pyproject.toml ]; then
    VK_LOADER_DEBUG=error,warn uv run python -c "
import sapien
from sapien.render import RenderMaterial
try:
    RenderMaterial(); print('RenderMaterial ok')
except Exception as e:
    print(type(e).__name__, e)
" 2>&1 | grep -iE 'icd|driver|loader|error|warn|manifest|libGLX|RenderMaterial' | head -25 | sed 's/^/  /'
else
    echo "  (needs uv and the repo root)"
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
