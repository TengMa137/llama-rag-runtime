#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/.." && pwd)"
mobile_agent_root="${1:-${repository_root}/../mobileAgent}"
android_sdk_root="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-${HOME}/Library/Android/sdk}}"

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    ndk_root="${android_sdk_root}/ndk"
    if [[ ! -d "${ndk_root}" ]]; then
        echo "Android NDK not found; set ANDROID_NDK_HOME" >&2
        exit 1
    fi
    ANDROID_NDK_HOME="$(find "${ndk_root}" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)"
    export ANDROID_NDK_HOME
fi

destination="${mobile_agent_root}/android/app/src/main/jniLibs/arm64-v8a"
if [[ ! -f "${mobile_agent_root}/pubspec.yaml" ]]; then
    echo "mobileAgent not found at ${mobile_agent_root}" >&2
    exit 1
fi

cd "${repository_root}"
cmake --preset android-arm64
cmake --build --preset android-arm64
cmake -E make_directory "${destination}"
cmake -E copy_if_different \
    "${repository_root}/build/android-arm64/lib/librag_mobile.so" \
    "${destination}/librag_mobile.so"

strip_tool="$(find "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt" \
    -path '*/bin/llvm-strip' \( -type f -o -type l \) -print -quit)"
if [[ -n "${strip_tool}" ]]; then
    "${strip_tool}" --strip-unneeded "${destination}/librag_mobile.so"
fi

echo "Installed ${destination}/librag_mobile.so"
