"""在导入 robosuite / mujoco 之前选择离屏渲染后端。

无显示器时 EGL 的 device 平台经常初始化失败。OSMesa 可以软件出图，
但必须进程里真能加载 libOSMesa；否则 PyOpenGL 会在 import 时崩掉。
"""

import ctypes
import ctypes.util
import os
import subprocess
import sys
from pathlib import Path

_READY = "ROBO_GL_READY"
_OSMESA_NAMES = ("libOSMesa.so", "libOSMesa.so.8", "libOSMesa.so.6")


def configure():
    """按平台设置 MUJOCO_GL。必须在 import robosuite 之前调用。"""
    if sys.platform == "darwin":
        os.environ.setdefault("MUJOCO_GL", "cgl")
        return

    # 无显示器时 EGL 的 device 平台经常失败。环境里若已有 MUJOCO_GL=egl，
    # 仍然优先改用 OSMesa；只有加载不到 libOSMesa 时才保留原来的设置。
    requested = os.environ.get("MUJOCO_GL", "").strip().lower()
    if requested not in ("", "egl", "osmesa"):
        return
    if _activate_osmesa():
        return
    if requested == "osmesa":
        raise SystemExit(install_hint("已指定 MUJOCO_GL=osmesa，但当前进程加载不了 libOSMesa。"))


def _activate_osmesa():
    found = _find_osmesa()
    if found is None:
        return False
    try:
        ctypes.CDLL(found, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False

    link_dir = _link_dir(found)
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    try:
        ctypes.CDLL("libOSMesa.so", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        if os.environ.get(_READY) == "1":
            raise SystemExit(install_hint("找到了 {}，但动态链接器仍然加载不了 libOSMesa.so。".format(found)))
        _prepend_library_path(link_dir)
        os.environ[_READY] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def _find_osmesa():
    directories = [Path(os.environ.get("CONDA_PREFIX") or sys.prefix) / "lib"]
    directories.extend(
        Path(entry)
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if entry
    )
    directories.extend(
        [
            Path("/usr/lib64"),
            Path("/usr/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib/aarch64-linux-gnu"),
        ]
    )
    for directory in directories:
        for name in _OSMESA_NAMES:
            path = directory / name
            if path.is_file():
                return str(path.resolve())
    soname = ctypes.util.find_library("OSMesa")
    if not soname:
        return None
    resolved = _resolve_ldconfig(soname)
    return resolved or soname


def _resolve_ldconfig(soname):
    try:
        output = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        try:
            output = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return None
    for line in output.splitlines():
        if soname not in line or "=>" not in line:
            continue
        path = line.split("=>", 1)[1].strip()
        if path and Path(path).is_file():
            return path
    return None


def _link_dir(found):
    path = Path(found)
    if not path.is_absolute():
        return path.parent if path.parent != Path("") else Path(".")
    if path.name == "libOSMesa.so":
        return path.parent
    directory = Path.home() / ".cache" / "robo-world-gl"
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / "libOSMesa.so"
    if not link.is_symlink() and link.exists():
        link.unlink()
    if not link.exists() or link.resolve() != path.resolve():
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(path.resolve())
    return directory


def _prepend_library_path(directory):
    directory = str(directory)
    current = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    if directory not in current:
        os.environ["LD_LIBRARY_PATH"] = ":".join([directory, *current])


def install_hint(reason):
    return (
        "{}\n"
        "测评要离屏画出 frontview。这台机器的 EGL 没有 device 平台，进程里也没有可用的 OSMesa。\n"
        "在运行评测的 conda 环境里安装后重试，不要再设 MUJOCO_GL=egl：\n"
        "  conda install -y -c conda-forge 'mesalib<25.1'\n"
        "没有 conda-forge 时，用系统包：\n"
        "  sudo yum install -y mesa-libOSMesa mesa-libOSMesa-devel\n"
    ).format(reason)
