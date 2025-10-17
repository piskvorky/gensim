"""Install a wheel for the current platform."""
import os
import platform
import subprocess
import sys


def main():
    subdir = sys.argv[1]
    vi = sys.version_info

    system = platform.system()
    machine = platform.machine().lower()

    if system == 'Darwin':
        arch = 'arm64' if machine == 'arm64' else 'x86_64'
    elif system == 'Linux':
        arch = 'x86_64' if machine in ('x86_64', 'amd64') else 'aarch64' if machine in ('aarch64', 'arm64') else 'amd64'
    elif system == 'Windows':
        arch = 'amd64' if machine in ('x86_64', 'amd64') else 'arm64' if machine in ('arm64', 'aarch64') else 'amd64'
    else:
        arch = 'amd64'

    want = f'-cp{vi.major}{vi.minor}-'
    suffix = f'_{arch}.whl'

    files = sorted(os.listdir(subdir))
    for f in files:
        if want in f and f.endswith(suffix):
            command = [sys.executable, '-m', 'pip', 'install', os.path.join(subdir, f)]
            subprocess.check_call(command)
            return 0

    print(f'no matches for {want} / {suffix} in {subdir}:')
    print('\n'.join(files))

    return 1



if __name__ == '__main__':
    sys.exit(main())
