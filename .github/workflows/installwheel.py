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
        if machine == 'arm64':
            arch = 'arm64'
        else:
            arch = 'x86_64'
    elif system == 'Linux':
        if machine in ('x86_64', 'amd64'):
            arch = 'x86_64'
        elif machine in ('aarch64', 'arm64'):
            arch = 'aarch64'
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
