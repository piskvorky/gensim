"""Install a wheel for the current platform."""
import os
import subprocess
import sys


def main():
    subdir = sys.argv[1]
    vi = sys.version_info
    want = f'-cp{vi.major}{vi.minor}-'

    for f in os.listdir(subdir):
        if want in f:
            command = [sys.executable, '-m', 'pip', 'install', os.path.join(subdir, f)]
            subprocess.check_call(command)
            return 0

    print(f'could not find wheel for {want} in {subdir}')

    return 1



if __name__ == '__main__':
    sys.exit(main())
