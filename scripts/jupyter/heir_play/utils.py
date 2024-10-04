from pathlib import Path
import os
import subprocess


ASSET_BASE_URL = "https://github.com/google/heir/releases/download/nightly/"

def abort_cleanup(filename):
    cwd = Path(dir=os.getcwd())
    tmpfile = cwd / filename
    os.remove(tmpfile)


def load_nightly(filename) -> Path:
    """Fetches the nightly heir-opt binary from GitHub and returns the path to it."""
    print(f"Loading {filename} nightly binary")
    # TODO: how to clean up the tmpdir after ipython closes?
    # At worst, the user will see this in their heir_play dir and delete it.
    cwd = Path(dir=os.getcwd())
    tmpfile = cwd / filename
    if os.path.isfile(tmpfile):
        print(f"Using existing local {filename}")
        return tmpfile

    # -L follows redirects, necessary for GH asset downloads
    asset_url = ASSET_BASE_URL + filename
    proc = subprocess.run(["curl", "-L", "-o", tmpfile, asset_url])
    if proc.returncode != 0:
        print(f"Error downloading {filename}")
        print(proc.stderr)
        return None

    proc = subprocess.run(["chmod", "a+x", tmpfile])
    if proc.returncode != 0:
        print(f"Error modifying permissions on {filename}")
        print(proc.stderr)
        abort_cleanup(filename)
        return None

    return tmpfile
