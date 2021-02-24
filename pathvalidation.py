import argparse
import os
import pathlib


def valid_writable_path(path: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.is_dir():
        try:
            path.mkdir(parents=True)
        except OSError:
            raise argparse.ArgumentError('Could not create given directory')
    if not path.owner() == os.getlogin():
        raise argparse.ArgumentError('Current user is not the owner of the given directory')
    return path
