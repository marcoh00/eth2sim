import argparse
import pathlib


def valid_writable_path(path: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.is_dir():
        try:
            path.mkdir(parents=True)
        except OSError:
            raise argparse.ArgumentError(None, 'Could not create given directory')
    return path
