import os
from pathlib import Path
from lib.Errors import NotFoundError

# constants
BASE_PATH = Path(os.getcwd())
LIB_PATH = BASE_PATH / "lib"
TESTS_PATH =  LIB_PATH / "tests"
DATA_PATH = BASE_PATH / "data"
XEROX_PATH = BASE_PATH / "xerox"

class Filer(object):

    def __init__(self) -> None:
        self.base_folderpaths: dict[str, Path] = self.set_base_folderpaths()

    def set_base_folderpaths(self):
        # define relevant paths
        base_folderpaths = {
            "cwd": BASE_PATH,
            "data" : DATA_PATH,
            "xerox" : XEROX_PATH,
            "lib" : LIB_PATH,
            "tests": TESTS_PATH
        }
        # if all relevant paths exist
        if all(map(lambda f: f.exists(), base_folderpaths.values())):
            # return them
            return base_folderpaths
        # otherwise
        else:
            # raise error
            raise NotFoundError(f"Missing paths: {[ str(f) for f in base_folderpaths.values() if not f.exists() ]}.")

    def get_base_folderpath(self, folderpath: str = "all") -> Path:
        # if user requests a valid specific path
        if folderpath in self.base_folderpaths.keys():
            # return it
            return self.base_folderpaths[folderpath]
        # otherwise
        else:
            # raise error
            raise NotFoundError(f"'{folderpath}' doesn't exist.")


    def get_test_folderpath(self, test : str) -> Path:
        # determine test folderpath
        test_folderpath =  TESTS_PATH / test
        # if test folderpath exists
        if test_folderpath.exists():
            # return test folderpath
            return test_folderpath.relative_to(BASE_PATH)
        # otherwise
        else:
            # raise error
            raise NotFoundError(f"'{test_folderpath}' doesn't exist.")
