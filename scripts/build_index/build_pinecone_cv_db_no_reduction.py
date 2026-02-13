"""
Wrapper to build CV Pinecone DB from HIE_DB_NO_REDUCTION
using the existing folds file.
"""

import subprocess
import sys


def main() -> int:
    cmd = [
        sys.executable,
        "build_pinecone_cv_db.py",
        "--data-dir",
        "HIE_DB_NO_REDUCTION",
        "--index-name",
        "your-pinecone-index-name-no-reduction",
        "--folds-file",
        "cv_folds/hie_db_folds.json",
    ]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
