"""
Setup script for pysz - Python bindings for SZ3
Automatically downloads and builds SZ3 with bundled zstd.
"""

import sys
import subprocess
import shutil
import os
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np



SZ3_VERSION = "3.3.2"

class BuildSZ3Extension(_build_ext):

    def run(self):
        sz3_dir = self.download_and_build_sz3()

        for ext in self.extensions:
            ext.include_dirs.insert(0, str(sz3_dir / "include"))
            ext.include_dirs.insert(0, str(sz3_dir / "build" / "include"))
            ext.include_dirs.append(str(sz3_dir / "build" / "_deps" / "zstdfetched-src" / "lib"))
            ext.library_dirs.append(str(sz3_dir / "build" / "tools" / "zstd"))
            ext.library_dirs.append(str(sz3_dir / "build" / "tools" / "zstd" / "Release"))
            ext.library_dirs.append(str(sz3_dir / "build" / "tools" / "zstd" / "Debug"))

        super().run()

        if sys.platform == "darwin":
            zstd_lib_name = "libzstd.dylib"
        elif sys.platform == "win32":
            zstd_lib_name = "zstd.dll"
        else:
            zstd_lib_name = "libzstd.so"

        zstd_base = sz3_dir / "build" / "tools" / "zstd"
        package_dir = Path(self.build_lib) / "pysz"
        if package_dir.exists():
            for subdir in ["", "Release", "Debug"]:
                zstd_lib = zstd_base / subdir / zstd_lib_name
                if zstd_lib.exists():
                    shutil.copy2(zstd_lib, package_dir / zstd_lib.name)
                    print(f"Copied {zstd_lib.name} to package")
                    break

    def download_and_build_sz3(self):
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)
        sz3_dir = build_temp / "SZ3"
        source_marker = sz3_dir / ".source_path"

        local_sz3_source = self.find_local_sz3_source()
        if local_sz3_source is not None:
            local_source_str = str(local_sz3_source.resolve())
            # Always refresh the copied SZ3 source so local header/source edits are picked up.
            if sz3_dir.exists():
                shutil.rmtree(sz3_dir)
            print(f"Using local SZ3 source: {local_source_str}")
            shutil.copytree(
                local_sz3_source,
                sz3_dir,
                ignore=shutil.ignore_patterns(".git", "build", "__pycache__"),
            )
            source_marker.write_text(local_source_str, encoding="utf-8")
        else:
            raise RuntimeError(
                "Local SZ3 source does not exist or is invalid. "
                "Set PYSZ_LOCAL_SZ3_DIR or ensure repository layout contains SZ3/."
            )
            # if not sz3_dir.exists():
            #     print(f"Cloning SZ3 v{SZ3_VERSION}...")
            #     subprocess.run(
            #         [
            #             "git",
            #             "clone",
            #             "--depth",
            #             "1",
            #             "--branch",
            #             f"v{SZ3_VERSION}",
            #             "--single-branch",
            #             "https://github.com/szcompressor/SZ3.git",
            #             str(sz3_dir),
            #         ],
            #         check=True,
            #     )

        build_dir = sz3_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        cmake_args = ["cmake"]
        cmake_args.extend([
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_SZ3_BINARY=OFF",
            "-DSZ3_USE_BUNDLED_ZSTD=ON",
            ".."
        ])
        subprocess.run(cmake_args, cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_dir, check=True)
        print(f"Built SZ3 v{SZ3_VERSION}")
        return sz3_dir

    def find_local_sz3_source(self):
        env_source = os.environ.get("PYSZ_LOCAL_SZ3_DIR")
        if env_source:
            candidate = Path(env_source).expanduser().resolve()
            if self.is_valid_sz3_source(candidate):
                return candidate
            raise RuntimeError(f"PYSZ_LOCAL_SZ3_DIR is not a valid SZ3 source: {candidate}")

        candidate = Path(__file__).resolve().parents[2]
        if self.is_valid_sz3_source(candidate):
            return candidate
        return None

    @staticmethod
    def is_valid_sz3_source(path: Path) -> bool:
        return (
            (path / "CMakeLists.txt").exists()
            and (path / "include" / "SZ3" / "version.hpp.in").exists()
            and (path / "include" / "SZ3" / "api" / "sz.hpp").exists()
        )




def create_extensions():
    include_dirs = [np.get_include()]
    library_dirs = []
    libraries = ['zstd']
    extra_compile_args = []
    extra_link_args = []
    
    if sys.platform == 'win32':
        extra_compile_args.extend(['/std:c++17', '/O2'])
    elif sys.platform == 'darwin':
        extra_compile_args.extend(['-std=c++17', '-O3', '-stdlib=libc++', '-mmacosx-version-min=10.9'])
        extra_link_args.extend(['-stdlib=libc++', '-Wl,-rpath,@loader_path'])
    elif sys.platform == 'linux':
        extra_compile_args.extend(['-std=c++17', '-O3'])
        extra_link_args.extend(['-Wl,-rpath,$ORIGIN'])
    
    extensions = [
        Extension(
            "pysz.sz",
            sources=["src/pysz/sz.pyx"],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]
    
    return cythonize(extensions, compiler_directives={'language_level': '3', 'embedsignature': True})


if __name__ == "__main__":
    setup(
        name="pysz",
        version="1.0.3",
        packages=["pysz"],
        package_dir={"": "src"},
        ext_modules=create_extensions(),
        cmdclass={'build_ext': BuildSZ3Extension},
        test_suite="tests",
        tests_require=["pytest"],
    )
