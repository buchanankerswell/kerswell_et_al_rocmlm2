#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import sys
import yaml
import shutil
import zipfile
import datetime
import platform
from git import Repo
import urllib.request
import importlib.metadata

#######################################################
## .1.     Download Data Assets and Perple_X     !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_conda_packages(condafile):
    """
    """
    try:
        with open(condafile, "r") as file:
            conda_data = yaml.safe_load(file)
    except (IOError, yaml.YAMLError) as e:
        print(f"Error in get_conda_packages():\n {e}")
        return None

    return conda_data.get("dependencies", [])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def print_session_info(condafile):
    """
    """
    print("Session info:")
    print(f"  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    version_string = ".".join(map(str, sys.version_info))
    print(f"  Python Version: {version_string}")

    print("  Loaded packages:")
    conda_packages = get_conda_packages(condafile)

    for package in conda_packages:
        if isinstance(package, str) and package != "python":
            package_name = package.split("=")[0]
            try:
                version = importlib.metadata.version(package_name)
                print(f"      {package_name} version: {version}")
            except importlib.metadata.PackageNotFoundError:
                print(f"      {package_name} not found ...")

    os_info = platform.platform()
    print(f"  Operating System: {os_info}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def download_and_unzip(url, filename, destination):
    try:
        response = urllib.request.urlopen(url)

        with open("temp.zip", "wb") as zip_file:
            zip_file.write(response.read())

        with zipfile.ZipFile("temp.zip", "r") as zip_ref:
            if filename == "all":
                zip_ref.extractall(destination)
            else:
                target_found = False

                for file_info in zip_ref.infolist():
                    if file_info.filename == filename:
                        zip_ref.extract(file_info, destination)
                        _, file_ext = os.path.splitext(filename)

                        if file_ext == ".zip":
                            with zipfile.ZipFile(f"{destination}/{filename}", "r") as in_zip:
                                in_zip.extractall(destination)

                            os.remove(f"{destination}/{filename}")

                        target_found = True
                        break

                if not target_found:
                    raise Exception(f"{filename} not found in zip archive !")

        os.remove("temp.zip")

    except urllib.error.URLError as e:
        raise Exception(f"Unable to download from {url} !")
    except zipfile.BadZipFile as e:
        raise Exception(f"The downloaded file is not a valid zip file !")
    except Exception as e:
        raise Exception(f"An unexpected error occurred:\n  {e}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def download_github_submodule(repository_url, submodule_dir, commit_hash):
    """
    """
    try:
        if os.path.exists(submodule_dir):
            shutil.rmtree(submodule_dir)
        print(f"Cloning {repository_url} repo [commit: {commit_hash}] ...")
        repo = Repo.clone_from(repository_url, submodule_dir, recursive=True)
        repo.git.checkout(commit_hash)
    except Exception as e:
        print(f"An error occurred while cloning the GitHub repository:\n  {e}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compile_perplex():
    """
    """
    try:
        config_dir = "assets/config"
        url = ("https://www.perplex.ethz.ch/perplex/ibm_and_mac_archives/OSX/"
               "previous_version/Perple_X_7.0.9_OSX_ARM_SP_Apr_16_2023.zip")
        print(f"Installing Perple_X from:\n  {url}")
        download_and_unzip(url, "dynamic.zip", "Perple_X")
        print("Perple_X install successful!")
    except Exception as e:
        print(f"Error in compile_perplex():\n  {e}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compile_hymatz():
    """
    """
    try:
        url = "https://github.com/wangyefei/HyMaTZ.git"
        download_github_submodule(url, "python/tmp", "411a378")
        shutil.move("python/tmp/HyMaTZ", "python/HyMaTZ")
        shutil.rmtree("python/tmp")
        print("HyMaTZ install successful!")
    except Exception as e:
        print(f"Error in compile_hymatz():\n  {e}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if not os.path.exists("assets"):
        url = ("https://files.osf.io/v1/resources/erdcz/providers/osfstorage/"
               "665d7b3dd835c427734cdd2d/?zip=")
        print(f"Downloading assets from OSF:\n  {url}")
        download_and_unzip(url, "all", "assets")
    else:
        print("Data assets found!")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if not os.path.exists("python/HyMaTZ"):
        compile_hymatz()
    else:
        print("HyMaTZ programs found!")

    if not os.path.exists("Perple_X"):
        compile_perplex()
    else:
        print("Perple_X programs found!")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print_session_info("python/conda-environment.yaml")
    print("=============================================")

if __name__ == "__main__":
    main()