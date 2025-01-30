import subprocess
import datetime
import os

def run_scripts():
    try:
        print(f"Starting data collection at {datetime.datetime.now()}")
        subprocess.run(["python", "./data_collector/main.py"], check=True)
        print(f"Starting training process at {datetime.datetime.now()}")
        subprocess.run(["python", "./modeling/train_process.py"], check=True)
        print("Scripts executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    lib_path = "/usr/lib/x86_64-linux-gnu/libta_lib.so.0.0.0"
    python_lib_path = "/usr/local/lib/python3.10/dist-packages/talib"

    # Remove old files if they exist
    if os.path.exists(lib_path):
        os.remove(lib_path)

    if os.path.exists(python_lib_path):
        subprocess.run(f"rm -rf {python_lib_path}", shell=True)

    # Download and extract
    urls = [
        "https://anaconda.org/conda-forge/libta-lib/0.4.0/download/linux-64/libta-lib-0.4.0-h166bdaf_1.tar.bz2",
        "https://anaconda.org/conda-forge/ta-lib/0.4.19/download/linux-64/ta-lib-0.4.19-py310hde88566_4.tar.bz2"
    ]

    subprocess.run(
        f"sudo curl -L {urls[0]} | tar xj -C /usr/lib/x86_64-linux-gnu/ lib --strip-components=1",
        shell=True,
        check=True
    )

    subprocess.run(
        f"sudo curl -L {urls[1]} | tar xj -C /usr/local/lib/python3.10/dist-packages/ lib/python3.10/site-packages/talib --strip-components=3",
        shell=True,
        check=True
    )

    print("TA-Lib installation complete!")
    run_scripts()