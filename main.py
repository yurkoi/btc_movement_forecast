import subprocess
import datetime

def run_scripts():
    try:
        print(f"Starting data collection at {datetime.datetime.now()}")
        subprocess.run(["python3", "./data_collector/main.py"], check=True)
        print(f"Starting training process at {datetime.datetime.now()}")
        subprocess.run(["python3", "./modeling/train_process.py"], check=True)
        print("Scripts executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_scripts()