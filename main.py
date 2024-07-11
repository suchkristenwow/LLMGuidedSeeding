import toml
import subprocess
import os
import argparse
import signal
import threading
import logging
import uuid
import shutil
import numpy as np 
import os
import time

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        with open(self.args.config, "r") as f:
            self.settings = toml.load(f)
        self.running_processes = []
        self.success = False
        self.logging_file = ""
        self.shutdown_event = threading.Event()
        self.policyGeneration_process = None
        self.policyRehearsal_process = None 
        self.policyExecution_process = None 
        self.flask_backend = None
        self.react_frontend = None
        
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        self.logging_dir = self.settings["logging_directory"] 
                
        self.vqa_url = f"http://{self.settings['common_ip']}:{self.settings['vqa_model']['port']}/completion"

    '''
    def create_logging_directory(self):
        """
        Creates the experiment log directory specified in the settings.

        Returns:
            None
        """
        logging_dir = self.logging_dir
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Attempting to create directory: {logging_dir}")
        try:
            os.makedirs(logging_dir, exist_ok=True)
            os.makedirs(logging_dir + "/application_logs", exist_ok=True)
            full_path = os.path.abspath(logging_dir)
            logging.info(f"Created experiment directory at {full_path} \n")
        except PermissionError as e:
            logging.error(f"Permission error: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
    '''

    def copy_logs_and_data(self, base_destination_dir):
        """
        Copies the logs and data from the experiment directories to a specific subdirectory
        within the specified base destination directory, based on the experiment outcome.

        Args:
            base_destination_dir (str): The base destination directory where logs and data should be copied.
        """
        # Determine the subdirectory based on the experiment outcome
        outcome_subdir = "success" if self.success else "failure"
        destination_dir = os.path.join(base_destination_dir, outcome_subdir)

        try:
            # Ensure the destination directory exists
            os.makedirs(destination_dir, exist_ok=True)
            # Copy the entire logging directory
            logging.info(f"Coppying logs and data to {destination_dir}")
            shutil.copytree(self.logging_dir, os.path.join(destination_dir, os.path.basename(self.logging_dir)), dirs_exist_ok=True)
            logging.info(f"Copied logs and data to {destination_dir}")
        except Exception as e:
            logging.error(f"Failed to copy logs and data: {e}")

    def load_prompt(self):
        with open(self.settings["prompt_file"], "r") as file:
            self.prompt_text = file.read().strip()
        if not self.prompt_text:
            raise ValueError("Prompt text is empty.")

    def start_process_with_terminal(self, launch_command, process_name, cwd=None):
        try:
            log_file_path = os.path.join(
                self.logging_dir + "/application_logs",
                f"{process_name}.log",
            )
            
            log_file_path_abs = os.path.abspath(log_file_path)
                        
            # Combine the launch command into a single string if it's a list
            if isinstance(launch_command, list):
                launch_command_str = " ".join(launch_command)
            else:
                launch_command_str = launch_command
            # Use tee to duplicate output to both logfile and terminal
            tee_command = f"{launch_command_str} | tee '{log_file_path_abs}'"
            
            process = subprocess.Popen(
                tee_command, shell=True, env=os.environ, cwd=cwd, preexec_fn=os.setsid
            )
            logging.info(
                f"Process '{process_name}' launched with command: {tee_command} \n"
            )

        except OSError as e:
            logging.error(f"Error launching process '{process_name}': {e} \n")

        return process 
    
    def start_process_monitoring(self):
        """
        Starts a separate thread that monitors all running processes.
        If a process exits unexpectedly, an exception is raised.
        """

        def monitor():
            while not self.shutdown_event.is_set():
                for process in self.running_processes:
                    if process.poll() is not None:  # Process has exited
                        # Optionally, log the exit code if needed
                        exit_code = process.returncode
                        logging.error(
                            f"Process {process.args} exited prematurely with exit code {exit_code}."
                        )
                        self.success = False
                        time.sleep(5)
                        self.shutdown_event.set()  # Set the shutdown event
                        raise Exception(
                            f"Process {process.args} died unexpectedly with exit code {exit_code}."
                        )
                self.shutdown_event.wait(1)  # Check every second

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

    def terminate_all_processes(self):
        """
        Terminates all running subprocesses gracefully. If a subprocess doesn't
        terminate within the timeout, it's forcefully killed.
        """

        '''
        if self.explore_process is not None:
            logging.info("Terminating explore process")
            os.killpg(os.getpgid(self.explore_process.pid), signal.SIGTERM)
            self.explore_process = None
        '''
        print(self.running_processes)
        for process in self.running_processes:
            logging.info(f"Terminating process {process.pid}...")
            if process.poll() is None:  # Check if the process is still running
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                # process.terminate()  # Attempt to terminate it gracefully
                # try:
                #     process.wait(timeout=5)  # Give it some time to terminate
                # except subprocess.TimeoutExpired:
                #     process.kill()  # Force termination if it doesn't respond
                #     process.wait()  # Wait for the killing to complete
                # logging.info(f"Process {process.pid} terminated.")

        # manually shutdown the servers
        print(f'Terminating backend PID: {self.flask_backend.pid}')
        os.killpg(os.getpgid(self.flask_backend.pid), signal.SIGTERM) #terminate gracefully, signal.SIGKILL : kill forcefully
        print(f'Terminating frontend PID: {self.react_frontend.pid} \n')
        os.killpg(os.getpgid(self.react_frontend.pid), signal.SIGTERM)

    
    def launch_policy_gen(self):
        logs_dir = os.path.join(self.logging_dir, "policy_generation_logs")
        logs_dir_absolute = os.path.abspath(logs_dir)
        print("Explore logs dir", logs_dir_absolute)
        os.makedirs(logs_dir_absolute, exist_ok=True)
        prompt_path = os.path.abspath(self.args.prompt)
        config_path = os.path.abspath(self.args.config)
        bounds_path = os.path.abspath(self.args.plot_bounds_path)
        print("this is config path: ",config_path)
        print()
        # Split the command into a list of arguments
        #TO DO: LOAD IN PLOT BOUNDS FROM USER INTERFACE 
        launch_command = [
            "python3", "generate_policies.py",
            "--prompt_path", prompt_path,
            "--config_path", config_path,
            "--logging_dir", logs_dir_absolute,
            "--plot_bounds", bounds_path
        ]
        
        #explore_runner_dir = os.path.abspath(os.path.join("../"))
        self.policyGeneration_process = self.start_process_with_terminal(launch_command, "generate_policies", cwd=os.getcwd())

    def launch_policy_rehearsal(self,policy): 
        logs_dir = os.path.join(self.logging_dir, "policy_rehearsal_logs")
        logs_dir_absolute = os.path.abspath(logs_dir)
        print("Explore logs dir", logs_dir_absolute)
        print()
        os.makedirs(logs_dir_absolute, exist_ok=True)
        
        config_path = os.path.abspath(self.args.config)
        # Split the command into a list of arguments
        launch_command = [
            "python3", "rehearse_policies.py",
            "--policy", policy,
            "--config_path", config_path,
            "--logging_dir", logs_dir_absolute
        ]
        
        #explore_runner_dir = os.path.abspath(os.path.join("../"))
        self.policyRehearsal_process = self.start_process_with_terminal(launch_command, "rehearse_policies", cwd=os.getcwd())

    def launch_policy_execution(self,policy):
        logs_dir = os.path.join(self.logging_dir, "policy_execution_logs")
        logs_dir_absolute = os.path.abspath(logs_dir)
        print(f'Explore logs dir {logs_dir_absolute} \n')
        
        os.makedirs(logs_dir_absolute, exist_ok=True)
        
        config_path = os.path.abspath(self.args.config)
        # Split the command into a list of arguments
        launch_command = [
            "python3", "execute_policy.py",
            "--policy", policy,
            "--config_path", config_path,
            "--logging_dir", logs_dir_absolute
        ]
        
        #explore_runner_dir = os.path.abspath(os.path.join("../"))
        self.policyExecution_process = self.start_process_with_terminal(launch_command, "execute_policy", cwd=os.getcwd())

    def launch_flask_app(self):
        log_file_path = os.path.join(self.logging_dir,"application_logs", "flask_server.log")
        log_file_path_abs = os.path.abspath(log_file_path)
        print(f'Explore flask log file: {log_file_path_abs} \n')
        
        app_dir = os.path.join("UI_pkg", "UserInterface", "backend")
        app_path = os.path.join(app_dir, "application.py")

        launch_command = [
            "python3", app_path,  # Use python3 instead of python
            "--logging_file", log_file_path_abs
        ]
        
        self.flask_backend = self.start_process_with_terminal(launch_command, "flask_server", cwd=os.getcwd())
        time.sleep(1) # wait for the flask app to launch

    '''
    def launch_flask_app(self):
        log_file_path = os.path.join(self.logging_dir,"application_logs", "flask_server.log")
        log_file_path_abs = os.path.abspath(log_file_path)
        print(f'Explore flask log file: {log_file_path_abs} \n')
        
        app_dir = os.path.join("UI_pkg", "UserInterface", "backend")
        app_path = os.path.join(app_dir, "application.py")

        launch_command = [
            "python", app_path,
            "--logging_file", log_file_path_abs
        ]
        
        self.flask_backend = self.start_process_with_terminal(launch_command, "flask_server", cwd = os.getcwd())
        time.sleep(1) # wait for the flask app to launch
    '''

    def launch_react(self):
        log_file_path = os.path.join(self.logging_dir, "application_logs", "react_server.log")
        log_file_path_abs = os.path.abspath(log_file_path)
        print(f'Explore react log file: {log_file_path_abs} \n')

        app_dir = os.path.join("UI_pkg", "UserInterface", "frontend", "app")
        #os.chdir(app_dir)  # Change directory to the React app directory

        launch_command = ["npm", "start"]
        
        self.react_frontend = self.start_process_with_terminal(launch_command, "react_server", cwd = app_dir)
        
    def run(self):
        
        # Launch Servers
        self.launch_flask_app()
        self.launch_react()

        #launch the robot 
        self.launch_policy_gen()
        #rehearsed_policy = self.launch_policy_rehearsal(policy)
        '''
        self.launch_policy_execution(rehearsed_policy)
        self.start_process_monitoring()
        '''
        
        try:
            while not self.shutdown_event.is_set():
                self.shutdown_event.wait(timeout=1)
        finally:
            # Cleanup code here (e.g., terminating running processes)
            self.terminate_all_processes()
            if self.args.log_dir != "None":
                self.copy_logs_and_data(self.args.log_dir)
            logging.info("Cleaning up and exiting...")

def setup_signal_handling(runner):
    def signal_handler(sig, frame):
        logging.info("Ctrl+C detected. Setting shutdown event...")
        runner.shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # os and user agnostic edits to TOML file
    config_path = os.path.join("configs","example_config.toml")
    config = toml.load(config_path)
    config['prompt_file'] = os.path.join(os.path.expanduser("~"), config['prompt_file'])
    config['logging_directory'] = os.path.join(os.path.expanduser("~"), config['logging_directory'])
    config['commonObj_path'] = os.path.join(os.path.expanduser("~"), config['commonObj_path'])

    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument(
        "--config",
        default="configs/example_config.toml",
        help="Path to the TOML configuration file",
    )

    parser.add_argument(
        "--plot_bounds_path",
        default="random_path.csv",
        help="The path to the csv file where the plot bounds (in the robot frame) are saved",
    )

    parser.add_argument(
        "--prompt",
        default="prompts/ex_query.txt",
        help="Path to the desired query",
    )

    parser.add_argument(
        "--log_dir",
        default="None",
        help="The directory to save the experiment logs",
    )

    args = parser.parse_args()

    runner = ExperimentRunner(args)  # Pass the config file from arguments
    setup_signal_handling(runner)
    runner.run()