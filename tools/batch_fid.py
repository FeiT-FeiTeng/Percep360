import os
import subprocess
import shutil
import yaml
from pathlib import Path


class FIDPipeline:
    def __init__(self, config):
        """
        Initialize FIDPipeline.
        :param config: Configuration dictionary loaded from the config file.
        """
        self.weights_list = config["weights"]
        self.base_fid_path = config["base_fid_path"]
        self.task_id = config["task_id"]
        self.tasks = config["tasks"]

    def clear_folder(self, folder_path):
        """Clear a folder."""
        if os.path.exists(folder_path):
            print(f"Clearing the folder: {folder_path}")
            shutil.rmtree(folder_path)

    def execute_command(self, command):
        """Execute a system command."""
        try:
            print("Executing command:", " ".join(command))
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {' '.join(command)}")
            print(f"Error details: {e}")
            return False
        return True

    def process_task(self, task, weight_path, fid_path):
        """
        Process a single task.
        :param task: Task configuration dictionary.
        :param weight_path: Weight path.
        :param fid_path: FID path.
        """
        task_name = task["name"]
        tool_path = task["tool"]

        if task_name == "Gen_Img":
            self.clear_folder(fid_path)
            command = [
                "accelerate",
                "launch",
                tool_path,
                f"resume_from_checkpoint={weight_path}",
                f"task_id={self.task_id}",
                f"fid.img_gen_dir={fid_path}",
                "+fid=data_gen",
                f"+exp={self.task_id}",
            ]
        else:
            command = [
                "python",
                tool_path,
                "cfg",
                f"resume_from_checkpoint={weight_path}",
                f"fid.rootb={fid_path}",
            ]

        self.execute_command(command)

    def process_weight(self, weight_path):
        """Process a single weight path."""
        weight_name = os.path.basename(weight_path)
        fid_path = os.path.join(self.base_fid_path, weight_name)

        for task in self.tasks:
            print(f"Running task '{task['name']}' for weight '{weight_name}'")
            self.process_task(task, weight_path, fid_path)

    def run(self):
        """Iterate over the weight list and process each item."""
        for weight_path in self.weights_list:
            print(f"Processing weight: {weight_path}")
            self.process_weight(weight_path)


if __name__ == "__main__":
    # Load the config file.
    project_root = Path(__file__).resolve().parents[1]
    config_path = os.environ.get("PERCEP360_METRIC_CONFIG", str(project_root / "tools" / "Metric.yaml"))
    def load_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    
    config = load_config(config_path)

    # Set environment variables.
    os.environ.setdefault("PANODREAMER_PATH", str(project_root))

    # Initialize and run the pipeline.
    pipeline = FIDPipeline(config)
    pipeline.run()
