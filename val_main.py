import os
import subprocess
from pathlib import Path

def main():
    """
    Main function that wraps val_set_gen.py.
    """
    # Default parameters.
    project_root = Path(__file__).resolve().parent
    resume_from_checkpoint = os.environ.get(
        "PERCEP360_CHECKPOINT",
        str(project_root / "pretrained" / "SDv1.5mv-rawbox_2023-09-07_18-39_224x400"),
    )
    task_id = "224x400"
    fid_img_gen_dir = os.environ.get(
        "PERCEP360_FID_DIR",
        str(project_root / "magicdrive-log" / "img_fid"),
    )
    fid = "data_gen"
    exp = "224x400"

    # Environment variables.
    os.environ.setdefault("PANODREAMER_PATH", str(project_root))
    print(f"Environment variable PANODREAMER_PATH set to: {os.environ['PANODREAMER_PATH']}")

    # Build the command.
    command = [
        "python",
        str(project_root / "perception" / "data_prepare" / "val_set_gen.py"),
        f"resume_from_checkpoint={resume_from_checkpoint}",
        f"task_id={task_id}",
        f"fid.img_gen_dir={fid_img_gen_dir}",
        f"+fid={fid}",
        f"+exp={exp}"
    ]
    
    # Print the command for debugging.
    print("Running command:", " ".join(command))
    
    # Execute the command.
    try: 
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Return code: {e.returncode}")

if __name__ == "__main__":
    main()
