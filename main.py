from accelerate import Accelerator
import subprocess

def main():
    # Initialize the accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        cpu=False,
        split_batches=False,
        deepspeed_plugin=None,
        fsdp_plugin=None
    )
    
    # Setting up environment variables for accelerate
    import os
    os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16'
    os.environ['ACCELERATE_GPU_IDS'] = 'all'
    os.environ['ACCELERATE_NUM_PROCESSES'] = '1'

    # Define the command and arguments
    command = [
        'python', 'Command/train.py',
        '+exp=224x400',
        'runner=debuggg'
        #"runner=resume"
    ]
    
    # Use subprocess to run the command
    result = subprocess.run(command, env=os.environ)

    # Check the result
    if result.returncode != 0:
        print("Training failed")
    else:
        print("Training completed successfully")

if __name__ == "__main__":
    main()