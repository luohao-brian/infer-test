import sys
import subprocess

def main():
    """
    Wrapper for vllm.entrypoints.openai.api_server.
    Passes all command line arguments directly to the vLLM module.
    """
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + sys.argv[1:]
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Server exited with error: {e}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()

