import os
import urllib.request
import subprocess
import threading
import time
import json
import urllib
import uuid
import websocket
import random
import requests
import shutil
import custom_node_helpers as helpers

from pathlib import Path
from node import Node
from weights_downloader import WeightsDownloader
from urllib.error import URLError


class ComfyUI:
    def __init__(self, server_address):
        self.weights_downloader = WeightsDownloader()
        self.server_address = server_address
        self.server_process = None
        self.comfyui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI")

    def install_comfyui(self):
        """Install ComfyUI if not already installed"""
        if not os.path.exists(self.comfyui_path):
            print("ComfyUI not found. Installing...")
            try:
                # Clone ComfyUI repository
                subprocess.run(
                    ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", self.comfyui_path],
                    check=True
                )
                print("ComfyUI installed successfully")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install ComfyUI: {e}")

    def start_server(self, output_directory, input_directory):
        """Start the ComfyUI server"""
        self.input_directory = input_directory
        self.output_directory = output_directory
        
        # Ensure ComfyUI is installed
        self.install_comfyui()
        
        # Create necessary directories
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(input_directory, exist_ok=True)
        
        self.apply_helper_methods("prepare", weights_downloader=self.weights_downloader)

        start_time = time.time()
        
        # Start server in a thread
        server_thread = threading.Thread(
            target=self.run_server,
            args=(output_directory, input_directory)
        )
        server_thread.daemon = True  # Make thread daemon so it exits when main program exits
        server_thread.start()

        # Wait for server to start
        while not self.is_server_running():
            if time.time() - start_time > 60:
                if self.server_process:
                    self.server_process.terminate()
                raise TimeoutError("Server did not start within 60 seconds. Check if port 8188 is available and ComfyUI dependencies are installed.")
            time.sleep(0.5)

        elapsed_time = time.time() - start_time
        print(f"Server started in {elapsed_time:.2f} seconds")

    def run_server(self, output_directory, input_directory):
        """Run the ComfyUI server process"""
        try:
            command = [
                "python",
                os.path.join(self.comfyui_path, "main.py"),
                "--output-directory", output_directory,
                "--input-directory", input_directory,
                "--disable-metadata",
                "--highvram",
                "--listen", "0.0.0.0"  # Listen on all interfaces
            ]
            
            self.server_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=self.comfyui_path  # Set working directory to ComfyUI path
            )
            
            # Monitor both stdout and stderr
            while True:
                stdout_line = self.server_process.stdout.readline()
                stderr_line = self.server_process.stderr.readline()
                
                if stdout_line:
                    print("ComfyUI:", stdout_line.strip())
                if stderr_line:
                    print("ComfyUI Error:", stderr_line.strip())
                    
                if self.server_process.poll() is not None:
                    remaining_stderr = self.server_process.stderr.read()
                    if remaining_stderr:
                        print("ComfyUI Error:", remaining_stderr.strip())
                    if self.server_process.returncode != 0:
                        raise RuntimeError(f"ComfyUI server failed to start with return code {self.server_process.returncode}")
                    break
                    
        except Exception as e:
            print(f"Error running ComfyUI server: {e}")
            if self.server_process:
                self.server_process.terminate()
            raise

    def stop_server(self):
        """Stop the ComfyUI server process"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None

    def cleanup_directories(self, directories):
        """Clean up the specified directories"""
        self.clear_queue()
        for directory in directories:
            if directory == self.input_directory:
                subject_path = os.path.join(directory, "subject.png")
                if os.path.exists(subject_path):
                    os.remove(subject_path)
            else:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                    os.makedirs(directory)

    def cleanup(self):
        """Clean up all resources"""
        self.stop_server()
        if hasattr(self, 'output_directory') and hasattr(self, 'input_directory'):
            self.cleanup_directories([self.output_directory, self.input_directory])

    def __del__(self):
        """Ensure cleanup when object is destroyed"""
        self.cleanup()

    def is_server_running(self):
        try:
            with urllib.request.urlopen(
                "http://{}/history/{}".format(self.server_address, "123")
            ) as response:
                return response.status == 200
        except URLError:
            return False

    def apply_helper_methods(self, method_name, *args, **kwargs):
        # Dynamically applies a method from helpers module with given args.
        # Example usage: self.apply_helper_methods("add_weights", weights_to_download, node)
        for module_name in dir(helpers):
            module = getattr(helpers, module_name)
            method = getattr(module, method_name, None)
            if callable(method):
                method(*args, **kwargs)

    def handle_weights(self, workflow, weights_to_download=[]):
        print("Checking weights")
        embeddings = self.weights_downloader.get_weights_by_type("EMBEDDINGS")
        embedding_to_fullname = {emb.split(".")[0]: emb for emb in embeddings}
        weights_filetypes = self.weights_downloader.supported_filetypes

        for node in workflow.values():
            self.apply_helper_methods("add_weights", weights_to_download, Node(node))

            for input in node["inputs"].values():
                if isinstance(input, str):
                    if any(key in input for key in embedding_to_fullname):
                        weights_to_download.extend(
                            embedding_to_fullname[key]
                            for key in embedding_to_fullname
                            if key in input
                        )
                    elif any(input.endswith(ft) for ft in weights_filetypes):
                        weights_to_download.append(input)

        weights_to_download = list(set(weights_to_download))

        for weight in weights_to_download:
            self.weights_downloader.download_weights(weight)
            print(f"✅ {weight}")

        print("====================================")

    def is_image_or_video_value(self, value):
        filetypes = [".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm"]
        return isinstance(value, str) and any(
            value.lower().endswith(ft) for ft in filetypes
        )

    def handle_known_unsupported_nodes(self, workflow):
        for node in workflow.values():
            self.apply_helper_methods("check_for_unsupported_nodes", Node(node))

    def handle_inputs(self, workflow):
        print("Checking inputs")
        seen_inputs = set()
        for node in workflow.values():
            if "inputs" in node:
                for input_key, input_value in node["inputs"].items():
                    if isinstance(input_value, str) and input_value not in seen_inputs:
                        seen_inputs.add(input_value)
                        if input_value.startswith(("http://", "https://")):
                            filename = os.path.join(
                                self.input_directory, os.path.basename(input_value)
                            )
                            if not os.path.exists(filename):
                                print(f"Downloading {input_value} to {filename}")
                                try:
                                    response = requests.get(input_value)
                                    response.raise_for_status()
                                    with open(filename, "wb") as file:
                                        file.write(response.content)
                                    node["inputs"][input_key] = filename
                                    print(f"✅ {filename}")
                                except requests.exceptions.RequestException as e:
                                    print(f"❌ Error downloading {input_value}: {e}")

                        elif self.is_image_or_video_value(input_value):
                            filename = os.path.join(
                                self.input_directory, os.path.basename(input_value)
                            )
                            if not os.path.exists(filename):
                                print(f"❌ {filename} not provided")
                            else:
                                print(f"✅ {filename}")

        print("====================================")

    def connect(self):
        self.client_id = str(uuid.uuid4())
        self.ws = websocket.WebSocket()
        self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")

    def post_request(self, endpoint, data=None):
        url = f"http://{self.server_address}{endpoint}"
        headers = {"Content-Type": "application/json"} if data else {}
        json_data = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(
            url, data=json_data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                print(f"Failed: {endpoint}, status code: {response.status}")

    # https://github.com/comfyanonymous/ComfyUI/blob/master/server.py
    def clear_queue(self):
        self.post_request("/queue", {"clear": True})
        self.post_request("/interrupt")

    def queue_prompt(self, prompt):
        try:
            # Prompt is the loaded workflow (prompt is the label comfyUI uses)
            p = {"prompt": prompt, "client_id": self.client_id}
            data = json.dumps(p).encode("utf-8")
            req = urllib.request.Request(
                f"http://{self.server_address}/prompt?{self.client_id}", data=data
            )

            output = json.loads(urllib.request.urlopen(req).read())
            return output["prompt_id"]
        except urllib.error.HTTPError as e:
            print(f"ComfyUI error: {e.code} {e.reason}")
            http_error = True

        if http_error:
            raise Exception(
                "ComfyUI Error – Your workflow could not be run. This usually happens if you're trying to use an unsupported node. Check the logs for 'KeyError: ' details, and go to https://github.com/fofr/cog-comfyui to see the list of supported custom nodes."
            )

    def wait_for_prompt_completion(self, workflow, prompt_id):
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break
                    elif data["prompt_id"] == prompt_id:
                        node = workflow.get(data["node"], {})
                        meta = node.get("_meta", {})
                        class_type = node.get("class_type", "Unknown")
                        print(
                            f"Executing node {data['node']}, title: {meta.get('title', 'Unknown')}, class type: {class_type}"
                        )
            else:
                continue

    def load_workflow(self, workflow):
        if not isinstance(workflow, dict):
            wf = json.loads(workflow)
        else:
            wf = workflow

        # There are two types of ComfyUI JSON
        # We need the API version
        if any(key in wf.keys() for key in ["last_node_id", "last_link_id", "version"]):
            raise ValueError(
                "You need to use the API JSON version of a ComfyUI workflow. To do this go to your ComfyUI settings and turn on 'Enable Dev mode Options'. Then you can save your ComfyUI workflow via the 'Save (API Format)' button."
            )

        self.handle_known_unsupported_nodes(wf)
        self.handle_inputs(wf)
        self.handle_weights(wf)
        return wf

    def reset_execution_cache(self):
        print("Resetting execution cache")
        with open("reset.json", "r") as file:
            reset_workflow = json.loads(file.read())
        self.queue_prompt(reset_workflow)

    def randomise_input_seed(self, input_key, inputs):
        if input_key in inputs and isinstance(inputs[input_key], (int, float)):
            new_seed = random.randint(0, 2**32 - 1)
            print(f"Randomising {input_key} to {new_seed}")
            inputs[input_key] = new_seed

    def randomise_seeds(self, workflow):
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            seed_keys = ["seed", "noise_seed", "rand_seed"]
            for seed_key in seed_keys:
                self.randomise_input_seed(seed_key, inputs)

    def run_workflow(self, workflow):
        print("Running workflow")
        prompt_id = self.queue_prompt(workflow)
        self.wait_for_prompt_completion(workflow, prompt_id)
        output_json = self.get_history(prompt_id)
        print("outputs: ", output_json)
        print("====================================")

    def get_history(self, prompt_id):
        with urllib.request.urlopen(
            f"http://{self.server_address}/history/{prompt_id}"
        ) as response:
            output = json.loads(response.read())
            return output[prompt_id]["outputs"]

    def get_files(self, directories, prefix=""):
        files = []
        if isinstance(directories, str):
            directories = [directories]

        for directory in directories:
            for f in os.listdir(directory):
                if f == "__MACOSX":
                    continue
                path = os.path.join(directory, f)
                if os.path.isfile(path):
                    print(f"{prefix}{f}")
                    files.append(Path(path))
                elif os.path.isdir(path):
                    print(f"{prefix}{f}/")
                    files.extend(self.get_files(path, prefix=f"{prefix}{f}/"))

        return sorted(files)
