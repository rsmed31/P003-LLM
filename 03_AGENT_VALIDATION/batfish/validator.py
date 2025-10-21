from flask import Flask, request, jsonify
from pybatfish.client.session import Session

from pybatfish.datamodel.flow import PathConstraints
import os
import time
import logging

app = Flask(__name__)

# configure basic logging
logging.basicConfig(level=logging.INFO)

# Define Batfish session globals (initialized via helper below)
bf = None
bfq = None


def create_bf_session(retries: int = 6, delay: int = 5):
    """Try to create a Batfish Session with retries.

    Uses environment variables BATFISH_HOST and BATFISH_PORT if set,
    otherwise defaults to localhost:9997 (matches the docker run mapping).
    """
    global bf, bfq
    host = os.getenv("BATFISH_HOST", "localhost")
    port = int(os.getenv("BATFISH_PORT", "9997"))

    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Attempting to connect to Batfish at {host}:{port} (attempt {attempt}/{retries})")
            bf = Session(host=host, port=port)
            bfq = bf.q
            logging.info("Connected to Batfish session")
            return bf
        except Exception as e:
            logging.warning(f"Unable to create Batfish session (attempt {attempt}): {e}")
            bf = None
            bfq = None
            if attempt < retries:
                time.sleep(delay)

    logging.error("Failed to establish Batfish session after retries")
    return None


# Try creating session on import/startup but tolerate failure (container may not be up yet)
create_bf_session(retries=3, delay=3)  # short initial attempts; functions will retry if needed

# Define snapshot name and path globally
snapshot_name = "snapshot"  # unified snapshot name
SNAPSHOT_PATH = r"C:\Users\Hassa\Desktop\P003-LLM\03_AGENT_VALIDATION\batfish"

def snapshot_init():
    global bf, snapshot_name, SNAPSHOT_PATH
    # Ensure we have a Batfish session (try to create one if missing)
    if bf is None:
        logging.info("No Batfish session, attempting to create one before snapshot init...")
        create_bf_session(retries=5, delay=2)
        if bf is None:
            print("Batfish session not available, cannot init snapshot.")
            return
    try:
        NETWORK_NAME = "network-test-4routers"
        bf.set_network(NETWORK_NAME)
        # Initialize snapshot from local directory
        bf.init_snapshot(SNAPSHOT_PATH, name=snapshot_name, overwrite=True)
        print("Snapshot loaded")
    except Exception as e:
        print(f"Error during snapshot initialization: {str(e)}")


@app.route('/verify', methods=['POST'])
def verify_configuration():
    global snapshot_name  # Declare snapshot_name as global

    data = request.get_json()

    # Extract the verification details from the request
    verification_type = data.get("verification_type", "DEFAULT")
    commands = data.get("commands", [])
    identifier = data.get("identifier", "DEFAULT")  # Default identifier if not provided

    # Apply the received configuration to Batfish (if applicable)
    try:
        if verification_type == "APPLY_CONFIG":
            hostname = data.get("hostname", "")
            # Apply received commands as a single file for the snapshot
            if not bf:
                raise RuntimeError("Batfish session not initialized")
            bf.init_snapshot(input_text={f"{hostname}.cfg": "\n".join(commands)}, name=snapshot_name, overwrite=True)
    except Exception as e:
        return jsonify({"result": "Error", "error": str(e)})

    # Proceed with the selected verification
    try:
        # Determine the type of Batfish question based on the identifier
        if identifier == "CP":
            # Configuration Properties question
            result = bfq.interfaceProperties().answer()
            resp = {"check": "CP", "frame": result.frame().to_dict() if not result.frame().empty else {}}
            return jsonify({"result": "CP completed", "cp": resp})

        elif identifier == "TP":
            # Topology question
            result = bfq.layer3Topology().answer()
            resp = {"check": "TP", "frame": result.frame().to_dict() if not result.frame().empty else {}}
            return jsonify({"result": "TP completed", "tp": resp})

        elif identifier == "ALL":
            # Run CP, TP and reachability sequentially and aggregate results
            aggregated = {}
            try:
                cp_ans = bfq.interfaceProperties().answer()
                aggregated['CP'] = cp_ans.frame().to_dict() if not cp_ans.frame().empty else {}
            except Exception as e:
                aggregated['CP_error'] = str(e)

            try:
                tp_ans = bfq.layer3Topology().answer()
                aggregated['TP'] = tp_ans.frame().to_dict() if not tp_ans.frame().empty else {}
            except Exception as e:
                aggregated['TP_error'] = str(e)

            try:
                reach_ans = bfq.reachability(
                    pathConstraints=PathConstraints(startLocation="enter(h1)", endLocation="enter(h2)")
                ).answer()
                aggregated['REACH'] = reach_ans.frame().to_dict() if not reach_ans.frame().empty else {}
            except Exception as e:
                aggregated['REACH_error'] = str(e)

            return jsonify({"result": "ALL completed", "aggregated": aggregated})

        else:
            # Default to reachability check
            result = bfq.reachability(
                pathConstraints=PathConstraints(startLocation="enter(h1)", endLocation="enter(h2)")
            ).answer()
            if result.frame().empty:
                return jsonify({"result": "Successful"})
            else:
                return jsonify({"result": "Verification failed", "details": result.frame().to_dict()})
    except Exception as e:
        # Print the specific error message
        print(f"Error during verification: {str(e)}")
        return jsonify({"result": "Error during verification", "error": str(e)})


@app.route('/syntax', methods=['POST'])
def syntax_checker_route():
    data = request.get_json()

    # Extract the configuration commands from the request
    commands = data.get("commands", [])

    # Perform syntax check (ensure function exists)
    try:
        if 'syntax_checker' not in globals():
            logging.info('No syntax_checker found, using fallback stub')
            # fallback stub implemented below will be used
        syntax_check_result = syntax_checker(commands)
        return jsonify(syntax_check_result)
    except Exception as e:
        return jsonify({"result": "Error", "error": str(e)})


@app.route('/llm_response', methods=['POST'])
def apply_llm_response():
    data = request.get_json()

    # Extract the LLM response details from the request
    device_name = data.get("device_name", "")
    llm_response = data.get("llm_response", [])

    try:
        # Apply the received LLM response to Batfish
        # Use the configured SNAPSHOT_PATH by default (fallback to provided path param)
        spath = SNAPSHOT_PATH
        apply_config_to_device(spath, device_name, "\n".join(llm_response))
        return jsonify({"result": "Successful"})
    except Exception as e:
        return jsonify({"result": "Error during LLM response application", "error": str(e)})


def apply_config_to_device(snapshot_path, device_name, llm_response):
    # existing helper functions (read_snapshot_config, update_existing_config, generate_cisco_config)
    # Defensive: ensure helper functions exist (stubs provided below)
    existing_config = read_snapshot_config(snapshot_path, device_name)
    new_cfg_piece = generate_cisco_config(llm_response, device_name)
    new_config = update_existing_config(existing_config, new_cfg_piece)

    # Ensure Batfish session exists
    if not bf:
        logging.info("Batfish session missing in apply_config_to_device, attempting to create one...")
        create_bf_session(retries=5, delay=2)
    if not bf:
        raise RuntimeError("Batfish session not initialized")

    # send the updated config as a single file into the snapshot
    bf.init_snapshot(input_text={f"{device_name}.cfg": new_config}, name=snapshot_name, overwrite=True)



@app.route('/topology', methods=['GET'])
def get_topology():
    try:
        frame = bf.q.layer3Edges().answer().frame()
        return jsonify({"topology": frame.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)})


def read_config_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


# ---------------------------
# Development stubs (replace with real implementations)
# ---------------------------
def syntax_checker(commands):
    """Simple syntax checker stub.

    Replace this with a real parser/validator. For now it checks for empty input
    and returns a dict describing basic findings.
    """
    if not commands:
        return {"ok": False, "errors": ["No commands provided"]}
    # naive check: ensure every command ends with a newline or looks like text
    return {"ok": True, "errors": []}


def read_snapshot_config(snapshot_path, device_name):
    """Read the existing device config from a snapshot directory if present.

    This stub looks for a file named <device_name>.cfg in snapshot_path and
    returns its text, or an empty string if missing.
    """
    cfg_file = os.path.join(snapshot_path, f"{device_name}.cfg")
    try:
        return read_config_file(cfg_file)
    except Exception:
        return ""


def update_existing_config(existing_config: str, new_piece: str) -> str:
    """Merge existing_config with new_piece. This stub appends the new piece.

    A realistic implementation should merge intelligently.
    """
    if not existing_config:
        return new_piece
    return existing_config + "\n" + new_piece


def generate_cisco_config(llm_response: str, device_name: str) -> str:
    """Convert an LLM response (text) into a Cisco-style config snippet.

    This stub simply returns the raw response.
    """
    return llm_response



if __name__ == '__main__':
    snapshot_init()
    app.run(host='0.0.0.0', port=5000) # Anounce the application on every IP or change batfish_ip to the batfish docker ip