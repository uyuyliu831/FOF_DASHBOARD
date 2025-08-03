import os
import sys
import streamlit.web.bootstrap as bootstrap
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0  # Returns True if in use

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "FOF_DASHBOARD.py")
    desired_port = 8502

    if is_port_in_use(desired_port):
        print(f"Error: Port {desired_port} is already in use. Please free it or choose another port.")
        sys.exit(1) # Exit with error code

    os.environ["STREAMLIT_SERVER_PORT"] = str(desired_port)

    try:
        bootstrap.run(script_path, 0, sys.argv[1:], flag_options={})
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)