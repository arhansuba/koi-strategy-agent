import os
import json
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_config(config_path: str) -> dict:
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def set_environment_variables(env_vars: dict):
    logging.info("Setting environment variables")
    for key, value in env_vars.items():
        os.environ[key] = value
        logging.info(f"Set {key} = {value}")

def install_dependencies(requirements_file: str):
    logging.info(f"Installing dependencies from {requirements_file}")
    subprocess.check_call(["pip", "install", "-r", requirements_file])

def main():
    # Define paths
    config_path = "config/config.json"
    requirements_file = "requirements.txt"

    # Load configuration
    config = load_config(config_path)

    # Set environment variables
    set_environment_variables(config.get("environment_variables", {}))

    # Install dependencies
    install_dependencies(requirements_file)

    logging.info("Environment setup completed successfully.")

if __name__ == "__main__":
    main()
