import paramiko
from scp import SCPClient
import os
from datetime import datetime

def fetch_image():
    hostname = ''
    #  hostname = '169.254.214.2'
    username = 'nao'
    password = 'nao'
    remote_path = "/home/nao/recordings/cameras/"
    local_path = 'C:/Users/User/Desktop/mw/image/'

   # Establish SSH connection
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, username=username, password=password)

    # Find all files in the remote directory via SSH
    stdin, stdout, stderr = ssh_client.exec_command(f"ls {remote_path}")
    files = stdout.read().decode().splitlines()
    errors = stderr.read().decode()

    if errors:
        print(f"Error reading remote directory: {errors}")
    else:
        print(f"Files found in remote path: {files}")

    # Download files from remote to local directory
    with SCPClient(ssh_client.get_transport()) as scp:
        for file in files:
            remote_file_path = remote_path + file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_file_path = os.path.join(local_path, f'{timestamp}.jpg')
            
            scp.get(remote_file_path, local_file_path)

    ssh_client.close()