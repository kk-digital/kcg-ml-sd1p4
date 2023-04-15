import os
import datetime
import time
import requests
import libtorrent as lt

if not os.path.exists('/tmp/output'):
    os.makedirs('/tmp/output')

if not os.path.exists('/tmp/logs'):
    os.makedirs('/tmp/logs')

magnet_link = 'magnet:?xt=urn:btih:3f8016061132ad4a475eb33dac569f3a88418974&dn=v1-5-pruned-emaonly.safetensors&tr=udp%3a%2f%2fopen.demonii.com%3a1337%2fannounce&tr=udp%3a%2f%2fexodus.desync.com%3a6969%2fannounce'
destination = '/tmp/input/models/'
output_dir = '/tmp/output'
logs_dir = '/tmp/logs'

try:
    ses = lt.session()
    h = lt.add_magnet_uri(ses, magnet_link, {'save_path': destination})

    print("Downloading magnet link...")

    while not h.is_seed():
        s = h.status()
        print(f"Progress: {s.progress * 100:.2f}%")
        time.sleep(1)

    print(f"Download complete. File saved to: {destination}/{h.name()}")
except Exception as e:
    print(f"Failed to download magnet link: {e}")

try:
    output_file = os.path.join(output_dir, 'output.txt')
    with open(output_file, 'w') as file:
        file.write("Hello, output!")
    print(f"Output saved to: {output_file}")
except Exception as e:
    print(f"Failed to save output: {e}")

try:
    log_file = os.path.join(
        logs_dir,
        f'log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(log_file, 'w') as file:
        file.write("This is a log entry.")
    print(f"Log saved to: {log_file}")
except Exception as e:
    print(f"Failed to save log: {e}")

