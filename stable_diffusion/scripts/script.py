import os
import shutil
import time
import subprocess


input_dir = '/host/tmp/input'

output_dir = '/host/tmp/output'

log_dir = '/app/output/logs'

start_time = time.time()

try:
    print('Downloading model...')
    cmd = 'transmission-cli "magnet:?xt=urn:btih:3f8016061132ad4a475eb33dac569f3a88418974&dn=v1-5-pruned-emaonly.safetensors&tr=udp%3a%2f%2fopen.demonii.com%3a1337%2fannounce&tr=udp%3a%2f%2fexodus.desync.com%3a6969%2fannounce" -w /models'
    subprocess.run(cmd, shell=True, check=True)
    # Simulate data processing
    print('Processing data...')
    time.sleep(5)  # Simulate 5 seconds of data processing

    # Move result to output directory
    output_file = os.path.join(output_dir, 'output.txt')
    with open(output_file, 'w') as f:
        f.write('Processing result')
    print('Result saved at:', output_file)

except Exception as e:
    # Error logging
    with open(os.path.join(log_dir, 'error.log'), 'w') as f:
        f.write(str(e))
    print('An error occurred. Please check the log file at:', os.path.join(log_dir, 'error.log'))

finally:
    # End time and download speed logging
    end_time = time.time()
    download_speed = os.path.getsize('/models') / (end_time - start_time) / (1024 * 1024)
    with open(os.path.join(log_dir, 'stats.log'), 'w') as f:
        f.write('Start time: {}\n'.format(time.ctime(start_time)))
        f.write('End time: {}\n'.format(time.ctime(end_time)))
        f.write('Download speed (M/S): {:.2f}\n'.format(download_speed))
