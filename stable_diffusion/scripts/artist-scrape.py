from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

# Configure ChromeOptions to ignore SSL errors
chrome_options = Options()
chrome_options.add_argument('--ignore-certificate-errors')

# Instantiate the Chrome WebDriver with the configured options
driver = webdriver.Chrome(chrome_options=chrome_options)

driver.get('http://adityashankar.xyz/')

input_field = driver.find_element(By.CSS_SELECTOR, 'input.input.input-bordered.w-full.text-center')
input_field.clear()
input_field.send_keys(' ')  # Enter the value you want to input to trigger the dynamic changes

time.sleep(1)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# Example: Extracting artist names from a drop-down
artist_buttons = soup.find_all('button', class_='artist-result')
artists = [button.text for button in artist_buttons]

pre_prompt_header = soup.find('h1', class_='text-s sm:text-xl md:text-3xl text-center mx-3 mt-4 md:mt-0')
pre_prompt = pre_prompt_header.text.strip()

# Save artists to txt file
file_path = 'artists.txt'
with open(file_path, 'w') as file:
    for artist in artists:
        file.write(artist + '\n')

print(f'Saved data to: {file_path}')

driver.quit()
