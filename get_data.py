import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def download_data():
    data_dir = os.path.abspath('data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    prefs = {
        'download.default_directory': data_dir,
        'download.prompt_for_download': False,
        'download.directory_upgrade': True,
        'safebrowsing.enabled': True
    }
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 10)

    try:
        url = 'https://atlas.hks.harvard.edu/data-downloads'
        driver.get(url)
        
        input("Filter down the datasets you want to download (click the table headings to filter), then press Enter to continue...")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.MuiTable-root')))
        
        page_num = 1
        downloaded_datasets = []
        while True:
            print(f"Processing page {page_num}...")
            time.sleep(1)
            
            download_buttons = driver.find_elements(By.CSS_SELECTOR, 'button[type="button"] svg[viewBox="0 0 24 24"] path[d="M5 20h14v-2H5zM19 9h-4V3H9v6H5l7 7z"]')
            download_buttons = [btn.find_element(By.XPATH, '../..') for btn in download_buttons]
            print(f"Found {len(download_buttons)} datasets on page {page_num}")
            
            # Process each dataset on the current page
            for i, button in enumerate(download_buttons):
                try:
                    print(f"  Processing dataset {i+1}/{len(download_buttons)}...")
                    driver.execute_script("arguments[0].click();", button)
                    modal = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]')))

                    dataset_info = extract_dataset_info(driver, modal)
                    dataset_file_name = dataset_info['filename']
                    
                    dataset_path = os.path.join(data_dir, dataset_file_name)

                    if os.path.exists(dataset_path):
                        print(f"    Dataset '{dataset_file_name}' already exists. Skipping download.")
                        downloaded_datasets.append(dataset_info)
                        
                        # Close modal and continue
                        close_btn = modal.find_element(By.CSS_SELECTOR, 'button svg[viewBox="0 0 24 24"] path[d*="19 6.41"]')
                        close_btn = close_btn.find_element(By.XPATH, '../..')
                        driver.execute_script("arguments[0].click();", close_btn)
                        wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]')))
                        continue

                    # If feature description exists, it will be overwritten.
                    # If dataset does not exist, download it.
                    save_feature_description(driver, modal, dataset_file_name, data_dir)
                    
                    final_download_button = modal.find_element(By.CSS_SELECTOR, "div[aria-labelledby] button.css-rp01fk")
                    final_download_button.click()
                    
                    time.sleep(3)
                    
                    downloaded_datasets.append(dataset_info)
                    print(f"    Downloaded: {dataset_info['name']}")
                    
                    # Close modal by clicking the X button
                    close_btn = modal.find_element(By.CSS_SELECTOR, 'button svg[viewBox="0 0 24 24"] path[d*="19 6.41"]')
                    close_btn = close_btn.find_element(By.XPATH, '../..')
                    driver.execute_script("arguments[0].click();", close_btn)
                    
                    # Wait for modal to close
                    wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]')))
                    
                except Exception as e:
                    print(f"    Error processing dataset {i+1}: {str(e)}")
                    # Try to close modal if it's still open
                    try:
                        close_btn = driver.find_element(By.CSS_SELECTOR, 'div[role="dialog"] button svg[viewBox="0 0 24 24"] path[d*="19 6.41"]')
                        close_btn = close_btn.find_element(By.XPATH, '../..')
                        driver.execute_script("arguments[0].click();", close_btn)
                        wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]')))
                    except:
                        pass
                    continue
            
            # Check if there's a next page
            if not go_to_next_page(driver, wait):
                break
            
            page_num += 1
        
        # Save summary of downloaded datasets
        if downloaded_datasets:
            summary_df = pd.DataFrame(downloaded_datasets)
            summary_df.to_csv(os.path.join(data_dir, 'downloaded_datasets_summary.csv'), index=False)
            print(f"\nDownload complete! Downloaded {len(downloaded_datasets)} datasets.")
            print(f"Summary saved to: {os.path.join(data_dir, 'downloaded_datasets_summary.csv')}")
        else:
            print("\nNo datasets were downloaded.")
    
    finally:
        pass
        # driver.quit() # Will this prematurely close the driver before all downloads finish?

def extract_dataset_info(driver, modal):
    """Extract dataset information from the modal"""
    try:
        # Get dataset name from modal title
        title_element = modal.find_element(By.CSS_SELECTOR, 'h2[id*=":"]')
        full_title = title_element.text
        
        # Split title into name and classification info
        title_spans = title_element.find_elements(By.TAG_NAME, 'span')
        if len(title_spans) >= 2:
            name = title_spans[0].text.strip()
            classification = title_spans[1].text.strip('()')
        else:
            name = full_title
            classification = ""
        
        # Get file information
        file_info = {}
        info_elements = modal.find_elements(By.CSS_SELECTOR, 'p.MuiTypography-body1')
        
        for elem in info_elements:
            text = elem.text
            if 'File Name:' in text:
                file_info['filename'] = text.split('File Name:')[1].strip()
            elif 'File Size:' in text:
                file_info['file_size'] = text.split('File Size:')[1].strip()
            elif 'Last Update:' in text:
                file_info['last_update'] = text.split('Last Update:')[1].strip()
        
        # Get description
        description_elem = modal.find_element(By.CSS_SELECTOR, 'p.MuiTypography-paragraph')
        description = description_elem.text
        
        return {
            'name': name,
            'classification': classification,
            'description': description,
            'filename': file_info.get('filename', 'unknown'),
            'file_size': file_info.get('file_size', ''),
            'last_update': file_info.get('last_update', '')
        }
    
    except Exception as e:
        print(f"    Error extracting dataset info: {str(e)}")
        return {
            'name': 'Unknown',
            'classification': '',
            'description': '',
            'filename': 'unknown',
            'file_size': '',
            'last_update': ''
        }

def save_feature_description(driver, modal, filename, data_dir):
    """Save the feature description table as CSV"""
    try:
        # Find the table in the modal
        table = modal.find_element(By.CSS_SELECTOR, 'table.MuiTable-root')
        
        # Extract table headers
        headers = []
        header_cells = table.find_elements(By.CSS_SELECTOR, 'thead th')
        for cell in header_cells:
            headers.append(cell.text.strip())
        
        # Extract table rows
        rows = []
        body_rows = table.find_elements(By.CSS_SELECTOR, 'tbody tr')
        for row in body_rows:
            cells = row.find_elements(By.CSS_SELECTOR, 'td')
            row_data = [cell.text.strip() for cell in cells]
            if row_data:  # Only add non-empty rows
                rows.append(row_data)
        
        # Create DataFrame and save
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers)
            
            # Create filename for feature description
            base_name = filename.replace('.csv', '') if filename.endswith('.csv') else filename
            feature_filename = f"{base_name}_features.csv"
            feature_path = os.path.join(data_dir, feature_filename)
            
            df.to_csv(feature_path, index=False)
            print(f"    Saved feature description: {feature_filename}")
    
    except Exception as e:
        print(f"    Error saving feature description: {str(e)}")

def go_to_next_page(driver, wait):
    """Navigate to the next page if available"""
    try:
        # Look for the next page button
        # The next button is the one with arrow pointing right that's not disabled
        next_buttons = driver.find_elements(By.CSS_SELECTOR, 'button[aria-label*="next"] svg path[d*="10 6"]')
        
        if not next_buttons:
            return False
        
        next_button = next_buttons[0].find_element(By.XPATH, '../..')
        
        # Check if button is disabled
        button_classes = next_button.get_attribute('class')
        if 'Mui-disabled' in button_classes:
            return False
        
        # Click next button
        driver.execute_script("arguments[0].click();", next_button)
        
        # Wait for page to load
        time.sleep(3)
        
        # Verify we're on a new page by waiting for table to reload
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.MuiTable-root')))
        
        return True
    
    except Exception as e:
        print(f"Error navigating to next page: {str(e)}")
        return False
    
if __name__ == '__main__':
    print('starting')
    download_data()
