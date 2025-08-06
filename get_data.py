import os
import time
import re
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
import os

def upsert_summary_csv(records, data_dir, key_cols=None, csv_name='downloaded_datasets_summary.csv'):
    """
    Upsert records into an existing summary CSV.
    key_cols: list of columns used to identify duplicates. If None, uses ['filename'].
    """
    if not records:
        return None

    key_cols = key_cols or ['filename']
    out_path = os.path.join(data_dir, csv_name)

    new_df = pd.DataFrame(records)

    if os.path.exists(out_path):
        try:
            old_df = pd.read_csv(out_path)
            # Align columns: union of old/new, then concat
            for col in set(new_df.columns) - set(old_df.columns):
                old_df[col] = pd.NA
            for col in set(old_df.columns) - set(new_df.columns):
                new_df[col] = pd.NA
            combined = pd.concat([old_df, new_df], ignore_index=True)

            # Drop duplicates by key columns, keeping the last occurrence (latest run wins)
            combined = combined.drop_duplicates(subset=key_cols, keep='last')
        except Exception as e:
            print(f"Warning: failed to read existing summary CSV, creating new one. Error: {e}")
            combined = new_df
    else:
        combined = new_df

    combined.to_csv(out_path, index=False)
    return combined

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
        driver.maximize_window()
        
        input("Filter down the datasets you want to download (click the table headings to filter), then press Enter to continue...")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.MuiTable-root')))
        
        page_num = 1
        downloaded_datasets = []
        while True:
            print(f"Processing page {page_num}...")
            time.sleep(1)
            
            # Parse the visible rows and their download buttons
            rows = parse_table_rows(driver, wait)
            print(f"Found {len(rows)} datasets on page {page_num}")
            
            for i, row in enumerate(rows):
                try:
                    print(f"  Processing dataset {i+1}/{len(rows)}: {row['name']}")

                    # Click its download button to open modal
                    driver.execute_script("arguments[0].click();", row['download_button'])
                    modal = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]')))

                    # Extract file info from modal
                    file_info = extract_modal_file_info(modal)

                    dataset_file_name = file_info.get('filename', 'unknown')
                    dataset_path = os.path.join(data_dir, dataset_file_name)

                    # Combine row info with modal file info
                    dataset_info = {
                        'name': row['name'],
                        'data_type': row['data_type'],
                        'classification': row['classification'],
                        'product_level': row['product_level'],
                        'years': row['years'],
                        'complexity_data': row['complexity_data'],
                        'filename': file_info.get('filename', 'unknown'),
                        'file_size': file_info.get('file_size', ''),
                        'last_update': file_info.get('last_update', ''),
                    }

                    if os.path.exists(dataset_path):
                        print(f"    Dataset '{dataset_file_name}' already exists. Skipping download.")
                        downloaded_datasets.append(dataset_info)

                        # Close modal and continue
                        close_modal(driver, wait, modal)
                        continue

                    # Save feature description (modal table) then download
                    save_feature_description(modal, dataset_file_name, data_dir)

                    # Click final download button inside modal
                    final_download_button = modal.find_element(By.CSS_SELECTOR, "div[aria-labelledby] button.css-rp01fk")
                    final_download_button.click()
                    
                    time.sleep(3)
                    
                    downloaded_datasets.append(dataset_info)
                    print(f"    Downloaded: {row['name']}")
                    
                    close_modal(driver, wait, modal)
                    
                except Exception as e:
                    print(f"    Error processing dataset {i+1}: {str(e)}")
                    # Try to close modal if it's still open
                    try:
                        modal = driver.find_element(By.CSS_SELECTOR, 'div[role="dialog"]')
                        close_modal(driver, wait, modal)
                    except:
                        pass
                    continue
            
            # Pagination
            if not go_to_next_page(driver, wait):
                break
            
            page_num += 1
        
        # Save summary (upsert into existing CSV)
        if downloaded_datasets:
            combined = upsert_summary_csv(
                downloaded_datasets,
                data_dir=data_dir,
                key_cols=['filename']  # choose keys; you can expand to include e.g., ['name','data_type','classification','product_level','years']
            )
            if combined is not None:
                print(f"\nSummary updated! Total rows now: {len(combined)}")
                print(f"Summary saved to: {os.path.join(data_dir, 'downloaded_datasets_summary.csv')}")
        else:
            print("\nNo datasets were downloaded.")
    
    finally:
        pass
        # driver.quit()

def norm_cell_text(text):
    t = (text or '').strip()
    return np.nan if t.upper() == 'N/A' else t

def parse_table_rows(driver, wait):
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.MuiTable-root')))
    table = driver.find_element(By.CSS_SELECTOR, 'table.MuiTable-root')
    tbody = table.find_element(By.CSS_SELECTOR, 'tbody')
    tr_rows = tbody.find_elements(By.CSS_SELECTOR, 'tr')

    TARGET_CLASSIFICATION_LABELS = {'HS12', 'HS92', 'SITC', 'Services Unilateral'}

    parsed = []
    for tr in tr_rows:
        tds = tr.find_elements(By.CSS_SELECTOR, 'td')
        if len(tds) < 7:
            continue

        # 0: Name
        name = norm_cell_text(tds[0].text)

        # 1: Data Type
        data_type = norm_cell_text(tds[1].text)

        # 2: Classification -> collect chip texts and keep only targets; join with commas
        chips = [chip.text.strip() for chip in tds[2].find_elements(By.CSS_SELECTOR, '.MuiChip-root')]
        filtered = [c for c in chips if c in TARGET_CLASSIFICATION_LABELS]
        if filtered:
            classification = ", ".join(filtered)
        else:
            # fallback to full cell text (still normalize N/A)
            classification = norm_cell_text(tds[2].text)

        # 3: Product Level -> extract digit or np.nan, but also convert "N/A" to np.nan
        raw_product_level = norm_cell_text(tds[3].text)
        if isinstance(raw_product_level, float) and np.isnan(raw_product_level):
            product_level = np.nan
        else:
            m = re.search(r'(\d+)', str(raw_product_level))
            product_level = int(m.group(1)) if m else np.nan

        # 4: Years
        years = norm_cell_text(tds[4].text)

        # 5: Complexity Data -> Yes/No -> True/False; if N/A, becomes np.nan via norm_cell_text
        raw_complexity = norm_cell_text(tds[5].text)
        if isinstance(raw_complexity, float) and np.isnan(raw_complexity):
            complexity_data = np.nan
        else:
            rc = str(raw_complexity).strip().lower()
            if 'yes' in rc:
                complexity_data = True
            elif 'no' in rc:
                complexity_data = False
            else:
                # leave other cases as np.nan for consistency
                complexity_data = np.nan

        # 6: Download button
        btn = None
        try:
            icon_path = tds[6].find_element(By.CSS_SELECTOR, 'svg[viewBox="0 0 24 24"] path[d="M5 20h14v-2H5zM19 9h-4V3H9v6H5l7 7z"]')
            btn = icon_path.find_element(By.XPATH, '../..')
        except:
            try:
                btn = tds[6].find_element(By.XPATH, ".//button[.//span[contains(., 'Download')] or contains(., 'Download')]")
            except:
                pass

        if btn is None:
            continue

        parsed.append({
            'name': name,
            'data_type': data_type,
            'classification': classification,
            'product_level': product_level,
            'years': years,
            'complexity_data': complexity_data,
            'download_button': btn
        })

    return parsed

def extract_modal_file_info(modal):
    """
    Extract only filename, file_size, last_update from the modal.
    """
    file_info = {'filename': 'unknown', 'file_size': '', 'last_update': ''}
    try:
        for elem in modal.find_elements(By.CSS_SELECTOR, 'p.MuiTypography-body1'):
            text = (elem.text or '').strip()
            if 'File Name:' in text:
                file_info['filename'] = text.split('File Name:')[1].strip()
            elif 'File Size:' in text:
                file_info['file_size'] = text.split('File Size:')[1].strip()
            elif 'Last Update:' in text:
                file_info['last_update'] = text.split('Last Update:')[1].strip()
    except Exception as e:
        print(f"    Error extracting modal file info: {str(e)}")
    return file_info

def save_feature_description(modal, filename, data_dir):
    """Save the feature description table from the modal as CSV"""
    try:
        table = modal.find_element(By.CSS_SELECTOR, 'table.MuiTable-root')
        
        # Headers
        headers = [cell.text.strip() for cell in table.find_elements(By.CSS_SELECTOR, 'thead th')]

        # Rows
        rows = []
        for row in table.find_elements(By.CSS_SELECTOR, 'tbody tr'):
            cells = row.find_elements(By.CSS_SELECTOR, 'td')
            row_data = [cell.text.strip() for cell in cells]
            if row_data:
                rows.append(row_data)
        
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers)
            base_name = filename[:-4] if filename.lower().endswith('.csv') else filename
            feature_filename = f"{base_name}_features.csv"
            feature_path = os.path.join(data_dir, feature_filename)
            df.to_csv(feature_path, index=False)
            print(f"    Saved feature description: {feature_filename}")
    
    except Exception as e:
        print(f"    Error saving feature description: {str(e)}")

def close_modal(driver, wait, modal):
    """Click the X button to close the modal and wait for invisibility"""
    try:
        close_btn = modal.find_element(By.CSS_SELECTOR, 'button svg[viewBox="0 0 24 24"] path[d*="19 6.41"]')
        close_btn = close_btn.find_element(By.XPATH, '../..')
        driver.execute_script("arguments[0].click();", close_btn)
        wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]')))
    except Exception as e:
        print(f"    Error closing modal: {str(e)}")

def go_to_next_page(driver, wait):
    """Navigate to the next page if available"""
    try:
        next_buttons = driver.find_elements(By.CSS_SELECTOR, 'button[aria-label*="next"] svg path[d*="10 6"]')
        if not next_buttons:
            return False
        
        next_button = next_buttons[0].find_element(By.XPATH, '../..')
        if 'Mui-disabled' in (next_button.get_attribute('class') or ''):
            return False
        
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(1)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.MuiTable-root')))
        return True
    
    except Exception as e:
        print(f"Error navigating to next page: {str(e)}")
        return False

if __name__ == '__main__':
    print('starting')
    download_data()