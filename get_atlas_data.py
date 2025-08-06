import os
import re
import time
import glob
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


ATLAS_URL = 'https://atlas.hks.harvard.edu/data-downloads'
TARGET_CLASSIFICATION_LABELS = {'HS12', 'HS92', 'SITC', 'Services Unilateral'}


def setup_driver(download_dir):
    os.makedirs(download_dir, exist_ok=True)
    prefs = {
        'download.default_directory': download_dir,
        'download.prompt_for_download': False,
        'download.directory_upgrade': True,
        'safebrowsing.enabled': True
    }
    options = webdriver.ChromeOptions()
    options.add_experimental_option('prefs', prefs)
    # options.add_argument('--headless=new')  # Uncomment for headless
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    return driver


def norm_cell_text(text):
    t = (text or '').strip()
    return np.nan if t.upper() == 'N/A' else t


def parse_table_rows(driver, wait):
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.MuiTable-root')))
    table = driver.find_element(By.CSS_SELECTOR, 'table.MuiTable-root')
    tbody = table.find_element(By.CSS_SELECTOR, 'tbody')
    tr_rows = tbody.find_elements(By.CSS_SELECTOR, 'tr')

    parsed = []
    for tr in tr_rows:
        tds = tr.find_elements(By.CSS_SELECTOR, 'td')
        if len(tds) < 7:
            continue

        name = norm_cell_text(tds[0].text)
        data_type = norm_cell_text(tds[1].text)

        # Classification: prefer chip labels with filtering; fallback to full cell text
        try:
            chips = [c.text.strip() for c in tds[2].find_elements(By.CSS_SELECTOR, '.MuiChip-label span')]
            chips = [c for c in chips if c]
            if not chips:
                chips = [c.text.strip() for c in tds[2].find_elements(By.CSS_SELECTOR, '.MuiChip-root')]
        except Exception:
            chips = []
        filtered = [c for c in chips if c in TARGET_CLASSIFICATION_LABELS]
        classification = ", ".join(filtered) if filtered else norm_cell_text(tds[2].text)

        # Product level: extract integer if present, else np.nan
        raw_product_level = norm_cell_text(tds[3].text)
        if isinstance(raw_product_level, float) and np.isnan(raw_product_level):
            product_level = np.nan
        else:
            m = re.search(r'(\d+)', str(raw_product_level))
            product_level = int(m.group(1)) if m else np.nan

        years = norm_cell_text(tds[4].text)

        # Complexity: parse yes/no, else np.nan
        raw_complexity = norm_cell_text(tds[5].text)
        if isinstance(raw_complexity, float) and np.isnan(raw_complexity):
            complexity_data = np.nan
        else:
            rc = str(raw_complexity).strip().lower()
            complexity_data = True if 'yes' in rc else False if 'no' in rc else np.nan

        # Download button: try icon path first, then text fallback
        btn = None
        try:
            icon_path = tds[6].find_element(
                By.CSS_SELECTOR,
                'svg[viewBox="0 0 24 24"] path[d="M5 20h14v-2H5zM19 9h-4V3H9v6H5l7 7z"]'
            )
            btn = icon_path.find_element(By.XPATH, '../..')
        except Exception:
            try:
                btn = tds[6].find_element(
                    By.XPATH,
                    ".//button[.//span[contains(., 'Download')] or contains(., 'Download')]"
                )
            except Exception:
                btn = None

        if btn is None:
            continue

        parsed.append({
            'name': name,
            'data_type': data_type,
            'classification': classification,
            'product_level': product_level,
            'years': years,
            'complexity_data': complexity_data,
            'download_button': btn,
        })

    return parsed


def extract_modal_file_info(modal):
    info = {'filename': 'unknown', 'file_size': '', 'last_update': ''}
    try:
        for elem in modal.find_elements(By.CSS_SELECTOR, 'p.MuiTypography-body1'):
            text = (elem.text or '').strip()
            if 'File Name:' in text:
                info['filename'] = text.split('File Name:')[1].strip()
            elif 'File Size:' in text:
                info['file_size'] = text.split('File Size:')[1].strip()
            elif 'Last Update:' in text:
                info['last_update'] = text.split('Last Update:')[1].strip()
    except Exception as e:
        print(f"    Error extracting file info: {e}")
    return info


def save_feature_description(modal, filename, data_dir):
    try:
        table = modal.find_element(By.CSS_SELECTOR, 'table.MuiTable-root')
        headers = [th.text.strip() for th in table.find_elements(By.CSS_SELECTOR, 'thead th')]
        rows = []
        for tr in table.find_elements(By.CSS_SELECTOR, 'tbody tr'):
            row_data = [td.text.strip() for td in tr.find_elements(By.CSS_SELECTOR, 'td')]
            if row_data:
                rows.append(row_data)
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers)
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            out_path = os.path.join(data_dir, f"{base_name}_features.csv")
            df.to_csv(out_path, index=False)
            print(f"    Saved feature description: {os.path.basename(out_path)}")
    except Exception as e:
        print(f"    Error saving feature description: {e}")


def close_modal(driver, wait, modal):
    try:
        # Click X button (scoped to modal), then wait for invisibility
        close_btn = modal.find_element(
            By.CSS_SELECTOR,
            'button svg[viewBox="0 0 24 24"] path[d*="19 6.41"]'
        ).find_element(By.XPATH, '../..')
        driver.execute_script("arguments[0].click();", close_btn)
        wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, 'div[role="dialog"]')))
    except Exception as e:
        print(f"    Error closing modal: {e}")


def go_to_next_page(driver, wait):
    try:
        next_icons = driver.find_elements(By.CSS_SELECTOR, 'button[aria-label*="next"] svg path[d*="10 6"]')
        if not next_icons:
            return False
        next_btn = next_icons[0].find_element(By.XPATH, '../..')
        if 'Mui-disabled' in (next_btn.get_attribute('class') or ''):
            return False
        driver.execute_script("arguments[0].click();", next_btn)
        time.sleep(1)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.MuiTable-root')))
        return True
    except Exception as e:
        print(f"Error navigating to next page: {e}")
        return False


def upsert_summary_csv(records, data_dir, key_cols=None, csv_name='downloaded_datasets_summary.csv'):
    if not records:
        return None
    key_cols = key_cols or ['filename']
    out_path = os.path.join(data_dir, csv_name)
    new_df = pd.DataFrame(records)

    if os.path.exists(out_path):
        try:
            old_df = pd.read_csv(out_path)
            for col in set(new_df.columns) - set(old_df.columns):
                old_df[col] = pd.NA
            for col in set(old_df.columns) - set(new_df.columns):
                new_df[col] = pd.NA
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep='last')
        except Exception as e:
            print(f"Warning: failed to read existing summary CSV, creating new one. Error: {e}")
            combined = new_df
    else:
        combined = new_df

    combined.to_csv(out_path, index=False)
    return combined


def wait_for_download(download_dir, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        partials = glob.glob(os.path.join(download_dir, '*.crdownload'))
        if not partials:
            return True
        time.sleep(1.0)
    return False


def download_data(download_dir='data', wait_download=True):
    driver = setup_driver(download_dir)
    wait = WebDriverWait(driver, 15)

    downloaded_datasets = []

    try:
        driver.get(ATLAS_URL)
        input("Filter the table as desired, then press Enter to start...")

        page_num = 1
        while True:
            print(f"\nProcessing page {page_num}...")
            rows = parse_table_rows(driver, wait)
            print(f"Found {len(rows)} datasets")

            for i, row in enumerate(rows, 1):
                print(f"  [{i}/{len(rows)}] {row['name']}")
                modal = None
                try:
                    driver.execute_script("arguments[0].click();", row['download_button'])
                    modal = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[role=\"dialog\"]')))

                    file_info = extract_modal_file_info(modal)
                    filename = file_info.get('filename', 'unknown')
                    target_path = os.path.join(download_dir, filename)

                    dataset_info = {
                        'name': row['name'],
                        'data_type': row['data_type'],
                        'classification': row['classification'],
                        'product_level': row['product_level'],
                        'years': row['years'],
                        'complexity_data': row['complexity_data'],
                        'filename': filename,
                        'file_size': file_info.get('file_size', ''),
                        'last_update': file_info.get('last_update', ''),
                    }

                    # Check if dataset files already exist
                    if os.path.exists(target_path):
                        if os.path.exists(os.path.join(download_dir, f"{filename.split('.')[0]}_features.csv")):
                            print(f"    Already exists: {filename} (skipping)")
                            close_modal(driver, wait, modal)
                            continue
                        else:
                            os.remove(target_path)

                    save_feature_description(modal, filename, download_dir)

                    # Robust download button inside modal
                    dl_btn = None
                    try:
                        dl_btn = modal.find_element(
                            By.XPATH,
                            ".//button[.//span[contains(., 'Download')] or contains(., 'Download')]"
                        )
                    except Exception:
                        try:
                            dl_btn = modal.find_element(By.CSS_SELECTOR, "div[aria-labelledby] button")
                        except Exception:
                            pass

                    if dl_btn is None:
                        print("    Could not find download button in modal.")
                        close_modal(driver, wait, modal)
                        continue

                    driver.execute_script("arguments[0].click();", dl_btn)
                    time.sleep(2)  # allow download to start

                    if wait_download:
                        ok = wait_for_download(download_dir, timeout=180)
                        if not ok:
                            print("    Warning: download may not have completed within timeout.")

                    downloaded_datasets.append(dataset_info)
                    print(f"    Download initiated: {filename}")

                except Exception as e:
                    print(f"    Error: {e}")
                    # Try to close modal if it is/was open
                    try:
                        if modal is None:
                            modal = driver.find_element(By.CSS_SELECTOR, 'div[role="dialog"]')
                        close_modal(driver, wait, modal)
                    except Exception:
                        pass
                    continue

            if not go_to_next_page(driver, wait):
                break
            page_num += 1

        combined = upsert_summary_csv(downloaded_datasets, data_dir=download_dir, key_cols=['filename'])
        if combined is not None:
            print(f"\nSummary updated. Total rows: {len(combined)}")
            print(f"Summary saved to: {os.path.join(download_dir, 'downloaded_datasets_summary.csv')}")
        else:
            print("\nNo datasets processed.")

    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == '__main__':
    download_data(download_dir=os.path.abspath('data'), wait_download=True)