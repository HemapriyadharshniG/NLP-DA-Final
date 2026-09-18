import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://tnrd.tn.gov.in"

# Live public scheme endpoints containing Tamil & English Government Orders
TARGET_SECTIONS = {
    "MGNREGS_Welfare_Schemes": "https://tnrd.tn.gov.in/rdweb_newsite/project/reports/Public/ListOfGO.php?scheme_code=Mg==&scheme_type=NA==",
    "Central_Welfare_Schemes": "https://tnrd.tn.gov.in/rdweb_newsite/project/reports/Public/ListOfGO.php?scheme_code=MQ==&scheme_type=NA==",
    "Panchayat_Raj_Orders": "https://tnrd.tn.gov.in/rdweb_newsite/project/reports/Public/ListOfGO.php?scheme_code=Mw==&scheme_type=NA=="
}

OUTPUT_DIR = "dataset_raw_gos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def clean_filename(text):
    return re.sub(r'[\/\\:*?"<>| ]', '_', text).strip('_')

def scrape_section(session, section_name, section_url, max_downloads=15):
    records = []
    dept_folder = os.path.join(OUTPUT_DIR, section_name)
    os.makedirs(dept_folder, exist_ok=True)

    print(f"\n[*] Fetching section: {section_name}")
    
    try:
        resp = session.get(section_url, headers=HEADERS, timeout=30, verify=False)
        if resp.status_code != 200:
            print(f" [!] Received status code {resp.status_code}")
            return records

        soup = BeautifulSoup(resp.content, "html.parser")
        table = soup.find("table")

        if not table:
            print(f" [-] No GO table located on page.")
            return records

        rows = table.find_all("tr")[1:]  # Skip headers
        print(f" [+] Found {len(rows)} orders listed. Starting downloads...")

        count = 0
        for row in rows:
            if count >= max_downloads:
                break

            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            go_num = cols[1].get_text(strip=True) if len(cols) > 1 else ""
            go_date = cols[2].get_text(strip=True) if len(cols) > 2 else ""
            subject = cols[3].get_text(strip=True) if len(cols) > 3 else ""

            # Extract PDF download link
            link_tag = row.find("a", href=True)
            if not link_tag or not link_tag["href"]:
                continue

            pdf_url = urljoin(section_url, link_tag["href"])
            file_title = f"{clean_filename(go_num)}_{clean_filename(go_date)}"
            if len(file_title) < 4:
                file_title = f"GO_{count+1}"

            pdf_filename = f"{file_title[:50]}.pdf"
            pdf_path = os.path.join(dept_folder, pdf_filename)

            # Download the document
            if not os.path.exists(pdf_path):
                try:
                    pdf_res = session.get(pdf_url, headers=HEADERS, timeout=35, verify=False)
                    if pdf_res.status_code == 200 and len(pdf_res.content) > 1000:
                        with open(pdf_path, "wb") as f:
                            f.write(pdf_res.content)
                        print(f"  [✓] Saved: {pdf_filename}")
                        count += 1
                    time.sleep(0.5)
                except Exception as err:
                    print(f"  [!] Failed downloading {pdf_url}: {err}")

            records.append({
                "category": section_name,
                "go_number": go_num,
                "date": go_date,
                "subject": subject,
                "pdf_url": pdf_url,
                "local_path": pdf_path
            })

    except Exception as e:
        print(f" [!] Error connecting to {section_name}: {e}")

    return records

if __name__ == "__main__":
    # Disable SSL insecure warnings for NIC state portals
    requests.packages.urllib3.disable_warnings()

    session = requests.Session()
    master_records = []

    for name, url in TARGET_SECTIONS.items():
        data = scrape_section(session, name, url, max_downloads=15)
        master_records.extend(data)

    csv_path = os.path.join(OUTPUT_DIR, "go_metadata.csv")
    pd.DataFrame(master_records).to_csv(csv_path, index=False)
    print(f"\n[✓] Completed. Ingested {len(master_records)} records into: {csv_path}")