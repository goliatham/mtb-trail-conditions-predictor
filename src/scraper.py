"""Scrape MTBProject condition history for Alum Creek Phase 1 & 2."""

import csv
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

TRAILS = {
    "phase1": {"id": "4080717", "name": "Alum Creek Phase 1", "trail_id": 0},
    "phase2": {"id": "4081038", "name": "Alum Creek Phase 2", "trail_id": 1},
}

_TRUSTED_PATH = Path(__file__).parent.parent / "config" / "trusted_users.txt"


def _load_trusted():
    lines = _TRUSTED_PATH.read_text().splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}


TRUSTED_USERS = _load_trusted()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
}

COLOR_TO_LABEL = {"green": 2, "yellow": 1, "red": 0}
COLOR_TO_NAME = {"green": "All Clear", "yellow": "Minor Issues", "red": "Bad/Closed"}

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "training_raw.csv"


def scrape_trail(trail_id: str) -> list[dict]:
    url = f"https://www.mtbproject.com/ajax/public/trail/conditions/{trail_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return parse_conditions(resp.text)


def parse_conditions(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for div in soup.find_all("div", class_="mb-1"):
        img = div.find("img", class_="condition")
        if not img:
            continue
        color = img["src"].split("/")[-1].replace(".svg", "")
        if color not in COLOR_TO_LABEL:
            continue

        label = COLOR_TO_LABEL[color]
        text = div.get_text(separator=" ", strip=True)

        # Extract date — format "Jun 1, 2024" or "1 hour ago" etc.
        date_str = _extract_date(text)
        if not date_str:
            continue

        # Extract username slug from profile URL
        user_link = div.find("a", href=re.compile(r"/user/"))
        username_slug = ""
        if user_link:
            m = re.search(r"/user/\d+/(.+)$", user_link["href"])
            if m:
                username_slug = m.group(1)

        trusted = username_slug in TRUSTED_USERS

        # Extract comment text (everything after the date and dash)
        comment = _extract_comment(text)

        records.append({
            "date": date_str,
            "label": label,
            "color": color,
            "username": username_slug,
            "trusted": trusted,
            "comment": comment,
        })
    return records


def _extract_date(text: str):
    m = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})",
        text,
    )
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(0).replace(",", ""), "%b %d %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _extract_comment(text: str) -> str:
    # Remove rating label and date, keep the comment
    text = re.sub(r"(All Clear|Minor Issues|Bad / Closed)\s*", "", text)
    text = re.sub(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}:\s*",
        "",
        text,
    )
    # Strip trailing username (after em dash)
    text = re.sub(r"\s*—\s*.+$", "", text).strip()
    return text


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    all_records = []
    for key, trail in TRAILS.items():
        print(f"Scraping {trail['name']}...")
        records = scrape_trail(trail["id"])
        for r in records:
            r["trail_key"] = key
            r["trail_id"] = trail["trail_id"]
        all_records.extend(records)
        print(f"  {len(records)} records")
        time.sleep(1)

    fieldnames = ["date", "trail_key", "trail_id", "label", "color", "username", "trusted", "comment"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nSaved {len(all_records)} records to {OUTPUT_PATH}")
    label_counts = {}
    for r in all_records:
        label_counts[r["color"]] = label_counts.get(r["color"], 0) + 1
    print("Label distribution:", label_counts)


if __name__ == "__main__":
    main()
