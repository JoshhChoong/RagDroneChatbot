import os
import re
import json
import hashlib
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from Data.DataScrape.RemoveExtraneousData import remove_extraneous_data


def fetch_html(url: str, timeout: int = 30) -> str:
    """
    Fetch the HTML content of the given URL.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def discover_sublinks(soup: BeautifulSoup, base_url: str, path_prefix: str) -> List[str]:
    """Return absolute URLs for links that start with the given path_prefix on the same domain."""
    links = set()
    base_parsed = urlparse(base_url)
    domain = base_parsed.netloc
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        # make absolute
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.netloc != domain:
            continue
        if parsed.path.startswith(path_prefix):
            links.add(abs_url)
    return sorted(links)


def save_file(path: str, data: str, mode: str = 'w') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding='utf8') as f:
        f.write(data)


def extract_title(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find('h1')
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return None


def extract_publication_year(soup: BeautifulSoup, text: str) -> Optional[int]:
    """Get publicaiton year"""
    # try meta tags
    for meta_name in ('date', 'pubdate', 'publication_date', 'dc.date', 'article:published_time'):
        tag = soup.find('meta', attrs={'name': meta_name}) or soup.find('meta', attrs={'property': meta_name})
        if tag and tag.get('content'):
            m = re.search(r'(20\d{2})', tag['content'])
            if m:
                return int(m.group(1))
    # regex if not found 
    m = re.search(r'\b(20\d{2})\b', text)
    if m:
        return int(m.group(1))
    return None


def infer_document_type(url: str, text: str) -> str:
    """Infer document type based on URL patterns and text heuristics."""
    path = urlparse(url).path.lower()
    if 'faq' in path or 'frequently-asked' in path:
        return 'FAQ'
    if 'regulation' in path or 'regulations' in path or 'acts-regulations' in path:
        return 'regulation'
    if 'advisory' in path or 'notice' in path or 'safety-advisory' in path:
        return 'advisory'
    if 'regulation' in text.lower() or 'section' in text.lower():
        return 'regulation'
    return 'unknown'


def compute_hash_id(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode('utf8'))
    return h.hexdigest()


def clean_and_save(html: str, url: str, out_dir: str, filename_override: Optional[str] = None) -> None:
    soup = BeautifulSoup(html, 'html.parser')
    cleaned = remove_extraneous_data(soup)
    # Use spaces instead of newlines when concatenating text to avoid '\n' artifacts,
    # and normalize consecutive whitespace into single spaces.
    text = cleaned.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()

    # metadata
    title = extract_title(cleaned)
    publication_year = extract_publication_year(cleaned, text)
    document_type = infer_document_type(url, text)
    parsed = urlparse(url)
    domain = parsed.netloc
    source_organization = 'Transport Canada' if 'tc.canada.ca' in domain else domain
    country = 'Canada' if 'canada' in domain or 'gc.ca' in domain else None
    trust_score = 0.9 if 'tc.canada.ca' in domain else 0.5
    hash_id = compute_hash_id(text)

    metadata = {
        'title': title,
        'source_organization': source_organization,
        'country': country,
        'publication_year': publication_year,
        'document_type': document_type,
        'url': url,
        'trust_score': trust_score,
        'hash_id': hash_id,
    }

    # file-safe name from path
    safe_path = parsed.path.strip('/').replace('/', '_') or 'index'
    raw_path = os.path.join(out_dir, 'raw', f"{safe_path}.html")
    if filename_override:
        json_path = filename_override
        # ensure directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    else:
        json_path = os.path.join(out_dir, 'processed', f"{safe_path}.json")

    save_file(raw_path, html, mode='w')
    # save JSON with text+metadata
    payload = {'text': text, 'meta': metadata}
    save_file(json_path, json.dumps(payload, ensure_ascii=False, indent=2), mode='w')


def download_data(root_url: str, out_dir: str = './Data/files', path_prefix: str = '/en/aviation/drone-safety', out_file: Optional[str] = None) -> List[str]:
    """Download the root_url and sublinks under path_prefix, clean and save them.

    Returns list of saved URLs.
    """
    html = fetch_html(root_url)
    soup = BeautifulSoup(html, 'html.parser')

    urls_saved = []
    # save root (allow writing root JSON to a specific filename)
    clean_and_save(html, root_url, out_dir, filename_override=out_file)
    urls_saved.append(root_url)

    sublinks = discover_sublinks(soup, root_url, path_prefix)
    for link in sublinks:
        try:
            page_html = fetch_html(link)
            clean_and_save(page_html, link, out_dir)
            urls_saved.append(link)
        except Exception as e:
            print(f"Failed to fetch {link}: {e}")

    return urls_saved


if __name__ == '__main__':
    ROOT = 'https://tc.canada.ca/en/aviation/drone-safety'
    out = download_data(ROOT, out_dir=os.path.join('Data', 'files'))
    print(f"Saved {len(out)} pages")
