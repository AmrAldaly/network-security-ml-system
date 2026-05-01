"""
feature_extraction.py
---------------------
Extracts all 30 features from a raw URL, matching exactly the training schema.

All output values are encoded as:
    1  → legitimate / safe signal
    0  → suspicious / neutral signal
   -1  → phishing / dangerous signal

Assumptions for non-reliably-extractable features:
    - Page_Rank      : defaults to -1 (unknown = suspicious) — no free API available
    - Google_Index   : attempts a HEAD request to google.com/search; defaults to -1 on failure
    - web_traffic    : defaults to 0 (unknown) — Alexa API deprecated
    - Statistical_report : defaults to -1 (conservative default, cannot verify without paid DB)
"""

import re
import socket
import ssl
import time
import ipaddress
from urllib.parse import urlparse
from datetime import datetime

import requests
import whois
from bs4 import BeautifulSoup

# ── constants ────────────────────────────────────────────────────────────────

SHORTENING_SERVICES = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|"
    r"is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|"
    r"su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|"
    r"Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|"
    r"kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|q\.gs|po\.st|"
    r"bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
    r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|"
    r"tweez\.me|v\.gd|tr\.im|link\.zip\.net",
    re.IGNORECASE,
)

REQUEST_TIMEOUT = 5  # seconds for all HTTP requests


# ── helpers ──────────────────────────────────────────────────────────────────

def _fetch_page(url: str):
    """Fetch page HTML. Returns (response, soup) or (None, None) on failure."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True)
        soup = BeautifulSoup(resp.text, "html.parser")
        return resp, soup
    except Exception:
        return None, None


def _get_domain(parsed) -> str:
    return parsed.netloc.replace("www.", "")


def _get_whois(domain: str):
    """Returns whois info dict or None."""
    try:
        return whois.whois(domain)
    except Exception:
        return None


# ── individual feature extractors ────────────────────────────────────────────

def having_ip_address(url: str) -> int:
    """Check if URL uses an IP address instead of a domain name.
    Phishing sites often use raw IPs to avoid domain registration traces."""
    try:
        host = urlparse(url).netloc.split(":")[0]
        ipaddress.ip_address(host)
        return -1  # IP address found → phishing signal
    except ValueError:
        return 1   # legitimate domain


def url_length(url: str) -> int:
    """URLs < 54 chars → legitimate, 54–75 → suspicious, >75 → phishing."""
    length = len(url)
    if length < 54:
        return 1
    elif length <= 75:
        return 0
    return -1


def shortening_service(url: str) -> int:
    """Detect use of URL shortening services (often used to disguise phishing URLs)."""
    return -1 if SHORTENING_SERVICES.search(url) else 1


def having_at_symbol(url: str) -> int:
    """'@' in URL forces browser to treat everything before it as credentials.
    This is a classic phishing trick."""
    return -1 if "@" in url else 1


def double_slash_redirecting(url: str) -> int:
    """Check if '//' appears after position 6 in the URL (after 'https://').
    A double-slash mid-URL is used for open redirect attacks."""
    return -1 if url.rfind("//") > 6 else 1


def prefix_suffix(url: str) -> int:
    """Check for '-' in the domain name. Phishing sites often use dashes
    to mimic legitimate domains (e.g., paypal-secure.com)."""
    domain = urlparse(url).netloc
    return -1 if "-" in domain else 1


def having_sub_domain(url: str) -> int:
    """Count the number of sub-domains.
    0–1 dots → legitimate, 2 dots → suspicious, 3+ dots → phishing."""
    domain = urlparse(url).netloc
    # Remove 'www.' before counting
    domain = domain.replace("www.", "")
    dot_count = domain.count(".")
    if dot_count == 1:
        return 1
    elif dot_count == 2:
        return 0
    return -1


def ssl_final_state(url: str) -> int:
    """Check if HTTPS is used AND if the SSL certificate is from a trusted issuer.
    No HTTPS → phishing; HTTPS with valid cert → legitimate."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return -1
    try:
        hostname = parsed.netloc.split(":")[0]
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.settimeout(REQUEST_TIMEOUT)
            s.connect((hostname, 443))
            cert = s.getpeercert()
            # Check certificate validity period
            not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
            if not_after > datetime.utcnow():
                return 1  # valid HTTPS cert
        return 0
    except Exception:
        return 0  # HTTPS present but cert check failed → suspicious


def domain_registration_length(domain_info) -> int:
    """Domains registered for ≤ 1 year are suspicious (phishers use short registrations).
    Assumption: if whois unavailable, default to -1."""
    try:
        expiry = domain_info.expiration_date
        if isinstance(expiry, list):
            expiry = expiry[0]
        creation = domain_info.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if expiry and creation:
            duration = (expiry - creation).days
            return 1 if duration > 365 else -1
    except Exception:
        pass
    return -1


def favicon(url: str, soup) -> int:
    """Check if favicon is loaded from a different domain.
    External favicons are a phishing indicator."""
    if soup is None:
        return -1
    try:
        parsed = urlparse(url)
        base_domain = _get_domain(parsed)
        for link in soup.find_all("link", rel=lambda r: r and "icon" in " ".join(r).lower()):
            href = link.get("href", "")
            if href and base_domain not in href and href.startswith("http"):
                return -1  # favicon loaded from external domain
        return 1
    except Exception:
        return -1


def port(url: str) -> int:
    """Check if a non-standard port is used in the URL.
    Non-standard ports (not 80/443) are a phishing signal."""
    parsed = urlparse(url)
    port_num = parsed.port
    if port_num is None or port_num in (80, 443):
        return 1
    return -1


def https_token(url: str) -> int:
    """Check for 'https' appearing in the domain part of the URL (not the scheme).
    e.g., 'http://https-paypal.com' → phishing trick."""
    domain = urlparse(url).netloc
    return -1 if "https" in domain.lower() else 1


def request_url(url: str, soup) -> int:
    """Check what percentage of page resources (img, script, link) load from external domains.
    >61% external → phishing, 22–61% → suspicious, <22% → legitimate."""
    if soup is None:
        return -1
    try:
        parsed = urlparse(url)
        base_domain = _get_domain(parsed)
        tags = soup.find_all(["img", "script", "link"])
        total = len(tags)
        if total == 0:
            return 1
        external = sum(
            1 for t in tags
            if base_domain not in (t.get("src", "") + t.get("href", ""))
            and (t.get("src", "").startswith("http") or t.get("href", "").startswith("http"))
        )
        ratio = external / total
        if ratio < 0.22:
            return 1
        elif ratio < 0.61:
            return 0
        return -1
    except Exception:
        return -1


def url_of_anchor(url: str, soup) -> int:
    """Check % of anchor <a> tags that link to a different domain or use '#'.
    >67% suspicious → phishing, 31–67% → suspicious, <31% → legitimate."""
    if soup is None:
        return -1
    try:
        parsed = urlparse(url)
        base_domain = _get_domain(parsed)
        anchors = soup.find_all("a", href=True)
        total = len(anchors)
        if total == 0:
            return 1
        unsafe = sum(
            1 for a in anchors
            if a["href"] in ("#", "javascript:void(0)", "")
            or (a["href"].startswith("http") and base_domain not in a["href"])
        )
        ratio = unsafe / total
        if ratio < 0.31:
            return 1
        elif ratio < 0.67:
            return 0
        return -1
    except Exception:
        return -1


def links_in_tags(url: str, soup) -> int:
    """Check % of <meta>, <script>, <link> tags pointing to external domains.
    >17% → phishing, 0–17% → suspicious."""
    if soup is None:
        return -1
    try:
        parsed = urlparse(url)
        base_domain = _get_domain(parsed)
        tags = soup.find_all(["meta", "script", "link"])
        total = len(tags)
        if total == 0:
            return 1
        external = sum(
            1 for t in tags
            if base_domain not in (t.get("src", "") + t.get("href", "") + t.get("content", ""))
            and any(
                (t.get(attr, "").startswith("http"))
                for attr in ("src", "href", "content")
            )
        )
        ratio = external / total
        if ratio < 0.17:
            return 1
        elif ratio < 0.81:
            return 0
        return -1
    except Exception:
        return -1


def sfh(url: str, soup) -> int:
    """Server Form Handler check: if form action is blank, 'about:blank', or external domain → phishing."""
    if soup is None:
        return -1
    try:
        parsed = urlparse(url)
        base_domain = _get_domain(parsed)
        forms = soup.find_all("form", action=True)
        if not forms:
            return 1
        for form in forms:
            action = form["action"].strip()
            if action in ("", "about:blank"):
                return -1
            if action.startswith("http") and base_domain not in action:
                return -1
        return 1
    except Exception:
        return -1


def submitting_to_email(soup) -> int:
    """Check if form uses 'mailto:' action to send data to email → phishing."""
    if soup is None:
        return 1
    try:
        for form in soup.find_all("form", action=True):
            if "mailto:" in form["action"].lower():
                return -1
        return 1
    except Exception:
        return 1


def abnormal_url(url: str, domain_info) -> int:
    """Check if the URL host matches the whois registered domain.
    Mismatch → phishing."""
    try:
        host = urlparse(url).netloc
        whois_domain = domain_info.domain_name
        if isinstance(whois_domain, list):
            whois_domain = whois_domain[0]
        if whois_domain and whois_domain.lower() in host.lower():
            return 1
        return -1
    except Exception:
        return -1


def redirect(resp) -> int:
    """Count HTTP redirects. ≤1 → legitimate, 2–4 → suspicious, >4 → phishing."""
    if resp is None:
        return -1
    try:
        redirect_count = len(resp.history)
        if redirect_count <= 1:
            return 1
        elif redirect_count <= 4:
            return 0
        return -1
    except Exception:
        return -1


def on_mouseover(soup) -> int:
    """Check for onMouseOver events that change the status bar (URL hiding trick)."""
    if soup is None:
        return 1
    try:
        page_text = str(soup)
        if "onmouseover" in page_text.lower() and "window.status" in page_text.lower():
            return -1
        return 1
    except Exception:
        return 1


def right_click(soup) -> int:
    """Check if right-click is disabled (prevents users from inspecting the page source)."""
    if soup is None:
        return 1
    try:
        page_text = str(soup)
        if "contextmenu" in page_text.lower() and "return false" in page_text.lower():
            return -1
        return 1
    except Exception:
        return 1


def popup_window(soup) -> int:
    """Check for popup windows requesting user input (credential harvesting)."""
    if soup is None:
        return 1
    try:
        page_text = str(soup)
        if "prompt(" in page_text or "window.open(" in page_text:
            return -1
        return 1
    except Exception:
        return 1


def iframe(soup) -> int:
    """Check for invisible iframes (hidden frames used for clickjacking)."""
    if soup is None:
        return 1
    try:
        for frame in soup.find_all("iframe"):
            style = frame.get("style", "")
            width = frame.get("width", "1")
            height = frame.get("height", "1")
            if "display:none" in style.replace(" ", "") or width == "0" or height == "0":
                return -1
        return 1
    except Exception:
        return 1


def age_of_domain(domain_info) -> int:
    """Domains younger than 6 months → phishing. Older → legitimate.
    Assumption: whois unavailable → -1."""
    try:
        creation = domain_info.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            age_days = (datetime.utcnow() - creation).days
            return 1 if age_days >= 180 else -1
    except Exception:
        pass
    return -1


def dns_record(domain_info) -> int:
    """If whois returns no record, the domain likely has no DNS entry → phishing."""
    try:
        if domain_info and domain_info.domain_name:
            return 1
        return -1
    except Exception:
        return -1


def web_traffic(url: str) -> int:
    """Alexa rank API is deprecated. Default: 0 (suspicious/unknown).
    Assumption: we cannot reliably determine traffic without a paid API."""
    # NOTE: Replace with a real traffic API (SimilarWeb, etc.) if available.
    return 0


def page_rank(url: str) -> int:
    """Google PageRank API is deprecated. Default: -1 (conservative/unknown).
    Assumption: unknown rank is treated as a phishing signal."""
    # NOTE: Replace with a real PageRank or domain authority API if available.
    return -1


def google_index(url: str) -> int:
    """Check if the page is indexed by Google via a site: search HEAD request.
    Assumption: request failure defaults to -1."""
    try:
        domain = urlparse(url).netloc
        search_url = f"https://www.google.com/search?q=site:{domain}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(search_url, timeout=REQUEST_TIMEOUT, headers=headers)
        # If Google returns results, the site is indexed
        if resp.status_code == 200 and "did not match any documents" not in resp.text:
            return 1
        return -1
    except Exception:
        return -1


def links_pointing_to_page(url: str, soup) -> int:
    """Count inbound links found on the page itself as a proxy.
    0 links → phishing, 1–2 → suspicious, 2+ → legitimate."""
    if soup is None:
        return -1
    try:
        parsed = urlparse(url)
        base_domain = _get_domain(parsed)
        # Count external links that point back to base domain
        inbound = sum(
            1 for a in soup.find_all("a", href=True)
            if base_domain in a["href"]
        )
        if inbound == 0:
            return -1
        elif inbound <= 2:
            return 0
        return 1
    except Exception:
        return -1


def statistical_report(url: str) -> int:
    """Checks against known phishing IPs/domains. Requires paid DBs (PhishTank, etc.).
    Assumption: defaults to -1 (conservative) — cannot verify without API access."""
    # NOTE: Integrate with PhishTank / OpenPhish API here if available.
    return -1


# ── main entry point ─────────────────────────────────────────────────────────

# The exact feature order used during training — DO NOT change this order.
FEATURE_ORDER = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report",
]


def extract_features(url: str) -> dict:
    """
    Extract all 30 features from a URL.

    Parameters
    ----------
    url : str
        The raw URL to analyse (include scheme, e.g. 'https://example.com').

    Returns
    -------
    dict
        Ordered dictionary with all 30 feature names mapped to their encoded value
        (-1, 0, or 1).
    """
    # Ensure scheme is present for urlparse to work correctly
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    domain = _get_domain(parsed)

    # Fetch page HTML once (reused by multiple features)
    resp, soup = _fetch_page(url)

    # Fetch whois once (reused by multiple features)
    domain_info = _get_whois(domain)

    features = {
        "having_IP_Address":          having_ip_address(url),
        "URL_Length":                 url_length(url),
        "Shortining_Service":         shortening_service(url),
        "having_At_Symbol":           having_at_symbol(url),
        "double_slash_redirecting":   double_slash_redirecting(url),
        "Prefix_Suffix":              prefix_suffix(url),
        "having_Sub_Domain":          having_sub_domain(url),
        "SSLfinal_State":             ssl_final_state(url),
        "Domain_registeration_length": domain_registration_length(domain_info),
        "Favicon":                    favicon(url, soup),
        "port":                       port(url),
        "HTTPS_token":                https_token(url),
        "Request_URL":                request_url(url, soup),
        "URL_of_Anchor":              url_of_anchor(url, soup),
        "Links_in_tags":              links_in_tags(url, soup),
        "SFH":                        sfh(url, soup),
        "Submitting_to_email":        submitting_to_email(soup),
        "Abnormal_URL":               abnormal_url(url, domain_info),
        "Redirect":                   redirect(resp),
        "on_mouseover":               on_mouseover(soup),
        "RightClick":                 right_click(soup),
        "popUpWidnow":                popup_window(soup),
        "Iframe":                     iframe(soup),
        "age_of_domain":              age_of_domain(domain_info),
        "DNSRecord":                  dns_record(domain_info),
        "web_traffic":                web_traffic(url),
        "Page_Rank":                  page_rank(url),
        "Google_Index":               google_index(url),
        "Links_pointing_to_page":     links_pointing_to_page(url, soup),
        "Statistical_report":         statistical_report(url),
    }

    # Guarantee output order matches training schema
    return {k: features[k] for k in FEATURE_ORDER}
