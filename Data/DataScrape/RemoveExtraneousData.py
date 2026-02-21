
def remove_nav(soup):
    """
    removes the navigation bar from the html data
    """
    # Remove semantic <nav> elements
    for nav in soup.find_all('nav'):
        nav.decompose()

    # Remove elements used as navigation by role or aria labels
    for nav in soup.find_all(attrs={"role": "navigation"}):
        nav.decompose()
    for nav in soup.find_all(attrs={"aria-label": True}):
        al = nav.get('aria-label', '').lower()
        if 'nav' in al or 'menu' in al or 'navigation' in al:
            nav.decompose()

    # Remove common header/nav containers by id/class heuristics
    nav_selectors = ['nav', 'menu', 'header', 'masthead', 'site-header', 'site-nav', 'global-nav', 'main-nav']
    for cls in nav_selectors:
        for el in soup.find_all(attrs={'id': lambda v: v and cls in v.lower()}):
            el.decompose()
        for el in soup.find_all(attrs={'class': lambda v: v and any(cls in c.lower() for c in v)}):
            el.decompose()

def remove_images(soup):
    """
    removes images from the html data
    """    
    images = soup.find_all('img')
    for image in images:
        image.decompose()

def remove_footer(soup):
    """
    removes the footer from the html data
    """
    footer = soup.find('footer')
    if footer:
        footer.decompose()

def remove_scripts(soup):
    """
    removes scripts from the html data
    """
    scripts = soup.find_all('script')
    for script in scripts:
        script.decompose()

def remove_styles(soup):
    """
    removes styles from the html data
    """
    styles = soup.find_all('style')
    for style in styles:
        style.decompose()

def remove_extraneous_data(soup):
    """
    removes extraneous data from the html data
    """
    remove_nav(soup)
    remove_images(soup)
    remove_footer(soup)
    remove_scripts(soup)
    remove_styles(soup)

    # Remove common header and skip-link elements that may remain
    for header in soup.find_all('header'):
        header.decompose()
    for skip in soup.find_all('a', attrs={'class': lambda v: v and 'skip' in v.lower()}):
        skip.decompose()

    return soup