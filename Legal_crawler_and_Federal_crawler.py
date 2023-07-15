import copy
import csv
import os
import re
import shutil

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import time
from datetime import datetime
from static.strings import PARENT_DIR, TODAY
from collections import deque
import logging
from logging.handlers import RotatingFileHandler

# Read the keywords dataframe
KEYWORDS_DF = pd.read_csv('static/SIGAL Keywords.csv')

# Replace "Keywords" column string value with list value
KEYWORDS_DF["Keywords"] = [row['Keywords'].split(';') for idx, row in KEYWORDS_DF.iterrows()]

# Strip spaces before or after keywords
for idx, row in KEYWORDS_DF.iterrows():
    for i in range(len(row['Keywords'])):
        row['Keywords'][i] = row['Keywords'][i].strip()

# Set webdriver for selenium and set "headless" to avoid opening the browser.
options = webdriver.ChromeOptions()
# options.add_argument("headless")
DRIVER_PATH = "chromedriver"
driver = webdriver.Chrome(executable_path=DRIVER_PATH, options=options)
# sets a global timeout value
# driver.implicitly_wait(10)

# Create a dictionary to store visited URLs
VISITED_URLS = set()


def keyword_detected(text: str,
                     keywords_df: pd.DataFrame) -> pd.DataFrame:
    """
     **Function Description**\n
    Loops through the rows of *"Sigal Keywords.csv"* and checks if keywords in the *"Keywords"* column is found in
    the text. Stores the result (*True* or *False*) in a new column. Stores the detected keywords in another column as a
    single string with a semicolon delimiter (;).
     **Example of Usage**\n
     **Parameter Description**\n
    :param text: section title
    :param keywords_df: keywords dataframe without "Contains keyword(s)" and "Found keyword(s)"
    :return: keywords dataframe with "Contains keyword(s)" and "Found keyword(s)"
    """
    # Replace dashes in text with a space
    text = re.sub(r'-', ' ', text)

    for index, row in keywords_df.iterrows():
        # Get keywords from DataFrame
        keywords = row['Keywords']

        # Found keywords will be added to a list
        found_keywords = list()

        for keyword in keywords:
            # Search for keyword in text and append to found_keywords if found
            if keyword.lstrip().lower() in text.lower():
                found_keywords.append(keyword)

        if len(found_keywords) > 0:
            keywords_df.at[index, "Contains keyword(s)"] = True
            keywords_df.at[index, "Found keyword(s)"] = '; '.join(found_keywords)
        else:
            keywords_df.at[index, "Contains keyword(s)"] = False
            keywords_df.at[index, "Found keyword(s)"] = None

    return keywords_df


def save_to_html(content: BeautifulSoup,
                 filepath: str) -> None:
    """
     **Function Description**\n
    Saves content to HTML file.
     **Example of Usage**\n
     **Parameter Description**\n
    :param content: BeautifulSoup result of a section page which either has keywords in title or has ancestor_keywords
    :param filepath: Path where HTML will be saved.<br>
    :return: None
    """
    # Remove icons from content before saving
    content_to_extract = list()

    if content.find("div", {"class": "code-options Section"}):
        content_to_extract.append(content.find("div", {"class": "code-options Section"}))
    if content.find("div", {"class": "code-options__buttons"}):
        content_to_extract.append(content.find("div", {"class": "code-options__buttons"}))
    if content.find("div", {"class": "code-options__kebab"}):
        content_to_extract.append(content.find("div", {"class": "code-options__kebab"}))

    for div_tag in content_to_extract:
        div_tag.extract()

    # Disable all links from content before saving
    hyperlinks = content.find_all('a', {'class': 'Jump'})
    for hyperlink in hyperlinks:
        hyperlink.attrs.pop('href', None)

    # prettify content and replace the
    content = content.prettify(formatter=lambda s: s.replace(u'\xa0', '&nbsp;'))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write('<!DOCTYPE html>\n<html>\n<head>\n\t<meta charset="UTF-8">\n</head>\n<body>\n\n')
        f.write(content)
        f.write('\n\n</body>\n</html>')

    logger.info(f"{filepath} has been written!!")


def replace_unwanted_character(string: str) -> str:
    """
     **Function Description**\n
    Replaces all unwanted characters of the input string to "_", and only keep 1 if there are concatenated "_".
     **Example of Usage**\n
     **Parameter Description**\n
    :param string: string of locality, trace or title
    :return: modified string of locality, trace or title
    """
    unwanted_characters = r'#|%|&|{|}|\\|<|>|\*|\?|/|\s|\$|!|"|:|@|\+|`|\||='
    unwanted_characters += r"|\'|;|,|-"  # |\.
    # Remove unwanted characters from string and replace with a hyphen
    string = re.sub(unwanted_characters, '_', string)

    # Replace multiple hyphens with just one
    string = re.sub("_+", "_", string)
    return string


def create_filename(locality: str,
                    trace: str,
                    title: str) -> str:
    """
     **Function Description**\n
    Creates and returns filename.
     **Example of Usage**\n
     **Parameter Description**\n
    :param locality: locality
    :param trace: trace
    :param title: title
    :return: string of "{locality}-{title}-{trace}"
    """

    locality = replace_unwanted_character(locality)
    trace = replace_unwanted_character(trace)
    title = replace_unwanted_character(title)

    # Concatenate locality, trace, and title to make up filename
    filename = f"{locality}-{title}-{trace}"

    # Cut filename if it is too long
    max_length = 250
    if len(filename) > max_length:
        filename = filename[:max_length]

    filename = f"{filename}.html"

    return filename


def create_cof_filename(trace: str,
                        title: str) -> str:
    """
     **Function Description**\n
    Creates and returns filename.
     **Example of Usage**\n
     **Parameter Description**\n
    :param trace: trace
    :param title: title
    :return: string of "{title}-{trace}"
    """

    trace = replace_unwanted_character(trace)
    title = replace_unwanted_character(title)

    # Concatenate locality, trace, and title to make up filename
    filename = f"{title}-{trace}"

    # Cut filename if it is too long
    max_length = 250
    if len(filename) > max_length:
        filename = filename[:max_length]

    filename = f"{filename}.html"

    return filename


def create_filepath(filename: str,
                    category: str,
                    state: str) -> str:
    """
     **Function Description**\n
    Creates and returns filepath.
     **Example of Usage**\n
     **Parameter Description**\n
    :param filename: filename
    :param category: category
    :param state: state
    :return: filepath
    """
    filepath = f"{PARENT_DIR}/output/{category}/{state}/{filename}"

    return filepath


def write_to_metadata(data_dict: dict,
                      category: str,
                      state: str) -> None:
    """
     **Function Description**\n
    Adds a single row to metadata CSV.
     **Example of Usage**\n
     **Parameter Description**\n
    :param data_dict: data_dict includes information:
                    State, Locality, Trace, Title, Filename, URL, Collection Date, Category, Keywords, Ancestor_keywords
    :param category: keywords category
    :param state: state
    :return: None
    """
    filepath = f"{PARENT_DIR}/output/{category}/{state}/Metadata.csv"
    fieldnames = ['State', 'Locality', 'Trace', 'Title', 'Filename', 'URL', 'Collection Date', 'Category', 'Keywords',
                  'Ancestor Keywords']
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Write the header row if the file is empty
        if f.tell() == 0:
            writer.writeheader()

        # Write the record to the CSV file
        writer.writerow(data_dict)
        logger.info(f"Logged in {category}/{state} Metadata summary. :)")


def get_section_content(title: str,
                        soup: BeautifulSoup,
                        soup_copy: BeautifulSoup) -> BeautifulSoup | None:
    """
     **Function Description**\n
    Gets the section content. Extracts table of the soup_copy, if something remains after extracting table in the
    soup_copy, return the 'curr-section' of the original soup, otherwise return None\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param title: section title
    :param soup: soup
    :param soup_copy: copy of soup
    :return:
    """
    # It should return None if the page is a table of contents
    # .extract() is irreversible, so using soup_copy to check if anything else remain after extract()
    content = soup_copy.find('div', {'id': 'curr-section'})
    pattern = re.compile(r'xsl-table.*?')
    ignore = 'ShareDownloadBookmarkPrint'
    div_tags = content.find_all('div', {'class': pattern})
    for i in div_tags:
        i.extract()
    remain = content.text.replace(title, '').replace(ignore, '').strip()
    if remain:
        content = soup.find('div', {'id': 'curr-section'})
    else:
        content = None

    return content


def get_cof_section_content(title: str,
                            soup: BeautifulSoup,
                            soup_copy: BeautifulSoup) -> BeautifulSoup | None:
    """
     **Function Description**\n
    Gets the section content. Extracts table of the soup_copy, if something remains after extracting table in the
    soup_copy, return the 'section' of the original soup, otherwise return None\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param title: section title
    :param soup: soup
    :param soup_copy: copy of soup
    :return:
    """
    # It should return None if the page is a table of contents
    # .extract() is irreversible, so using soup_copy to check if anything else remain after extract()
    content = soup_copy.find('div', {'class': 'section'})
    pattern = re.compile(r'xsl-table.*?')
    ignore = 'ShareDownloadBookmarkPrint'
    div_tags = content.find_all('div', {'class': pattern})
    for i in div_tags:
        i.extract()
    remain = content.text.replace(title, '').replace(ignore, '').strip()
    if remain:
        content = soup.find('div', {'class': 'section'})
    else:
        content = None

    return content


def format_title(string: str) -> str:
    """
     **Function Description**\n
    Removes the unwanted character in original title, and trims unnecessary spaces\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param string: section title
    :return: formatted section title
    """
    # Replace invalid characters with spaces, remove '.' in the end
    string = re.sub(r'§', '', string)  # [^\w\s\-_]
    if string[-1] == ".":
        string = string[:-1]
    # Remove extra spaces and trim leading/trailing spaces
    string = re.sub(r'\s+', ' ', string).strip()

    return string


def get_subtitle(locality_link: str,
                 locality: str) -> list:
    """
     **Function Description**\n
    Gets subtitles of a locality link, wrap each subtitle into a PageInfo object, and puts the PageInfor objects into
    a list\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param locality: the locality name
    :param locality_link: locality link
    :return: a list of PageInfo objects
    """

    list_of_child_url_info = []
    driver.get(locality_link)
    time.sleep(3)
    html = driver.page_source.encode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')

    anchor = soup.find('div', {'class': 'toc-entry__wrap toc-entry__wrap--is-viewed'})
    next_element = anchor.find_next_sibling('div')
    while next_element:
        href, subtitle = next_element.find('a')['href'], next_element.find('a').text
        trace = f"{locality} Overview" + " > " + subtitle

        # Todo: check if keyword for title, if include key words, set get_all_child to True
        # instantiation object for each page and append to the list
        list_of_child_url_info.append(PageInfo(href, subtitle, trace, dict()))
        next_element = next_element.find_next_sibling()
    return list_of_child_url_info


def get_cof_subtitle(url):
    """
     **Function Description**\n
    Gets subtitles of a chapter link, wrap each subtitle into a PageInfo object, and puts the PageInfor objects into
    a list\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param url: chapter links of each title
    :return: a list of PageInfo objects
    """
    list_of_child_url_info = []
    driver.get(url)
    time.sleep(3)
    html = driver.page_source.encode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')

    tbody_tags = soup.find("tbody")
    tr_tag = tbody_tags.find_all('tr', {"class": ""})
    for i in range(len(tr_tag)):
        href, subtitle = tr_tag[i].find('a')['href'], [i.text for i in tr_tag[i].find_all('a')]

        # example: ['Title 1', 'General Provisions', 'view changes']
        # subtitle = f"{subtitle[0]} {subtitle[1]}"
        subtitle_no = subtitle[0]
        subtitle = subtitle[1]
        trace = f"{subtitle_no} {subtitle}"  # subtitle
        list_of_child_url_info.append(PageInfo(href, subtitle, trace, dict()))
    return list_of_child_url_info


def traverse(url: str,
             state: str,
             locality: str,
             base_url: str):
    """
     **Function Description**\n
    Gets the subtitle objects of the input locality link, Puts all the subtitle object into a queue, traverses the queue
     and execute process_page for each object in the queue,\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param url: locality overview link
    :param state: state
    :param locality: locality
    :param base_url: home page of the website. e.g. https://codelibrary.amlegal.com/
    :return:
    """
    if url is None:
        return

    # check if the link is a locality overview link
    overview_page_pattern = r"https://codelibrary\.amlegal\.com/codes/[a-zA-Z_]+/latest/overview"
    list_of_subtitle_page = list()
    if re.match(overview_page_pattern, url):
        list_of_subtitle_page = get_subtitle(url, locality)
    else:
        logger.info("This is not an overview link!")

    # put the link to VISITED_URLS
    VISITED_URLS.add(url)

    # implement a queue to save all the PageInfo objects
    queue = deque()
    for i in list_of_subtitle_page:
        queue.append(i)
        VISITED_URLS.add(i.url + i.title)

    # while queue is not empty, pop and process the pages
    while queue:
        length = len(queue)
        for i in range(length):
            cur_page = queue.popleft()
            logger.info(f"{cur_page.url} {cur_page.trace}")
            # logger.info(f"[out] current page: {cur_page.url}  {cur_page.title}")
            # logger.info(f"      trace: {cur_page.trace}")
            # logger.info(f"      ancestor_keywords: {cur_page.ancestor_keywords}")
            process_page(queue, cur_page, state, locality, base_url)
    logger.info(f"The traverse and page extraction is completed for {locality}!")


def traverse_cof(url: str,
                 base_url: str):
    """
     **Function Description**\n
    Gets the subtitle objects of the input COF title link, Puts all the subtitle object into a queue, traverses the queue
    and execute process_page_cof for each object in the queue,\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param url: chapter, subchapter, part, etc... link
    :param base_url: home page of the website. e.g. https://www.ecfr.gov/
    :return:
    """
    list_of_subtitle_page = get_cof_subtitle(url)

    VISITED_URLS.add(url)

    # implement a queue to save all the PageInfo objects
    queue = deque()
    for i in list_of_subtitle_page:
        queue.append(i)
        VISITED_URLS.add(i.url)

    # while queue is not empty, pop and process the pages
    while queue:
        length = len(queue)
        for i in range(length):
            cur_page = queue.popleft()
            logger.info(f"{cur_page.url} {cur_page.trace}")
            process_page_cof(queue, cur_page, base_url)
    logger.info(f"The traverse and page extraction is completed for COF!")


def record_ancestor_keywords(ancestor_keywords: dict,
                             keywords_df: pd.DataFrame) -> pd.DataFrame:
    """
     **Function Description**\n
    Check the keyword dictionary is empty or not. If not, extract the value in the
    'Category' column and stores it in a variable called category. Checks if the category key exists, if it does,
    the function sets the value in the 'Contains keyword(s)' column to True and sets the value in the
    'Ancestor_keywords' column to a string that concatenates the values associated with the category key in
    ancestor_keywords. If category key is not present in the ancestor_keywords dictionary, the function sets the
    value in the 'Ancestor_keywords' column to None for the current row\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param ancestor_keywords: is a dictionary
    :param keywords_df: is a Pandas DataFrame
    :return: keywords_df
    """
    if not ancestor_keywords:
        ancestor_keywords = None

    for index, row in keywords_df.iterrows():
        category = row['Category']
        if ancestor_keywords is None:
            keywords_df.at[index, "Ancestor_keywords"] = None
        else:
            if ancestor_keywords.get(category):
                keywords_df.at[index, "Contains keyword(s)"] = True
                keywords_df.at[index, "Ancestor_keywords"] = "; ".join(ancestor_keywords[category])
            else:
                keywords_df.at[index, "Ancestor_keywords"] = None
    return keywords_df


def process_page(queue: deque,
                 page,
                 state: str,
                 locality: str,
                 base_url: str):
    """
     **Function Description**\n
    American Legal Publishing:
    Checks if the current page is a leaf page and any keywords in title. Extracts this page if it's a leaf page and has
    keywords in title, or if it's a leaf page and has keywords in its ancestor page(s) title; Inherits the ancestor
    keywords and creates child page objects if it's not a leaf page\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param queue: the queue stored PageInfo objects
    :param page: the current visiting PageInfo object
    :param state: state
    :param locality: locality
    :param base_url: base_url
    :return: maintain the queue
    """
    driver.get(base_url + page.url)
    time.sleep(3)
    html = driver.page_source.encode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')
    soup_copy = copy.copy(soup)
    # find the anchor by using page title
    anchor = soup.find("a", string=page.title)
    # if anchor exist, continue process the page
    if anchor:
        # format the title, this will be used to generate trace
        formatted_title = format_title(page.title)

        # detect the keywords, put the detected result in the dataframe
        keywords_df = keyword_detected(formatted_title, KEYWORDS_DF)

        # sibling_anchor could be used to tell whether there are subpages or not
        sibling_anchor = anchor.parent.next_sibling

        # if there's no sibling_anchor, means current page is a leaf page
        if not sibling_anchor:
            # write ancestor_keywords into keywords_df
            # if page.ancestor_keywords:
            #    record_ancestor_keywords(page.ancestor_keywords, keywords_df)
            keywords_df = record_ancestor_keywords(page.ancestor_keywords, keywords_df)
            # if current leaf page contains keywords, extract it.
            if keywords_df['Contains keyword(s)'].any():
                content = get_section_content(title=page.title, soup=soup, soup_copy=soup_copy)
                if content is not None:
                    for index, row in keywords_df[keywords_df['Contains keyword(s)'] == True].iterrows():
                        # Get category from row
                        category = row['Category']
                        # Get found keywords from row
                        found_keywords = row['Found keyword(s)']
                        # get ancestor_keywords from row
                        ancestor_keywords = row["Ancestor_keywords"]

                        filename = create_filename(locality=locality,
                                                   trace=page.trace[:-(len(formatted_title) + 3)],
                                                   title=formatted_title)

                        filepath = create_filepath(filename=filename,
                                                   category=category,
                                                   state=state)

                        # This represents the row to be added to the metadata
                        data_dict = {"State": state,
                                     "Locality": locality,
                                     "Trace": page.trace[:-(len(formatted_title) + 3)],
                                     "Title": formatted_title,
                                     "Filename": filename,
                                     "URL": base_url + page.url,
                                     "Collection Date": TODAY,
                                     "Category": category,
                                     "Keywords": found_keywords,
                                     "Ancestor Keywords": ancestor_keywords}

                        write_to_metadata(data_dict=data_dict,
                                          category=category,
                                          state=state)

                        save_to_html(content=content,
                                     filepath=filepath)

        # if the current page has subpage(s)
        else:
            # deep copy the ancestor_keywords dict of this page, which need to be inherited by its child page(s)
            heritage_keywords = copy.deepcopy(page.ancestor_keywords)

            # if current page contain keywords
            if keywords_df['Contains keyword(s)'].any():
                logger.info(f"=====> {formatted_title} has child page(s), "
                            f"the ancestor_keywords of this page is:\n  {page.ancestor_keywords} ")

                # iterate the keywords by category
                for index, row in keywords_df[keywords_df['Contains keyword(s)'] == True].iterrows():
                    category = row['Category']
                    found_keywords = row['Found keyword(s)']
                    # change found_keywords to set
                    found_keywords_set = set(found_keywords.split("; "))
                    # add the keywords to the inherited keywords dict (in corresponding category)
                    heritage_keywords.setdefault(category, set()).update(found_keywords_set)

                logger.info(f"=====> after new keywords appended, "
                            f"the ancestor keywords for its child pages is:\n  {heritage_keywords}\n")

            # for each child page, create PageInfo object and append to the queue
            for i in sibling_anchor:
                subtitle_anchor = i.find("div", {"class": "toc-entry__wrap"})
                if subtitle_anchor:
                    for j in subtitle_anchor.select('a.toc-link'):
                        href, title = j['href'], j.text
                        # filter out title with “Repealed” or “Expired”
                        if title and not skip_title(title):
                            formatted_title = format_title(title)
                            trace = page.trace + " > " + formatted_title
                            key = href + title
                            if key not in VISITED_URLS:
                                VISITED_URLS.add(key)
                                # logger.info(f"[in] new child page: {href}  {formatted_title}")
                                # logger.info(f"trace of this child page: {trace}")
                                queue.append(PageInfo(href, title, trace, heritage_keywords))
                            else:
                                logger.info(f"!!! Found repeat in visited set: {href}")
                        else:
                            logger.info(f"Skipped title: {href} {title}")


def process_page_cof(queue: deque,
                     page,
                     base_url: str):
    """
     **Function Description**\n
    Code of Federal Regulations:
    Checks if the current page is a leaf page and any keywords in title. Extracts this page if it's a leaf page and has
    keywords in title, or if it's a leaf page and has keywords in its ancestor page(s) title; Inherits the ancestor
    keywords and creates child page objects if it's not a leaf page\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param queue: the queue stored PageInfo objects
    :param page: the current visiting PageInfo object
    :param base_url: base_url
    :return: maintain the queue
    """
    driver.get(base_url + page.url)
    time.sleep(3)
    html = driver.page_source.encode("utf-8")
    soup = BeautifulSoup(html, 'html.parser')
    soup_copy = copy.copy(soup)
    state = "Federal"
    locality = ""
    # find the anchor by using page title
    anchor = soup.find("td", string=page.title)
    # if anchor exist, continue process the page

    # format the title, this will be used to generate trace
    formatted_title = format_title(page.title)

    # detect the keywords, put the detected result in the dataframe
    keywords_df = keyword_detected(formatted_title, KEYWORDS_DF)
    if not anchor:
        keywords_df = record_ancestor_keywords(page.ancestor_keywords, keywords_df)

        # if current leaf page contains keywords, extract it.
        if keywords_df['Contains keyword(s)'].any():

            content = get_cof_section_content(title=page.title, soup=soup, soup_copy=soup_copy)
            if content is not None:

                for index, row in keywords_df[keywords_df['Contains keyword(s)'] == True].iterrows():
                    # Get category from row
                    category = row['Category']
                    # Get found keywords from row
                    found_keywords = row['Found keyword(s)']
                    # get ancestor_keywords from row
                    ancestor_keywords = row["Ancestor_keywords"]

                    filename = create_cof_filename(trace=page.trace[:-(len(formatted_title) + 3)],
                                                   title=formatted_title)

                    filepath = create_filepath(filename=filename,
                                               category=category,
                                               state=state)

                    # This represents the row to be added to the metadata
                    data_dict = {"State": state,
                                 "Locality": locality,
                                 "Trace": page.trace[:-(len(formatted_title) + 3)],
                                 "Title": formatted_title,
                                 "Filename": filename,
                                 "URL": base_url + page.url,
                                 "Collection Date": TODAY,
                                 "Category": category,
                                 "Keywords": found_keywords,
                                 "Ancestor Keywords": ancestor_keywords}

                    write_to_metadata(data_dict=data_dict,
                                      category=category,
                                      state=state)

                    save_to_html(content=content,
                                 filepath=filepath)

    # if the current page has subpage(s)
    else:
        # deep copy the ancestor_keywords dict of this page, which need to be inherited by its child page(s)
        heritage_keywords = copy.deepcopy(page.ancestor_keywords)

        # if current page contain keywords
        if keywords_df['Contains keyword(s)'].any():
            logger.info(f"=====> {formatted_title} has child page(s), "
                        f"the ancestor_keywords of this page is:\n  {page.ancestor_keywords} ")

            # iterate the keywords by category
            for index, row in keywords_df[keywords_df['Contains keyword(s)'] == True].iterrows():
                category = row['Category']
                found_keywords = row['Found keyword(s)']
                # change found_keywords to set
                found_keywords_set = set(found_keywords.split("; "))
                # add the keywords to the inherited keywords dict (in corresponding category)
                heritage_keywords.setdefault(category, set()).update(found_keywords_set)

            logger.info(f"=====> after new keywords appended, "
                        f"the ancestor keywords for its child pages is:\n  {heritage_keywords}\n")

        # locate tbody to get child pages
        sibling_anchor = anchor.parent.next_sibling
        tbody = sibling_anchor.find("tbody")

        # for each child page, create PageInfo object and append to the queue
        for child in tbody.children:
            if child.name == 'tr' and 'toggler-active' in child.get('class', []):
                href = child.find('a')['href']
                title_no = child.find('a').text
                if not child.find('td', {'class': 'description'}):
                    continue
                title = child.find('td', {'class': 'description'}).text
                if title:
                    cur_trace = f"{title_no} {title}"
                    trace = page.trace + " > " + cur_trace
                    if href not in VISITED_URLS:
                        VISITED_URLS.add(href)
                        queue.append(PageInfo(href, title, trace, heritage_keywords))
                    else:
                        logger.info(f"!!! Found repeat in visited set: {href}")


class PageInfo:
    def __init__(self,
                 url: str,
                 title: str,
                 trace: str,
                 ancestor_keywords: dict):
        """
         **Function Description**\n
        Instantiates object with the page's info include url, title, trace, ancestor_keywords
        :param url: url of this page
        :param title: title of this page
        :param trace: trace of this page
        :param ancestor_keywords: ancestor_keywords of this page
        """
        self.url = url
        self.title = title
        self.trace = trace
        self.ancestor_keywords = ancestor_keywords


def skip_title(title: str) -> bool:
    """
     **Function Description**\n
    Creates a regular expression to check if the title include certain words that indicate the page content is unwanted
     **Example of Usage**\n
         title = "3.1 REPEALED"
         return True
     **Parameter Description**\n
    :param title: title
    :return: True if the title contains any of the specified keywords, otherwise False
    """

    keywords = ['Repealed', 'Expired']

    # Creates a regular expression pattern that matches any of the keywords in any form (case-insensitive)
    pattern = re.compile("|".join(keyword.lower() + r"\b" for keyword in keywords), re.IGNORECASE)

    if pattern.search(title):
        return True
    else:
        return False


def create_category_directories(keywords_df: pd.DataFrame) -> None:
    """
     **Function Description**\n
    Creates a directory for each category.\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param keywords_df: keywords dataframe
    :return: None
    """
    # Create directory for each category
    for index, row in keywords_df.iterrows():
        category = row["Category"]

        # Declare directory for the category
        directory = f"{PARENT_DIR}/output/{category}"

        # If directory does not exist, create it
        if not os.path.exists(directory):
            os.mkdir(directory)


def create_state_directories(keywords_df: pd.DataFrame,
                             state: str) -> None:
    """
     **Function Description**\n
    Creates a directory for each state. It overwrites the directory if it already exists. A CSV for the Metadata is also
    created within each category folder.\n
    Reference: https://stackoverflow.com/questions/11660605/how-to-overwrite-a-folder-if-it-already-exists-when-creating-it-with-makedirs
     **Example of Usage**\n
     **Parameter Description**\n
    :param keywords_df: keywords dataframe
    :param state: state
    :return: None
    """
    # Create state directory within each category
    for index, row in keywords_df.iterrows():
        category = row["Category"]

        # Declare directory for the state
        state_directory = f"{PARENT_DIR}/output/{category}/{state}"

        # If the state_directory exists, remove it and recreate it
        if os.path.exists(state_directory):
            shutil.rmtree(state_directory)
        os.mkdir(state_directory)

        metadata_directory = f"{state_directory}/Metadata.csv"

        with open(metadata_directory, "w") as f:
            # Write Metadata CSV to meta_directory with the column labels
            fieldnames = ['State', 'Locality', 'Trace', 'Title', 'Filename', 'URL', 'Collection Date', 'Category',
                          'Keywords', 'Ancestor Keywords']
            writer = csv.writer(f)
            writer.writerow(fieldnames)


def get_state_locality_anchors(link: str,
                               base_url: str) -> dict:
    """
     **Function Description**\n
    Soups the input link, creates a dictionary to store the state/locality URLs and corresponding title of that input
    link. \n
     **Example of Usage**\n
     **Parameter Description**\n
    :param link: American Legal home page or State level page
    :param base_url: base_url
    :return: the dictionary of all sub-URLs in the input page
    """
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    anchors = soup.find('div', {'class': 'browse-columns roboto'}).find_all('a', {'class': 'browse-link roboto'})
    # Initiate a dictionary, to save the sub-page title and url
    anchor_links = dict()
    for anchor in anchors:
        anchor_name, anchor_url = anchor.text, base_url + anchor['href']
        anchor_links[anchor_name] = anchor_url
    return anchor_links


def main(base_url: str,
         keywords_df: pd.DataFrame):
    """
     **Function Description**\n
    This is the main function.\n
     **Example of Usage**\n
     **Parameter Description**\n
    :param base_url: home page url (American Legal Publishing)
    :param keywords_df: keywords dataframe
    :return:
    """
    state_urls = get_state_locality_anchors(base_url, base_url)
    user_input = input("If you want Legal American Publishing, Enter a specific state (case-sensitive) or 'all': \n"
                       "If you want Code Of Federal Regulations, Enter 'Federal': \n")

    if user_input == "Federal":
        # Create directory for each category
        create_category_directories(keywords_df=KEYWORDS_DF)

        # Create directory for state (for each category)
        create_state_directories(keywords_df=KEYWORDS_DF,
                                 state=user_input)

        # cof_base_url: home page url (Code Of Federal Regulations)
        cof_base_url = "https://www.ecfr.gov"
        cof_url = cof_base_url
        logger.info(f"Code Of Federal crawling start!")
        traverse_cof(cof_url, cof_base_url)

    if user_input not in state_urls and user_input != "all":
        logger.info("Invalid state name.")

    elif user_input == "all":
        # Create directory for each category
        create_category_directories(keywords_df=keywords_df)

        # Begin crawler for all states...
        for state, state_url in state_urls.items():
            logger.info(f"Crawling start for {state} ...")
            # Create directory for state (for each category)
            create_state_directories(keywords_df=keywords_df,
                                     state=state)

            locality_urls = get_state_locality_anchors(state_url, base_url)
            for locality, locality_url in locality_urls.items():
                logger.info(f"Crawling start for {locality} ...")
                traverse(locality_url, state, locality, base_url)

    else:
        # Create directory for each category
        create_category_directories(keywords_df=KEYWORDS_DF)

        # Create directory for state (for each category)
        create_state_directories(keywords_df=KEYWORDS_DF,
                                 state=user_input)

        # Get state URL from dictionary
        state_url = state_urls[user_input]

        locality_urls = get_state_locality_anchors(state_url, base_url)

        for locality, locality_url in locality_urls.items():
            logger.info(f"Crawling start for {locality} ...")
            traverse(locality_url, user_input, locality, base_url)


def setup_logger(log_file: str) -> object:
    """
     **Function Description**\n
    Instantiates a logger object.\n
    :param log_file:
    :return:
    """
    # create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler for the logger
    # file_handler = logging.FileHandler(log_file)
    file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=1 * 1024 * 1024,
                                       backupCount=10, encoding=None, delay=False)
    file_handler.setLevel(logging.INFO)

    # create a console handler for the logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # create a formatter for the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    # Generate log file name, and create a logger
    log_file = f"{PARENT_DIR}/output/{datetime.now().strftime('%Y%m%d%H%M')}.log"
    # check the output path is exist or not
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = setup_logger(log_file)

    main(base_url='https://codelibrary.amlegal.com',
         keywords_df=KEYWORDS_DF)

    logger.info(f"Crawling completed, all the best!")

# run this code if you just want to test for 1 locality:

# base_url = 'https://codelibrary.amlegal.com'
# link = "https://codelibrary.amlegal.com/codes/los_angeles/latest/overview"
# state = 'California'
# locality = 'Los Angeles'

# log_file = f"output/{datetime.now().strftime('%Y%m%d%H%M')}.log"
# logger = setup_logger(log_file)
# logger.info(f"Crawling start!")

# traverse(link, state, locality, base_url)

# logger.info(f"Crawling completed, all the best!")
