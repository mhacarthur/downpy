#!/usr/bin/python
import os
import time
import requests
import pandas as pd

start_time = time.time()

class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)
    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (original_parsed.hostname != redirect_parsed.hostname) and \
                        redirect_parsed.hostname != self.AUTH_HOST and \
                        original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return

def download_my_urls(list_of_urls, out_dir, username, password):
    session = SessionWithHeaderRedirection(username, password)
    n_urls = len(list_of_urls)
    for iiu in range(n_urls):
        print(iiu)
        url = list_of_urls[iiu]
        path_and_name = os.path.join(out_dir, url[url.rfind('/') + 1:])
        try:
            # submit the request using the session
            response = session.get(url, stream=True)
            # print(response.status_code)
            # raise an exception in case of http errors
            response.raise_for_status()
            # save the file
            with open(path_and_name, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fd.write(chunk)
        except requests.exceptions.HTTPError as e:
            # handle any errors here
            print(e)

def download_my_urls_v2(url, out_file, username, password):
    try:
        session = SessionWithHeaderRedirection(username, password)
        response = session.get(url, stream=True)
        response.raise_for_status()
        with open(out_file, 'wb') as fd:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    fd.write(chunk)
    except requests.exceptions.HTTPError as e:
            print(e)

username = os.environ['EARTH_DATA_USER']
password = os.environ['EARTH_DATA_PASS']

out_dir = os.path.join('..','data','tmpa_raw_data')

if not os.path.exists(out_dir):
    print('download_tmpa_data WARNING: output folder not found-must create it!')
else:
    print(f'Data download to: {out_dir}')

urls_folder = os.path.join('..','codes_download_data','tmpa_urls')
url_dir = os.path.join(urls_folder,'urls_2008_2016.csv')
URLS = pd.read_csv(url_dir)

URLS = URLS.sort_values(by=['year'])
URLS = URLS.reset_index(drop=True)

print('Number of url files {}'.format(len(URLS)))
for iiy in range(len(URLS)):
    name_tmp = URLS['url'][iiy].split('//')[-1].split('/')[-1]
    out_file = os.path.join('..','data','tmpa_raw_data',name_tmp)

    if not os.path.isfile(out_file):
        download_my_urls_v2(URLS['url'][iiy], out_file, username, password)
    else:
        print(f'File exists')
        continue

execution_time = time.time() - start_time
print("---execution time was %s minutes ---" % (execution_time/60))
