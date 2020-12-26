import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}

def get_job(word):
    URL = f"https://remoteok.io/remote-dev+{word}-jobs"
    # URL = "https://remoteok.io/remote-dev+python-jobs"
    # print(URL)
    # URL = f"https://weworkremotely.com/remote-jobs/search?term={word}"

    html = requests.get(URL,headers=headers)
    soup = BeautifulSoup(html.text, 'html.parser')
    rw_job = []
    jobs_board=soup.find('table', {"id":"jobsboard"})
    # jobs_board = soup.find('div', {"class": "container"})

    jobs = jobs_board.find_all("tr", {"class": "job"})
    for job in jobs:
        link=job['data-url']
        company=job['data-company']
        title=job['data-search']
        rw_job.append({'title':title, "company":company, "link":f"https://remoteok.io{link}"})

    return rw_job


if __name__ == "__main__":
    jobs = get_job("python")
    print(jobs)
    # [print(job['title'], job['company'], job['link']) for job in jobs]
