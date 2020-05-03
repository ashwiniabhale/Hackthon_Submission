# importing requests package
import requests

def newforcovid():
    # BBC news api
    main_url = "http://newsapi.org/v2/top-headlines?country=in&category=health&apiKey=dbf949d1eea745a6a83a4d45c6206404"

    # fetching data in json format
    open_bbc_page = requests.get(main_url).json()

    # getting all articles in a string article
    article = open_bbc_page["articles"]

    # empty list which will
    # contain all trending news
    results = []
    results_link = []
    for ar in article:
        results.append(ar["title"])
        results_link.append(ar["url"])

    for i in range(len(results)):
        # printing all trending news
        print(i + 1, results[i])
        print("Link of News ", results_link[i])

    # Driver Code


if __name__ == '__main__':
    # function call
    newforcovid()
