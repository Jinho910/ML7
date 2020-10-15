import requests


# q_continue = True
def restart():
    yn = input("continue? (y/n)  ")
    if yn.lower() == 'y':
        main()
    elif yn.lower() == 'n':
        return
    else:
        print('invalid answer')
        restart()

def main():
    s = input("콤마로 구별된  url을 입력하세요 : ")
    URLs = s.lower().strip().split(',')
    for i, s in enumerate(URLs):
        s = s.strip()

        if s.find('.') == -1:
            print(f"{s} is not a valid url.")
            continue
        if s.find('http') == -1:
            s = "http://" + s
        # print(s)
        try:
            html = requests.get(s)
            if html.status_code == 200:
                print(f"{s} is up")
            else:
                print(f"{s} down {html.status_code}")
        except:
            print(f"{s} down")

    restart()



main()
