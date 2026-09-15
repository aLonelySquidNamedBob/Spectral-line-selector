import requests
import pandas as pd

def get_atomic_data(session: requests.Session, elems: list[str], lls: str, low: float, high: float):
    headers = {
        'Accept': 'text/plain, */*; q=0.01',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,fr;q=0.7,de;q=0.6',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json; charset=UTF-8',
        'DNT': '1',
        'Origin': 'https://websme-dev.chetec-infra.eu',
        'Pragma': 'no-cache',
        'Referer': 'https://websme-dev.chetec-infra.eu/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }

    json_data = {
        'elements': elems,
        'linelist': lls,
        'wl_limit_low': low,
        'wl_limit_high': high,
    }

    response = session.post('https://websme.chetec-infra.eu/get_atomic_data', headers=headers, json=json_data)

    atomic_data = pd.DataFrame(response.json()['data'])

    return atomic_data


# elements = ["Sc", "V", "Si", "K", "C", "N", "O", "Al", "Fe1", "Fe2", "Li", "Cu", "Ba", "Nd", "Na", "Ni", "Mg", "Mn", "Ca", "Ti", "Eu", "Pb", "Sr", "Th", "Zr", "Cr", "Co"]
elements = ["Li", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe1", "Fe2", "Co", "Ni", "Cu"]
linelist = "GAIAESOYY"
wl_low = 4000
wl_high = 6640

s = requests.Session()

s.verify = False
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,fr;q=0.7,de;q=0.6',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'DNT': '1',
    'Pragma': 'no-cache',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
s.get('https://websme.chetec-infra.eu/', headers=headers)

lines = get_atomic_data(s, elements, linelist, wl_low, wl_high)
lines.to_csv("./6_python stuff/results/lines_complete_4000_YY.csv", index=False)