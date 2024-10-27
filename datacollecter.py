import requests


pip install browser - cookie3

import browser_cookie3
cj = browser_cookie3.chrome()
url = 'https://stathead.com/basketball/pgl_finder.cgi?request=1&match=game&order_by_asc=0&order_by=pts&year_min=1961&year_max=2022&is_playoffs=N&lg_id=NBA&age_min=0&age_max=99&season_start=1&season_end=-1'



response = requests.get(url,cookies = cj)


html = response.text

