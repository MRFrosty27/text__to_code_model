import requests
from bs4 import BeautifulSoup
import csv
import time
import re


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

Brand_names = ('Toyota','Audi','BMW','BYD','Bentley','Ferrari','Fiat','Ford','GWM','Haval','Honda','Hyundai','Isuzu','JAECOO','Jaguar','Jeep','Jetour','Kia','Lamborghini','Land Rover','Lexus','MINI','Mahindra','Maserati','Mazda','McLaren','Mercedes-AMG','Mercedes-Benz','Mercedes-Maybach','Mitsubishi','Nissan','OMODA','Porche','Rolls-Royce','Susuki','Volkswagen','Volvo')

def scrape_autotrader(brand):
    data = []
    base_url = f"https://www.autotrader.co.za/cars-for-sale/{brand}"
    
    response = requests.get(base_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    base_element = soup.find('div',class_='e-results-body__e2m4a-Isq-Q-')

    max_page_number = soup.find('span',class_='e-text__2BikEDs6gKY-')

    for page in range(1, int(max_page_number.string)):
        url = f"{base_url}?pagenumber={page}" if page > 1 else base_url
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        data = []

        if response.status_code != 200:
            print(f"Error fetching AutoTrader page {page}: {response.status_code}")
            continue

        listings = base_element.find_all("a")
        
        for listing in listings:
            price = listing.find('span',class_='e-top-section__xefYFe--jN8-')
            price = price.find('span',class_='e-result-tile-content__v-FUMHk1LrQ-')
            price = price.find('span',class_='e-result-tile-inner-wrapper__5dMTluUlMkE-')

            milage = price.find('span',class_='e-content-wrapper__bKOvAQCE3Wk-')

            price = price.find('span',class_='e-price-wrapper__Sbx7fC9n-Pg-')

            price_rating = price.find('span',class_='e-indicator-wrapper__x5dgy0Ik7EE-')
            
            price = price.find('span',class_='e-price-content__ckwLbmj-9F4-')
            price = price.find('h2',class_='e-price__Yd0hRaoM6rg-')

            price_rating = price_rating.find('span',class_='b-listing-indicators__TPkItAg1hFc-')
            price_rating = price_rating.find('span',class_='e-price-rating-wrapper__gyb4W0NEwpw-')
            price_rating = price_rating.find('span',class_='e-price-rating__mF-CrAabl8M- ')
            price_rating = price_rating.find('div',class_='e-badges-wrapper__xto9KwpYIZI-')
            price_rating = price_rating.find('span',class_='b-price-rating__e1nRsMxXb4Q- m-fair__cm-id0UzVgk-')
            price_rating = price_rating.find('span',class_='e-rating-text__WFA6pv-cUx4-')

            milage = milage.find('span',class_='e-content-top__-lDSu-7N26c-')

            model = milage.find('span', class_='e-highlight-wrapper-top__Iqx3qBN1yFg-')
            
            milage = milage.find('span',class_='e-highlight-wrapper-bottom__dMpsqTaXuTA-')
            milage = milage.find('span',class_='b-vehicle-specifications__5abDq5x647I-')
            milage = milage.find('span',class_='b-vehicle-spec-tag__D9MBKkRYlKw- m-small__knZ1OuQHeEU- m-auto-desktop-hd-size-increase__zShIQp3k-pE-')
            milage = milage.find('span',class_='e-text__ZIvs3UMtWxs-')

            variant = model.find('span', class_='e-variant-title__4VK0xJjilbI-')

            model = model.find('span',class_='e-make-model-title__x6ofmTGPOrM-')

            data.append({"brand": brand, "model": model[4:],"Variant": variant,"Year":model[0:3],"Milage": milage,"Price_rating": price_rating, "Price": price})
        time.sleep(2)  # Be polite
    
    return data

def scrape_cars_co_za(brand):
    data = []
    base_url = "https://www.cars.co.za/usedcars/"
    response = requests.get(base_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    max_pages = soup.find('a',class_='mantine-focus-auto Pagination_control__Wc8lz m_326d024a mantine-Pagination-control m_87cf2631 mantine-UnstyledButton-root')

    for page in range(1, max_pages): 
        url = base_url if page == 1 else f"{base_url}?make_model_variant={brand}&sort=sort_rank&price_type=listing_price&P={page}"
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        if response.status_code != 200:
            print(f"Error fetching Cars.co.za page {page}: {response.status_code}")
            continue
        listings = soup.find_all('div',class_='m_8bffd616 mantine-Flex-root __m__-r6il')
        
        for listing in listings:
            price = listing.find('h3', class_='title_root__U3F4v vehicle-price m_8a5d1357 mantine-Title-root __m__-r8ad')

            model = listing.find('h3',class_='title_root__U3F4v cy-result-title m_8a5d1357 mantine-Title-root __m__-r8a9')
            model = model.strip(brand)
            
            variant = listing.find('p',class_='mantine-focus-auto m_b6d8b162 mantine-Text-root')

            year = listing.find('span', class_='mantine-focus-auto m_b6d8b162 mantine-Text-root')

            milage = listing.find('span', class_='mantine-focus-auto m_b6d8b162 mantine-Text-root')

            price_rating = listing.find('span', class_='m_5add502a mantine-Badge-label')

            data.append({"brand": brand, "model": model,"Variant": variant,"Year":year,"Milage": milage,"Price_rating": price_rating, "Price": price})

        print(f"Cars.co.za page {page} scraped - {len(listings)} listings found")
        time.sleep(2)
    
    return data

# Main execution
all_data = []

for brand in Brand_names:
    all_data.extend(scrape_autotrader(brand))

for brand in Brand_names:
    all_data.extend(scrape_cars_co_za())

# Save to CSV
csv_file = "south_african_car_market_dataset.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["brand", "model", "price", "source"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"\nScraping complete! {len(all_data)} cars saved to {csv_file}")