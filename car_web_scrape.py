import requests
from bs4 import BeautifulSoup
import csv
import time


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

    max_page_number = soup.find_all('span',class_='e-text__2BikEDs6gKY-')
    max_page_number = max_page_number[-1].get_text()
    print(f"max pages- {max_page_number}")

    for page in range(1, int(max_page_number)):
        print(f'Next page\n page {page} out of {max_page_number}')
        url = f"{base_url}?pagenumber={page}" if page > 1 else base_url
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        data = []

        if response.status_code != 200:
            print(f"Error fetching AutoTrader page {page}: {response.status_code}")
            continue

        listings = base_element.find_all("a")
        
        for listing in listings:
            print('next listing')
            try:
                price = listing.find('h2',class_='e-price__Yd0hRaoM6rg-')
                print(price)
                price = price.get_text()

                price_rating = listing.find('span',class_='e-rating-text__WFA6pv-cUx4-')
                print(price_rating)
                price_rating = price_rating.get_text()

                milage = listing.find('span',class_='e-text__ZIvs3UMtWxs-')
                print(milage)
                milage = milage.get_text()

                variant = listing.find('span', class_='e-variant-title__4VK0xJjilbI-')
                print(variant)
                variant = variant.get_text()

                model = listing.find('span',class_='e-make-model-title__x6ofmTGPOrM-')
                print(model)
                model = model.get_text()

                data.append({"brand": brand, "model": model[4:],"Variant": variant,"Year":model[0:3],"Milage": milage,"Price_rating": price_rating, "Price": price})
            except:
                print('Failed to get data')
                continue
            
        time.sleep(2)  # Be polite
    print('next brand')
    return data

def scrape_cars_co_za(brand):
    data = []
    base_url = "https://www.cars.co.za/usedcars/"
    response = requests.get(base_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    max_pages = soup.find('a',class_='mantine-focus-auto Pagination_control__Wc8lz m_326d024a mantine-Pagination-control m_87cf2631 mantine-UnstyledButton-root')
    max_pages = max_pages.get_text()

    for page in range(1, int(max_pages)): 
        print(f'Next page\npage {page} out of {max_pages}\n')
        url = base_url if page == 1 else f"{base_url}?make_model_variant={brand}&sort=sort_rank&price_type=listing_price&P={page}"
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        if response.status_code != 200:
            print(f"Error fetching Cars.co.za page {page}: {response.status_code}")
            continue
        listings = soup.find_all('div',class_='m_8bffd616 mantine-Flex-root __m__-r6il')
        
        for listing in listings:
            try:
                print('next listing')
                price = listing.find('h3', class_='title_root__U3F4v vehicle-price m_8a5d1357 mantine-Title-root __m__-r8ad')
                price = price.get_text()

                model = listing.find('h3',class_='title_root__U3F4v cy-result-title m_8a5d1357 mantine-Title-root __m__-r8a9')
                model = model.get_text()
                
                variant = listing.find('p',class_='mantine-focus-auto m_b6d8b162 mantine-Text-root')
                variant = variant.get_text()

                year = listing.find('span', class_='mantine-focus-auto m_b6d8b162 mantine-Text-root')
                year = year.get_text()

                milage = listing.find('span', class_='mantine-focus-auto m_b6d8b162 mantine-Text-root')
                milage = milage.get_text()

                price_rating = listing.find('span', class_='m_5add502a mantine-Badge-label')
                price_rating = price_rating.get_text()

                data.append({"brand": brand, "model": model,"Variant": variant,"Year":year,"Milage": milage,"Price_rating": price_rating, "Price": price})
            except:
                print('failed to get data')
                continue
        print(f"Cars.co.za page {page} scraped - {len(listings)} listings found")
        time.sleep(2)
    
    return data

# Main execution
all_data = []

for brand in Brand_names:
    print('scraping next brand from auto trader')
    all_data.extend(scrape_autotrader(brand))

for brand in Brand_names:
    print('scraping next brand from cars.co.za')
    all_data.extend(scrape_cars_co_za())

# Save to CSV
csv_file = "south_african_car_market_dataset.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["brand", "model", "price", "source"])
    writer.writeheader()
    writer.writerows(all_data)

print(f"\nScraping complete! {len(all_data)} cars saved to {csv_file}")