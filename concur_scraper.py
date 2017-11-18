from selenium import webdriver
from time import sleep
import re
from fuzzywuzzy import fuzz
class TimedOutExc(Exception):
    pass
def deadline(timeout, *args):
    '''
    give a deadline to subject function, usage:
    @deadline(900)
    function will raise error TimedOutExc after  seconds
    '''
    def decorate(f):
        def handler(signum, frame):
            raise TimedOutExc()
        def new_f(*args):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            return f(*args)
            signa.alarm(0)
        new_f.__name__ = f.__name__
        return new_f
    return decorate
#rewrite this in a class structure
#normalize before comparison
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1200x600')
    
    driver = webdriver.Chrome(chrome_options=options)
    return driver
def login_into_concur(driver):
    driver.get('https://www.concursolutions.com')
    driver.implicitly_wait(10)
    email = driver.find_element_by_css_selector('#userid')
    password = driver.find_element_by_css_selector('#password')
    login = driver.find_element_by_css_selector('#btnSubmit')
    email.send_keys('WSadminRocketrip@kellogg.com')
    password.send_keys('Welcome1')
    login.click()
    return None
def _parse_names_from_list(results_list):
    return [item.split('\n')[0] for item in results_list]
def _parse_results_raw_string_into_list(results_raw_text_str):
    return re.compile("\d+[.]\s").split(results_raw_text_str)
def _compare_and_rank_names(target_name, hotel_names, comparison_func=fuzz.ratio,):
    #this one liner typically very bad style but I'm too lazy to break it down
    return sorted(zip([comparison_func(target_name, name) for name in hotel_names],hotel_names), key=lambda x:x[0], reverse=True)
# @deadline(60)
def get_results_from_hotel_search(driver, city_input, state_abbrev_input, start_date, end_date):
    something = city_input + ',' + state_abbrev_input
    driver.find_element_by_css_selector('#hotelTab').click()
    driver.find_element_by_css_selector('#searchRefPoint').send_keys(something)
    driver.find_element_by_css_selector('#hotelStartDate').send_keys(start_date)
    driver.find_element_by_css_selector('#hotelEndDate').send_keys(end_date)
    driver.find_element_by_css_selector('#btnHotelLaunchWizard').click()
    
    #this pause is necessary otherwise it won't see the popup and it breaks
    sleep(10)
    
    driver.find_element_by_css_selector('#geocodechoosebutton').click()
    results = driver.find_element_by_css_selector('#dynamicTable')
    return _parse_results_raw_string_into_list(results.text)
if __name__ == '__main__':
    driver = init_driver()
    login_into_concur(driver)
    #wrap and loop on this code
    results = get_results_from_hotel_search(driver, 'memphis', 'TN', '11/23/17', '11/30/17')
    hotel_names = _parse_names_from_list(results)
    ranked_list = _compare_and_rank_names('Homewood Suites Memphis East', hotel_names)
    print (ranked_list)
    '''
    to obtain such output from usuch input, just write a wrapper function 
    and store results in a dictionary with out the structure 
    described in the output: {str keys: order_lists}
    
    Homewood Suites Memphis East (232), 
    Embassy Suites Memphis (120), 
    Courtyard Memphis Airport (109), Doubletree By Hilton Hotel Memphis (62)
    {
    "Homewood Suites Memphis East (232)":[Homewood, ..., N]
        "Embassy Suites Memphis":
        "Doubletree By Hilton Hotel Memphis":
        }
    '''