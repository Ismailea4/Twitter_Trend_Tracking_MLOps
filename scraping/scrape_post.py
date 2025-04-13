#Webdriver de Selenium qui permet de contrôler un navigateur
from selenium import webdriver

#Permet d'accéder aux différents élements de la page web
from selenium.webdriver.common.by import By

#Pour attendre qu'une condition soit remplie avant de poursuivre l'exécution du script
from selenium.webdriver.support.ui import WebDriverWait

#fournit des conditions d'attente prédéfinies pour être utilisées avec WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.support.ui import Select


#Pour utiliser les fonctionnalités liées à la gestion du temps
import time

import re

from selenium.webdriver.common.keys import Keys

from selenium.webdriver.chrome.options import Options

from selenium.webdriver.chrome.service import Service


import pandas as pd

from datetime import datetime


from bs4 import BeautifulSoup

#Sous fonction pour extraire les données de Twitter
def twitter_time(soup1):
    tweets = soup1.find_all('article',attrs={'data-testid':"tweet"})

    dates=[]
    for tweet in tweets:
        times = tweet.find_all('time')
        try:
            dates.append(times[0].get('datetime'))
        except:
            dates.append('NaN')
            
    return dates

def twitter_user(soup1):
    users = soup1.find_all('div',attrs={'data-testid':'User-Name'})

    username=[]
    for div in users:
        divss = div.find('a')
        print(divss['href'])
        username.append(divss.get('href')[1:])
    return username

def twitter_tweets(soup1):
    tweets = soup1.find_all('article',attrs={'data-testid':"tweet"})

    texts=[]
    for tweet in tweets:
        tweet = tweet.find_all('div',attrs={'data-testid':"tweetText"})
        try:
            texts.append(tweet[0].text)
        except:
            texts.append('NaN')
            
    return texts

def twitter_react(soup1):
    likes = soup1.find_all('div',attrs={'role':'group'})

    reacts=[]
    order = ['replies', 'reposts', 'likes', 'bookmarks', 'views']
    
    for like in likes:
        react_text=like['aria-label']

        # Initialize dictionary to store numbers
        react_stat = {word: 0 for word in order}

        # Find all numbers and words using regex
        matches = re.findall(r"(\d+) (\w+)", react_text)

        # Update numbers dictionary with extracted values
        for match in matches:
            number, word = match
            if word in react_stat:
                react_stat[word] = int(number)
        reacts.append([react_stat[word] for word in order])

        
    react_sorted = [[row[i] for row in reacts] for i in range(len(reacts[0]))]
        
    return react_sorted

def exctract_twitter_data(soup1):
    dictio = {}
    dictio['username']=twitter_user(soup1)
    dictio['date']=twitter_time(soup1)
    dictio['text']=twitter_tweets(soup1)
    react=twitter_react(soup1)
    dictio['replies']=react[0]
    dictio['reposts']=react[1]
    dictio['likes']=react[2]
    dictio['bookmarks']=react[3]
    dictio['views']=react[4]
    
    data = pd.DataFrame(dictio)
    
    return data

#Fonction principale pour scraper les commentaires Twitter
#Cette fonction prend en entrée une URL de tweet et renvoie un DataFrame contenant les commentaires associés à ce tweet.
def scrape_twitter_comments(url):
    start = time.time()
    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless")  # Décommente pour exécuter sans interface graphique
    options.add_argument("--log-level=3")  # Suppresses most logs

    service = Service(log_path='NUL')  # 'NUL' on Windows, '/dev/null' on Linux/Mac

    tweets_df = pd.DataFrame()
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    time.sleep(10)

    # Scroll initial
    total_height = driver.execute_script("return document.body.scrollHeight")
    driver.execute_script(f"window.scrollTo(0, {total_height * 10});")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    wait = WebDriverWait(driver, 20)

    # Cliquer sur "Lire les réponses"
    read_replies = wait.until(EC.presence_of_element_located((By.XPATH, "//a[@data-testid='logged_out_read_replies_pivot']")))
    read_replies.click()

    # Login
    log = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@autocomplete='username']")))
    log.send_keys('mana_isma@hotmail.com')  # <-- Remplace par ton email
    log.send_keys(Keys.RETURN)

    try:
        pseudo = wait.until(EC.presence_of_element_located((By.XPATH, "//input")))
        pseudo.send_keys('Izumiu10')  # <-- Remplace par ton pseudo
        pseudo.send_keys(Keys.RETURN)
    except:
        pass

    password = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@name='password']")))
    password.send_keys('Izumiii0')  # <-- Remplace par ton mot de passe
    password.send_keys(Keys.RETURN)

    # Attente pour chargement des commentaires
    time.sleep(10)
    SCROLL_PAUSE_TIME = 10
    itera = 0
    last_height = driver.execute_script("return document.body.scrollHeight")

    while itera < 200:
        page_content = driver.page_source
        soup = BeautifulSoup(page_content, 'html.parser')

        try:
            df = exctract_twitter_data(soup)  # <-- Assure-toi que cette fonction est définie ailleurs
            tweets_df = pd.concat([tweets_df, df], ignore_index=True)
        except TypeError as e:
            print("Une erreur s'est produite :", e)
        except KeyError:
            print("Erreur de keyerror")

        itera += 200

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            itera += 50
        last_height = new_height

    driver.quit()
    print('execution en :',time.time()-start)
    return tweets_df


def scrape_twitter_comments2(urls, path_to_save):
    first_time_login = True
    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless")
    options.add_argument("--log-level=3")  # Suppresses most logs

    service = Service(log_path='NUL')  # 'NUL' on Windows, '/dev/null' on Linux/Mac
    
    driver = webdriver.Chrome(options=options)

    for url in urls:
        print(f"Processing URL: {url}")
        start = time.time()

        tweets_df = pd.DataFrame()
        driver.get(url)

        time.sleep(20)

        if first_time_login:
            # Scroll initial
            total_height = driver.execute_script("return document.body.scrollHeight")
            driver.execute_script(f"window.scrollTo(0, {total_height * 10});")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            wait = WebDriverWait(driver, 20)

            read_replies = wait.until(EC.presence_of_element_located(
                (By.XPATH, "//a[@data-testid='logged_out_read_replies_pivot']")))
            read_replies.click()


            # Login
            log = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@autocomplete='username']")))
            log.send_keys('mana_isma@hotmail.com')  # <-- Remplace par ton email
            log.send_keys(Keys.RETURN)

            try:
                pseudo = wait.until(EC.presence_of_element_located((By.XPATH, "//input")))
                pseudo.send_keys('Izumiu10')  # <-- Remplace par ton pseudo
                pseudo.send_keys(Keys.RETURN)
            except:
                pass

            password = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@name='password']")))
            password.send_keys('Izumiii0')  # <-- Remplace par ton mot de passe
            password.send_keys(Keys.RETURN)

            first_time_login = False

            time.sleep(30)
                
        # Attente pour chargement des commentaires

        SCROLL_PAUSE_TIME = 15
        itera = 0
        last_height = driver.execute_script("return document.body.scrollHeight")

        while itera < 200:
            page_content = driver.page_source
            soup = BeautifulSoup(page_content, 'html.parser')

            try:
                df = exctract_twitter_data(soup)  # <-- Ta fonction d'extraction
                tweets_df = pd.concat([tweets_df, df], ignore_index=True)
            except TypeError as e:
                print("Une erreur s'est produite :", e)

            itera += 1
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                itera += 50
            last_height = new_height


        # Créer un nom de fichier à partir de l'URL
        filename = path_to_save + url.split('/')[-1] + ".csv"
        # Enregistrer le DataFrame dans un fichier CSV
        tweets_df.to_csv(filename, index=False)
        print(f"{filename} saved. Execution time: {time.time() - start:.2f} seconds\n")
        
    driver.quit()
