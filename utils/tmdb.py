import csv, json, os, time, requests, streamlit as st

def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_id_list(api_key, year, max_retries=2):
    url = f'https://api.themoviedb.org/3/discover/movie?api_key={api_key}&primary_release_year={year}&include_video=false&language=en-US&sort_by=popularity.desc'

    movie_ids = []
    total_pages = 3  # 5 pages of ids = 100 movies
    for page in range(1, total_pages + 1):
        for i in range(max_retries):
            response = requests.get(url + f'&page={page}')
            if response.status_code == 429:
                # If the response was a 429, wait and then try again
                print(
                    f"Request limit reached. Waiting and retrying ({i+1}/{max_retries})")
                time.sleep(2 ** i)  # Exponential backoff

            else:
                # If the response was not a 429, continue
                dict = response.json()
                st.write(dict)
                for film in dict['results']:
                    movie_ids.append(str(film['id']))
                break
    return movie_ids

def write_file(filename, dict):
    csvFile = open(filename, 'a')
    csvwriter = csv.writer(csvFile)
    # unpack the result to access the "collection name" element
    title = dict['title']
    runtime = dict['runtime']
    language_code = dict['original_language']
    release_date = dict['release_date']
    overview = dict['overview']
    all_genres = dict['genres']
    prod_companies = dict['production_companies']

    # Parsing release date
    release_year = release_date.split('-')[0]

    # Converting language
    # try:
    #     language = languages.get(alpha2=language_code).name
    # except KeyError:
    #     language = 'Unknown'

    # Parsing genres
    genre_str = ""
    for genre in all_genres:
        genre_str += genre['name'] + ", "
    genre_str = genre_str[:-2]

    # Parsing keywords (remove non-English words)
    all_keywords = dict['keywords']['keywords']
    keyword_str = ""
    for keyword in all_keywords:
        if is_english(keyword['name']):
            keyword_str += keyword['name'] + ", "
    if keyword_str == "":
        keyword_str = "None"
    else:
        keyword_str = keyword_str[:-2]

    # Parsing watch providers
    watch_providers = dict['watch/providers']['results']
    stream_str, buy_str, rent_str = "", "", ""
    if 'US' in watch_providers:
        watch_providers = watch_providers['US']
        provider_strings = ['flatrate', 'buy', 'rent']
        for string in provider_strings:
            if string not in watch_providers:
                continue

            _str = ""

            for element in watch_providers[string]:
                _str += element['provider_name'] + ", "
            _str = _str[:-2] + " "

            if string == 'flatrate':
                stream_str += _str
            elif string == 'buy':
                buy_str += _str
            else:
                rent_str += _str

    credits = dict['credits']
    actor_list, director_list = [], []

    # Parsing cast
    cast = credits['cast']
    NUM_ACTORS = 5
    for member in cast[:NUM_ACTORS]:
        actor_list.append(member["name"])

    # Parsing crew
    crew = credits['crew']
    for member in crew:
        if member['job'] == 'Director':
            director_list.append(member["name"])

    actor_str = ', '.join(list(set(actor_list)))
    director_str = ', '.join(list(set(director_list)))

    # Parsing production companies
    prod_str = ""
    for company in prod_companies:
        prod_str += company['name'] + ", "
    prod_str = prod_str[:-2]

    result = [title, runtime, language, overview,
              release_year, genre_str, keyword_str,
              actor_str, director_str, stream_str,
              buy_str, rent_str, prod_str]

    # write data
    csvwriter.writerow(result)
    csvFile.close()


def write_ids_csv(ids, year, CSV_HEADER, TMDB_API_KEY):
    # Grab list of ids for all films made in {YEAR}
    movie_list = list(set(ids))

    FILE_NAME = f'./data/{year}_movie_collection_data.csv'

    # Creating file
    with open(FILE_NAME, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

    # Iterate through list of ids to get data
    for id in movie_list:
        dict = get_data(TMDB_API_KEY, id)
        write_file(FILE_NAME, dict)


def get_data(TMDB_API_KEY, Movie_ID, max_retries=2):
    query = 'https://api.themoviedb.org/3/movie/' + Movie_ID + '?api_key='+ TMDB_API_KEY + '&append_to_response=keywords,' + 'watch/providers,credits&language=en-US'

    for i in range(max_retries):
        response = requests.get(query)
        if response.status_code == 429:
            # If the response was a 429, wait and then try again
            print(
                f"Request limit reached. Waiting and retrying ({i+1}/{max_retries})")
            time.sleep(2 ** i)  # Exponential backoff
        else:
            dict = response.json()
            return dict
