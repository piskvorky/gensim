"""Generate changelog entries for all PRs merged since the last release."""
import requests

get = requests.get('https://api.github.com/repos/RaRe-Technologies/gensim/releases')
get.raise_for_status()

last_release_date = get.json()[0]['published_at'][:10]


def g():
    page = 1
    done = False

    while not done:
        get = requests.get(
            'https://api.github.com/repos/RaRe-Technologies/gensim/pulls',
            params={'state': 'closed', 'page': page},
        )
        get.raise_for_status()
        pulls = get.json()
        if not pulls:
            break

        for i, pr in enumerate(pulls):
            if pr['created_at'] < last_release_date:
                done = True
                break

            if pr['merged_at'] and pr['merged_at'] > last_release_date:
                yield pr

        page += 1


for pr in g():
    pr['user_login'] = pr['user']['login']
    pr['user_html_url'] = pr['user']['html_url']
    print('* %(title)s [#%(number)d](%(html_url)s), __[@%(user_login)s](%(user_html_url)s)__' % pr)
