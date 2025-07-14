import praw
import json
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")

def extract_username(url: str) -> str:
    path = urlparse(url).path
    return path.strip("/").split("/")[-1]

def init_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=USER_AGENT
    )

def scrape_user_data(username: str):
    reddit = init_reddit()
    user = reddit.redditor(username)

    data = {
        "username": username,
        "posts": [],
        "comments": []
    }

    print(f"Fetching posts for u/{username}...")
    for post in user.submissions.new(limit=None):
        data["posts"].append({
            "title": post.title,
            "selftext": post.selftext,
            "permalink": f"https://www.reddit.com{post.permalink}",
            "subreddit": str(post.subreddit),
            "created_utc": post.created_utc
        })

    print(f"Fetching comments for u/{username}...")
    for comment in user.comments.new(limit=None):
        data["comments"].append({
            "body": comment.body,
            "permalink": f"https://www.reddit.com{comment.permalink}",
            "subreddit": str(comment.subreddit),
            "created_utc": comment.created_utc
        })

    return data

def save_json(data: dict, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filename}")

def run_scraper(profile_url: str) -> str:
    username = extract_username(profile_url)
    scraped_data = scrape_user_data(username)
    save_json(scraped_data, f"{username}_data.json")
    return username
