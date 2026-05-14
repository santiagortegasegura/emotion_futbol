import os
import tweepy
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Get bearer token
bearer_token = os.getenv("BEARER_TOKEN")

print("\nTOKEN LOADED SUCCESSFULLY\n")

# Create client
client = tweepy.Client(
    bearer_token=bearer_token,
    wait_on_rate_limit=True
)

try:

    response = client.search_recent_tweets(
        query="football lang:es -is:retweet",
        max_results=10
    )

    print("\nAPI CONNECTION SUCCESS\n")

    print(response)

except Exception as e:

    print("\nERROR:\n")
    print(e)
