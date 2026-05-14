import os
import json
import re

from datetime import datetime

from dotenv import load_dotenv
from googleapiclient.discovery import build

# ============================================
# LOAD ENV VARIABLES
# ============================================

load_dotenv()

api_key = os.getenv("YOUTUBE_API_KEY")

# ============================================
# CREATE YOUTUBE CLIENT
# ============================================

youtube = build(
    "youtube",
    "v3",
    developerKey=api_key
)

# ============================================
# VIDEO CONFIGURATION
# ============================================

video_id = "g75avihz7m0"

event_name = "youtube_match_event"

max_comments = 1000

# ============================================
# TIMESTAMP
# ============================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================
# FILTER FUNCTION
# ============================================

def is_valid_comment(text):

    text = text.strip()

    if len(text) < 10:
        return False

    if re.fullmatch(r'[^\w\s]+', text):
        return False

    return True

# ============================================
# START COLLECTION
# ============================================

print("\n===================================")
print("STARTING YOUTUBE COLLECTION")
print("===================================")

data = []

seen_comment_ids = set()

next_page_token = None

# ============================================
# PAGINATION LOOP
# ============================================

while len(data) < max_comments:

    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        pageToken=next_page_token,
        textFormat="plainText"
    ).execute()

    for item in response["items"]:

        # ============================================
        # UNIQUE COMMENT ID
        # ============================================

        comment_id = item["id"]

        # Skip duplicates
        if comment_id in seen_comment_ids:
            continue

        seen_comment_ids.add(comment_id)

        # ============================================
        # COMMENT DATA
        # ============================================

        comment = item["snippet"]["topLevelComment"]["snippet"]

        text = comment["textDisplay"].strip()

        # ============================================
        # FILTER BAD COMMENTS
        # ============================================

        if not is_valid_comment(text):
            continue

        created_at = comment["publishedAt"]

        author = comment["authorDisplayName"]

        likes = comment["likeCount"]

        reply_count = item["snippet"]["totalReplyCount"]

        # ============================================
        # SHOW COMMENTS
        # ============================================

        print("\n----------------------------")
        print(f"AUTHOR: {author}")
        print(f"DATE: {created_at}")
        print(f"LIKES: {likes}")
        print(f"COMMENT:\n{text}")

        # ============================================
        # SAVE DATA
        # ============================================

        data.append({
            "comment_id": comment_id,
            "source": "youtube",
            "event": event_name,
            "video_id": video_id,
            "author": author,
            "created_at": created_at,
            "likes": likes,
            "reply_count": reply_count,
            "text": text
        })

        if len(data) >= max_comments:
            break

    # ============================================
    # NEXT PAGE
    # ============================================

    next_page_token = response.get("nextPageToken")

    if not next_page_token:
        break

# ============================================
# CREATE DATASETS DIRECTORY
# ============================================

os.makedirs("datasets", exist_ok=True)

# ============================================
# JSON FILE PATH
# ============================================

json_path = f"datasets/{event_name}_{timestamp}.json"

# ============================================
# EXPORT JSON
# ============================================

with open(json_path, "w", encoding="utf-8") as json_file:

    json.dump(
        data,
        json_file,
        ensure_ascii=False,
        indent=4
    )

# ============================================
# FINAL RESULTS
# ============================================

print("\n===================================")
print("DATASET CREATED SUCCESSFULLY")
print("===================================")

print(f"\nTotal unique comments collected: {len(data)}")

print(f"\nJSON saved at:")
print(json_path)
