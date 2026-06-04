import os
import sys
import time
import json
import googleapiclient.discovery

# 1. Pull secure secrets passed down from GitHub Environments
VIDEO_ID = os.getenv("YOUTUBE_VIDEO_ID")
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not VIDEO_ID or not API_KEY:
    print("Error: Missing configurations. Ensure YOUTUBE_VIDEO_ID and YOUTUBE_API_KEY are set.")
    sys.exit(1)

# Initialize the Google API client engine
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

# 2. Convert standard Live Stream Video ID into its internal Live Chat Room ID
try:
    video_response = youtube.videos().list(part="liveStreamingDetails", id=VIDEO_ID).execute()
    items = video_response.get("items", [])
    if not items:
        print(f"Error: Video ID {VIDEO_ID} not found.")
        sys.exit(1)
    
    live_details = items[0].get("liveStreamingDetails", {})
    live_chat_id = live_details.get("activeLiveChatId")
    
    if not live_chat_id:
        print("Error: This video ID is not actively live streaming right now.")
        sys.exit(1)
    print(f"Pipeline Connected! Live Chat Room ID: {live_chat_id}")
except Exception as e:
    print(f"Failed to fetch live broadcast details: {e}")
    sys.exit(1)

# Target configurations matching your repository file name
DATASET_FILE = "final_unified_dataset.json"
next_page_token = None

# Safety timer parameters (Cap runtime around 2.5 hours so nothing runs forever on GitHub)
start_time = time.time()
max_duration = 2.5 * 60 * 60  

print("Now actively listening to live stream messages...")

# 3. Continuous Data Catching Loop
while time.time() - start_time < max_duration:
    try:
        response = youtube.liveChatMessages().list(
            liveChatId=live_chat_id,
            part="snippet,authorDetails",
            pageToken=next_page_token
        ).execute()
        
        items = response.get("items", [])
        next_page_token = response.get("nextPageToken")
        
        if items:
            # Load your existing file data safely
            if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 0:
                with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
            else:
                dataset = []

            # Create a quick set of your text strings to filter out spammed text instantly
            existing_texts = {msg['text'] for msg in dataset if 'text' in msg}
            
            new_added_count = 0
            for item in items:
                snippet = item.get("snippet", {})
                author = item.get("authorDetails", {})
                text_content = snippet.get("displayMessage")
                
                # Only save if the comment text isn't already a duplicate
                if text_content and text_content not in existing_texts:
                    comment_map = {
                        "comment_id": item.get("id"),
                        "source": "youtube_live",
                        "event": "world_cup_live_event",
                        "video_id": VIDEO_ID,
                        "author": author.get("displayName", "Anonymous User"),
                        "created_at": snippet.get("publishedAt"),
                        "likes": 0,
                        "reply_count": 0,
                        "text": text_content
                    }
                    dataset.append(comment_map)
                    existing_texts.add(text_content)
                    new_added_count += 1
            
            # Commit the updates back onto the file right away
            if new_added_count > 0:
                with open(DATASET_FILE, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=4, ensure_ascii=False)
                print(f"Successfully harvested {new_added_count} brand-new comments.")
            else:
                print("Polled batch parsed: No unique text strings detected.")

        # Crucial: Sleep exactly the amount of milliseconds YouTube asks us to wait
        polling_interval = response.get("pollingIntervalMillis", 4000) / 1000.0
        time.sleep(polling_interval)

    except Exception as e:
        print(f"Broadcast closed or connection timed out: {e}")
        break

print("Catching lifecycle complete.")
