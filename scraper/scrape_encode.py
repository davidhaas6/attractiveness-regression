# Pulls images from Reddit and encodes their facial structures 

# Retrieving content
from io import BytesIO
import praw, psaw
from PIL import Image
import requests
import pickle

# Processing content
import face_recognition
import numpy as np

# Overhead
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import os
import traceback


# Connect to Reddit and Pushshift
reddit_client = praw.Reddit(client_id='6ZOjAwnqUehb5Q', 
                            client_secret='gc4rkA50yNq9pBn1diU11Xj1nKY', 
                            user_agent='ffinder_test')
api = psaw.PushshiftAPI(reddit_client)
deleted_reddit, deleted_imgur = pickle.load(open("scraper/deleted_binaries.pkl",'rb'))
verbose = False

# Fix dlib multiprocessing issues with MacOS https://github.com/davisking/dlib/issues/1555
ctx = multiprocessing
if "forkserver" in multiprocessing.get_all_start_methods():
    ctx = multiprocessing.get_context("forkserver")

def log(logstr):
    if verbose:  print(logstr)

def is_img(path):
    return path[-4:] in {".png", '.jpg'}

def notify(message):
    """ Notifies a Pushover device (phone) of a message
    """
    import http.client, urllib
    try:
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
        urllib.parse.urlencode({
            "token": "ab9g9edu1pcsms9mqpegw8bmtu4hbd",
            "user": "ue7i7x5qpqyjku7q1n1av9ewwi7ieo",
            "message": message,
        }), { "Content-type": "application/x-www-form-urlencoded" })
        conn.getresponse()
    except Exception as e:
        print(e)


def process_image(api_result):
    """ Downloads image from the Pushshift api entry and encodes it if there are faces present

        Arguments:
            api_result (psaw api result): A single entry in the psaw search query result
        Returns:
            False if unsuccessful
            tuple of encoding and metadata if successfull
    """
    if not is_img(api_result.url):  return False
    # Fetch and load image
    try:
        img_bytes = requests.get(api_result.url).content
        if img_bytes == deleted_imgur or img_bytes == deleted_reddit:  return False
        img = face_recognition.load_image_file(BytesIO(img_bytes))
    except:
        return False

    # Search for faces and encode
    face_locs = face_recognition.face_locations(img)
    if len(face_locs) > 0:
        encodings = face_recognition.face_encodings(img, known_face_locations=face_locs)
        return encodings, (api_result.shortlink, api_result.url, api_result.score)
    else:
        return False


def generate_encodings(subreddit, face_limit=5e5):
    """ Generates a set of facial encodings from each valid image on a subreddit

        Given a subreddit, this function collects its images, starting with the most recent in multi-day chunks.
        After downloading an image, it is analyzed for faces, and if faces are present it generates an encoding
        and stores metadata about the image to be pickled and saved.
        
        Args:
            subreddit (str): The subreddit name to process images from
            image_limit (int): The maximum number of images to process.

        Returns:
            tuple: (A Nx128 matrix of encodings, a list of metadata corresponding by index to each row in the matrix)
    """

    encodings = np.zeros((0,128))
    metadata = []
    
    CHUNK_SIZE = 1 * 86400  # 1 day chunks
    chunk_end = datetime.now().timestamp()
    chunk_start = chunk_end - CHUNK_SIZE
    stop = False

    while not stop:
        prev_length = encodings.shape[0]
        face_data = []
        # Process the chunk of data
        try:
            chunk_data = api.search_submissions(before=int(chunk_end), after=int(chunk_start), subreddit=subreddit)

            log("\nProcessing images...")
            with ctx.Pool() as pool:
                face_data = list(tqdm(pool.imap_unordered(process_image, chunk_data), unit=" Images"))

            # Separate and store face_data into encodings and metadata
            for entry in face_data:
                if type(entry) is not tuple:  continue
                enc, meta = entry
                encodings = np.vstack((encodings,enc))

                num_faces = len(enc)
                for i in range(num_faces):
                    metadata.append(meta)

            # Print results
            length = encodings.shape[0]
            num_added = length - prev_length
            face_per_img = 0 if len(face_data) == 0 else num_added/len(face_data)
            
            stop = length >= face_limit or length == prev_length
            log("%i new faces encoded (avg %.2f faces per image)! Total = %i" % (num_added, face_per_img, length))
        
        except Exception as e:
            if e is KeyboardInterrupt:
                break
            print("ERROR: ")
            traceback.print_exc()

        chunk_end = chunk_start-1
        chunk_start = chunk_end - CHUNK_SIZE
        
    return encodings, metadata

def main(subreddits, out_dir='./encodings/'):
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    for subreddit in subreddits:
        encodings, metadata = generate_encodings(subreddit)

        # Save encodings
        name = out_dir + subreddit + "_" + datetime.now().strftime('%H%M-%m%d%y') + ".pkl"
        pickle.dump((encodings,metadata), open(name, "wb"))
        log("Saved encodings to %s" % name)
        notify("Saved %i encodings for r/%s" % (len(metadata), subreddit))

    notify("RUN COMPLETED")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download and encode faces from a subreddit")
    parser.add_argument("subreddits", nargs="*", help="The subreddits to process")
    parser.add_argument("-v", '--verbose', action="store_true", default=False, help="The subreddits to process")
    args = parser.parse_args()

    verbose = args.verbose
    main(args.subreddits)