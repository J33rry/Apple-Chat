import pandas as pd
import json
import random
from collections import defaultdict
from tqdm import tqdm

def extract_apple_support_threads(csv_path, output_golden, output_knowledge, num_golden=300):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Convert IDs to string, handling NaNs
    # Some IDs might be floats if there are NaNs
    df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].fillna(-1).astype(pd.Int64Dtype()).astype(str)
    df['tweet_id'] = df['tweet_id'].astype(str)
    
    # Ensure inbound is boolean (may be string "True"/"False" in some pandas versions)
    if df['inbound'].dtype == object:
        df['inbound'] = df['inbound'].map({'True': True, 'False': False, True: True, False: False})
    df['inbound'] = df['inbound'].astype(bool)
    
    tweets = {}
    for row in tqdm(df.itertuples(), total=len(df), desc="Indexing tweets"):
        tweets[row.tweet_id] = {
            'tweet_id': row.tweet_id,
            'author_id': str(row.author_id),
            'inbound': bool(row.inbound),
            'text': str(row.text),
            'in_response_to_tweet_id': row.in_response_to_tweet_id
        }
        
    print("Finding AppleSupport conversations...")
    apple_tweet_ids = [t['tweet_id'] for t in tweets.values() if t['author_id'] == 'AppleSupport']
    
    roots = set()
    for tid in tqdm(apple_tweet_ids, desc="Finding roots"):
        current = tid
        # limit loop to prevent infinite cycle just in case
        loop_count = 0
        while loop_count < 100:
            parent = tweets[current]['in_response_to_tweet_id']
            if parent == '-1' or parent == '<NA>' or parent not in tweets:
                roots.add(current)
                break
            current = parent
            loop_count += 1
            
    children = defaultdict(list)
    for t in tweets.values():
        parent = t['in_response_to_tweet_id']
        if parent != '-1' and parent != '<NA>':
            children[parent].append(t['tweet_id'])
            
    threads = []
    
    # Iterative DFS to avoid RecursionError
    for root in tqdm(roots, desc="Building threads"):
        stack = [([tweets[root]], root)]
        while stack:
            current_thread, node = stack.pop()
            if not children[node]:
                threads.append(current_thread)
            else:
                for child in children[node]:
                    stack.append((current_thread + [tweets[child]], child))
        
    valid_threads = []
    seen_roots = set()
    for th in threads:
        authors = set(t['author_id'] for t in th)
        root_id = th[0]['tweet_id']
        # Deduplicate: only keep the first (longest) thread per root tweet
        if 'AppleSupport' in authors and len(authors) > 1 and root_id not in seen_roots:
            valid_threads.append(th)
            seen_roots.add(root_id)
            
    print(f"Found {len(valid_threads)} valid AppleSupport threads (deduplicated).")
    
    # Thread length statistics
    lengths = [len(th) for th in valid_threads]
    if lengths:
        print(f"Thread length stats: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.1f}, median={sorted(lengths)[len(lengths)//2]}")
    
    random.seed(42)
    random.shuffle(valid_threads)
    
    golden = valid_threads[:num_golden]
    knowledge = valid_threads[num_golden:]
    
    with open(output_golden, 'w', encoding='utf-8') as f:
        json.dump(golden, f, indent=2)
        
    with open(output_knowledge, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, indent=2)
        
    print(f"Saved {len(golden)} threads to {output_golden}")
    print(f"Saved {len(knowledge)} threads to {output_knowledge}")

if __name__ == "__main__":
    csv_path = "data/twcs.csv"
    output_golden = "data/golden_dataset.json"
    output_knowledge = "data/knowledge_base.json"
    extract_apple_support_threads(csv_path, output_golden, output_knowledge, 300)
