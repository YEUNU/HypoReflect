import re
import sys

def test_parsing(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"File: {filename}")
    print(f"Total Content Length: {len(content)}")
    
    # 1. re.split (Old method)
    page_pattern_split = r'--- Page (\d+) ---'
    parts = re.split(page_pattern_split, content)
    print(f"[re.split] Parts count: {len(parts)}")
    
    # 2. re.finditer (New method)
    page_pattern = re.compile(r'--- Page (\d+) ---')
    matches = list(page_pattern.finditer(content))
    print(f"[re.finditer] Matches found: {len(matches)}")
    
    if matches:
        print(f"First match: {matches[0].group(0)} at {matches[0].start()}")
        print(f"Last match: {matches[-1].group(0)} at {matches[-1].start()}")
        
        pages = []
        for i in range(len(matches)):
            start_match = matches[i]
            page_num = int(start_match.group(1))
            content_start = start_match.end()
            if i < len(matches) - 1:
                content_end = matches[i+1].start()
            else:
                content_end = len(content)
            page_text = content[content_start:content_end].strip()
            pages.append({"num": page_num, "len": len(page_text)})
        
        print(f"Parsed Pages: {len(pages)}")
        print(f"Page 1 length: {pages[0]['len']}")
        if len(pages) > 1:
            print(f"Page 2 length: {pages[1]['len']}")

if __name__ == "__main__":
    test_parsing("data/finance_corpus_sample_ocr/text/3M_2018_10K.txt")
