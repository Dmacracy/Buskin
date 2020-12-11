import sys
from entities import *
from commands import *


'''
Function to read metadata json
'''
def get_metadata(metaDataFile):
    with open(metaDataFile,'r') as fob:
        metaBooks = json.load(fob)
    return metaBooks

'''
Function for running the parsing pipeline on an entire dataset.
Inputs:
    DataPath - path to dir with the book txt files
    metaDataFile - json filename with metaData for the dataset
    outDir - directory where outputs will be written    
    [verbose] - bool indicating whether to print progress and timing
    [n] - int determines how many books to parse
Outputs:
    no function return value, but resulting json files are written to disk in the outDirs.
'''
def run_pipe(dataPath, metaDataFile, outDir, verbose=True, n=1):
    
    metaBooks = get_metadata(metaDataFile)
    
    if not os.path.exists(outDir):
        os.makedirs(outDir, exist_ok=True)
    
    # Iterate over books and run pipeline on each
    parsed = 0
    for bookEntry in metaBooks:
        if parsed < n:
            fileName = bookEntry['filename']
            title = os.path.splitext(fileName)[0]
            outFileName = os.path.join(outDir, f'{title}_parsed.json')
            # Skip already processed books
            if not os.path.exists(outFileName):
                p, book = parse_book(os.path.join(dataPath, fileName), verbose=verbose)
                p.close()
                if book != None:
                    with open (outFileName, 'w') as outFile:
                        json.dump(book.to_json(), outFile, separators=(',', ':'), indent=4)
                    parsed += 1
        else:
            break

        
if __name__ == "__main__":
    dataPath = sys.argv[1] #"../datasets/Gutenberg/txt/"
    metaDataFile = sys.argv[2] #"results/metadata-final.json"
    outDir = sys.argv[3] #"results/book_parses"
    n = int(sys.argv[4])

    run_pipe(dataPath, metaDataFile, outDir, verbose=True, n=n)
