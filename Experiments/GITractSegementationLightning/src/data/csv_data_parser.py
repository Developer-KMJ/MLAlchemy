# Standard library imports
import csv
import os

# Related third party imports
from numpy import number

class RecordData:

    label: str
    full_path: str
    case: str
    day: str
    slice: str
    image_width: int
    image_height: int
    pixel_width: number
    pixel_height: number
    organ_list: dict[str, str]

    def __init__(self, 
                 label: str = '', 
                 full_path: str = '', 
                 case: str = '', 
                 day: str = '', 
                 slice: str = '', 
                 image_width: int = 0, 
                 image_height: int = 0, 
                 pixel_width: number = 0,
                 pixel_height: number = 0, 
                 organ_list: dict[str, str] = None):
        
        self.label = label
        self.full_path = full_path
        self.case = case
        self.day = day
        self.slice = slice
        self.image_width = image_width
        self.image_height = image_height
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.organ_list = organ_list if organ_list is not None else {}
    
    @property
    def has_organs(self) -> bool:
        for value in self.organ_lists():
            if len(value) > 0:
                return True
        return False

def parse_csv_into_record_data(data_directory: str, csv_file: str) -> [RecordData]:
    """
    Create a lookup table mapping labels with case information, including file,
    segmentation data, and pixel sizes.
    """
    records = {}

    for root, _, files in os.walk(data_directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_record = __parse_filename(full_path)
            records[file_record.label] = file_record

    __update_records_inline_with_organ_data(csv_file, records)

    return records
         
def __parse_filename(full_path: str) -> RecordData:
    """
    Parse the label into its component parts
    """
    try:
        record_data = RecordData()

        record_data.case, next_start = __parse_number_after_word(full_path, '/case', '/', 0)
        if record_data.case is None:
            print(f'Error parsing case number in {full_path}')
            return None

        record_data.day, next_start = __parse_number_after_word(full_path, '_day', '/', next_start)
        if record_data.day is None:
            print(f'Error parsing day number in {full_path}')
            return None

        record_data.slice, next_start = __parse_number_after_word(full_path, '/slice_', '_', next_start)
        if record_data.slice is None:
            print(f'Error parsing slice number in {full_path}')
            return None

        filename = full_path[next_start:]
        filename = filename.replace('.png', '')
        filename_parts = filename.split('_')
        if len(filename_parts) >= 4:
            try:
                record_data.image_width = int(filename_parts[1])
                record_data.image_height = int(filename_parts[2])
                record_data.pixel_width = float(filename_parts[3])
                record_data.pixel_height = float(filename_parts[4])
            except ValueError:
                print(f'Unable to parse 4 parts of the filename')
                return None

        record_data.scan_file = full_path

        slice_number_padded = str(record_data.slice).zfill(4)
        record_data.label = f'case{record_data.case}_day{record_data.day}_slice_{slice_number_padded}'

        return record_data
    
    except:
        print(f'Error parsing {full_path}')
        return None

def __parse_number_after_word(target_str: str, start_word: str, end_word: str, start: int = 0) -> [int, int]:
    
    number_result = None
    end_position = -1

    if not start_word or not end_word:
        print('no start or end words to parse')
        return number_result, end_position

    try:
        # Parse out the case number.
        word_start = target_str.find(start_word, start) 
        if (word_start != -1):
            word_start += len(start_word)
            word_end = target_str.find(end_word, word_start)
            if word_start != -1 and word_end > word_start:
                end_position = word_end
                number_result = int(target_str[word_start:word_end])
    except ValueError:
        print(f'Invalid number: {target_str[word_start:word_end]}')
        number_result = None
    
    return number_result, end_position
   

def __update_records_inline_with_organ_data(csv_path: str, data_records: dict[str, RecordData]) -> None:
    """
    Read the data records from the CSV file
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist.")
        return None

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # skip header row

        for row in csv_reader:
            label = str(row[0])

            working_record = data_records.get(label)
            if (working_record is None):
                print(f'Error finding label {label} in lookup table - this means the filesystem and csv are out of sync skipping')
                continue;
            
            organ = str(row[1])

            # Split the segmentations where ever there is a space
            # into offset and length pairs 
            rle = str(row[2])
            segmentation_pairs = []
            line_tokens = rle.split(None)
            for i in range(0, len(line_tokens), 2):
                offset = int(line_tokens[i])
                length = int(line_tokens[i+1])
                segmentation_pairs.append((offset, length))
        
            working_record.organ_list[organ] = segmentation_pairs

    return data_records

def generate_train_csv_file(data: [RecordData], output_path: str):
    """
    Generate the csv file for training
    """
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id','class','segmentation'])
        for row in sorted(data.keys()):
            row = data[row]
            for organ in sorted(row.organ_list.keys()):
                rle_pairs_str = ''
                for pair in row.organ_list[organ]:
                    rle_pairs_str += f'{pair[0]} {pair[1]} '
                writer.writerow([row.label, organ, rle_pairs_str])
    
    return    