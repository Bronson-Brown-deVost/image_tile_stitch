import re
import os
import ntpath
from tqdm import tqdm
from colorama import Fore, Style

name_parser = re.compile(r'(?P<plate>.*)-(?P<frag>Fg\d+)-(?P<side>[A-z]+).*-(?P<type>.*)\.')
col_parser = re.compile(r'.*-C(?P<col>\d+)-.*')
row_parser = re.compile(r'.*-R(?P<row>\d+)-.*')

def parse_image_files(dir: str):
    image_collection = {}
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                file_list.append(os.path.join(root, file))

    for file in tqdm(file_list, desc=f"{Fore.CYAN}Parsing and Gathering Image Files in {dir}{Style.RESET_ALL}", leave=False, position=1):
        matches = name_parser.match(file)
        if matches is not None:
            plate, frag, side, type = matches.group('plate'), matches.group('frag'), matches.group('side'), matches.group('type')
            if plate is not None and frag is not None and type is not None and side is not None:
                plate = ntpath.basename(plate)
                label = f'{plate}-{frag}-{side}'
                if label not in image_collection:
                    image_collection[label] = {}
                if type not in image_collection[label]:
                    image_collection[label][type] = {'rows': {}}
                col_matches = col_parser.match(file)
                row_matches = row_parser.match(file)
                if col_matches is not None and row_matches is not None:
                    col = col_matches.group('col')
                    row = row_matches.group('row')
                    if row not in image_collection[label][type]['rows']:
                        image_collection[label][type]['rows'][row] = []
                    image_collection[label][type]['rows'][row].append({'file': file, 'col': col, 'row': row})
                    # This is a bit lazy, but the lists are not long, so it probably doesn't matter
                    if len(image_collection[label][type]['rows'][row]) > 1:
                        image_collection[label][type]['rows'][row] = sorted(image_collection[label][type]['rows'][row], key=lambda d: int(d['col']))
                    if len(image_collection[label][type]['rows'].keys()) > 1:
                        image_collection[label][type]['rows'] = {key: image_collection[label][type]['rows'][key] for key in sorted(image_collection[label][type]['rows'].keys())}

    return image_collection


if __name__ == '__main__':
    catalog = parse_image_files('./images')
    print(catalog)
