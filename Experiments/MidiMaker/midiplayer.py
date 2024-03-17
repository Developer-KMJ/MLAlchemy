import numpy as np
import os
import regex as re
from timidity import Parser, play_notes

cwd = os.path.dirname(__file__)


def load_training_data(filename):
    with open(os.path.join(cwd, "data", filename), "r") as f:
        text = f.read()
    result_songs = extract_song_snippet(text)
    return result_songs


def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    result_songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(result_songs)))
    return result_songs


def save_song_to_midi(song, filename):
    basename = save_song_to_abc(song, filename)
    if os.path.exists(basename + '.midi'):
        os.remove(basename + '.midi')

    abc2midi(basename + '.abc')
    return os.path.exists(basename + '.midi')


def play_song(song, filename):
    if save_song_to_midi(song, filename):
        return play_midi(filename + '.midi')

    print("Unable to generate midi")
    return None


def save_song_to_abc(song, filename="tmp"):
    save_name = os.path.join(cwd, f"{filename}.abc")
    with open(save_name, "w") as f:
        f.write(song)
    return filename


def abc2midi(abc_file):
    path_to_tool = os.path.join(cwd, 'abcMidi', 'abc2midi.exe')
    midi_file = abc_file.replace(".abc", ".midi")
    cmd = f"\"{path_to_tool}\" {abc_file} -o {midi_file}"
    return os.system(cmd)


def play_midi(midi_file):
    ps = Parser(midi_file)
    play_notes(*ps.parse(), np.sin)
