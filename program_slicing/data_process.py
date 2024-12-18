import re
import pickle


def process_slices(slice):
    data = ' '.join(slice.split("\n")[2:-2])
    label = slice.split()[-1]
    return data, label


def open_slices_txt(filename):
    with open(filename, 'r') as file:
        content = file.read()
        segments = content.split('------------------------------')

        segments = list(filter(lambda x: len(x) >= 2, segments))

        location = list()

        for segment in segments:
            line = segment.split("\n")
            line = list(filter(None, line))

            words = line[0].split(" ")
            location.append((words[1], words[2], words[3]))

        processed_segments = [process_slices(segment)[0] for segment in segments if segment.strip()]
        label = [process_slices(segment)[1] for segment in segments if segment.strip()]

        return processed_segments, label, location


def write_slices_txt(save_tokens_path, save_labels_path, save_location_path, all_slices_of_token, label, location):
    with open(save_tokens_path, 'wb') as file:
        pickle.dump(all_slices_of_token, file)
    with open(save_labels_path, 'wb') as file:
        pickle.dump(label, file)
    with open(save_location_path, 'wb') as file:
        pickle.dump(location, file)


def main():
    train_slices_path = 'data_new/integeroverflow_slices.txt'
    save_tokens_path = "data_new/tokens/train_tokens.pkl"
    save_labels_path = "data_new/tokens/train_labels.pkl"
    save_location_path = "data_new/tokens/train_location.pkl"
    all_slices_of_token, label, location = open_slices_txt(train_slices_path)
    write_slices_txt(save_tokens_path, save_labels_path, save_location_path, all_slices_of_token, label, location)


if __name__ == "__main__":
    main()
