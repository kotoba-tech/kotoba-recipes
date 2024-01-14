import argparse
import json


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='pubmed dataset builder')
    parser.add_argument('--data-path', type=str, default=None, help='data directory')
    parser.add_argument('--output-path', type=str, default=None, help='output directory')

    args = parser.parse_args()
    return args


def sentence_splitter(text: str) -> tuple[str, str, str, str, str]:
    splitted_texts: list[str] = text.split("|")

    if len(splitted_texts) < 4:
        print(f"dataset must contain (id, english_text, japanese_text). but this line doesn't. invalid text: {text}")
        raise ValueError

    id: str = splitted_texts[0]  # document id (ex: 22)
    english_text: str = ""
    japanese_text: str = ""

    for index in range(len(splitted_texts) // 2 - 1):
        en_text, ja_text = splitted_texts[index * 2 + 1], splitted_texts[(index + 1) * 2]
        if en_text == "" and ja_text == "":
            break
        english_text += en_text
        japanese_text += ja_text

    journal_name: str = splitted_texts[-3]
    data_index: str = splitted_texts[-1]

    return id, english_text, japanese_text, journal_name, data_index


def main() -> None:
    args = arg_parse()

    data_path: str = args.data_path
    output_path: str = args.output_path

    data_count: int = 0
    with open(data_path, 'r') as file:
        with open(output_path, 'w') as write_file:
            for line in file:
                try:
                    id, en_text, ja_text, data_source, index = sentence_splitter(line)
                except Exception:
                    continue

                json_object: dict[str, str] = {
                    "id": id,
                    "english": en_text,
                    "japanese": ja_text,
                    "source": data_source,
                    "index": index,
                }
                write_file.write(json.dumps(json_object, ensure_ascii=False) + "\n")
                data_count += 1
                if data_count % 1000 == 0:
                    print(f"LOG: processing {data_count}")

    print(f"LOG: {data_count} total")


if __name__ == '__main__':
    main()
