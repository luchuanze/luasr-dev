
import json
import argparse
import torchaudio


def main():
    parser = argparse.ArgumentParser(description='compute duration from data list file')
    parser.add_argument('--train_data_in', required=True, help='train data file')
    parser.add_argument('--train_data_out', required=True, help='train data file')

    args = parser.parse_args()
    data_list = []
    with open(args.train_data_in, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for json_line in lines:
            obj = json.loads(json_line)
            assert 'key' in obj
            assert 'wav' in obj
            assert 'txt' in obj
            key = obj['key']
            wav_file = obj['wav']
            txt = obj['txt']

            wavform, sample_rate = torchaudio.load(wav_file)
            obj['dur'] = wavform.size(1)
            jt = json.dumps(obj, ensure_ascii=False)
            data_list.append(jt)

        with open(args.train_data_out, 'w', encoding='utf-8') as fout:
            fout.writelines('\n'.join(data_list))


if __name__ == '__main__':
    main()