"""Download the AMI dataset and prepare it for benchmarking.

More information here: https://groups.inf.ed.ac.uk/ami/corpus/

This dataset is used for Speaker Diarization.

Usage: TODO

"""

from pathlib import Path

from rtasr.utils import download_file, resolve_cache_dir

# from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

rttm_url = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/only_words/rttms/{}/{}.rttm"
uem_url = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/uems/{}/{}.uem"
list_url = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/lists/{}.meetings.txt"


audio_types = ['Mix-Headset', 'Array1-01']

# these two IDs in the train set are missing download links for Array1-01.
# We exclude them as a result.
not_found_ids = ['IS1007d', 'IS1003b']

# Inspired from:
# https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/speaker_tasks/get_ami_data.py
def download_ami_dataset(output_dir: str = None) -> None:
    """
    Download the AMI dataset and prepare it for benchmarking.

    Args:
        output_dir (str):
            Path to the directory where the dataset is stored.
            If not specified, the dataset will be stored in the
            default `~/.cache/rtasr/datasets` directory.
    """
    if output_dir is None:
        cache_datasets_dir = resolve_cache_dir() / "datasets"
        cache_datasets_dir.mkdir(parents=True, exist_ok=True)
        output_dir = cache_datasets_dir / "ami"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("test", "dev", "train"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        filepath = download_file(list_url.format(split), split_dir)
        with open(filepath) as f:
            ids = f.read().strip().split('\n')
        # for id in [file_id for file_id in ids if file_id not in not_found_ids]:
        #     for audio_type in audio_types:
        #         audio_type_path = os.path.join(audio_path, audio_type)
        #         os.makedirs(audio_type_path, exist_ok=True)
        #         os.system(
        #             f"wget -P {audio_type_path} https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{id}/audio/{id}.{audio_type}.wav"
        #         )
        #     rttm_download = rttm_url.format(split, id)
        #     os.system(f"wget -P {rttm_path} {rttm_download}")
        #     uem_download = uem_url.format(split, id)
        #     os.system(f"wget -P {uem_path} {uem_download}")

download_ami_dataset()
        

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Download the AMI Corpus Dataset for Speaker Diarization")
#     parser.add_argument(
#         "--test_manifest_filepath",
#         help="path to output test manifest file",
#         type=str,
#         default='AMI_test_manifest.json',
#     )
#     parser.add_argument(
#         "--dev_manifest_filepath", help="path to output dev manifest file", type=str, default='AMI_dev_manifest.json',
#     )
#     parser.add_argument(
#         "--train_manifest_filepath",
#         help="path to output train manifest file",
#         type=str,
#         default='AMI_train_manifest.json',
#     )
#     parser.add_argument("--data_root", help="path to output data directory", type=str, default="ami_dataset")
#     args = parser.parse_args()

#     data_path = os.path.abspath(args.data_root)
#     os.makedirs(data_path, exist_ok=True)

#     for manifest_path, split in (
#         (args.test_manifest_filepath, 'test'),
#         (args.dev_manifest_filepath, 'dev'),
#         (args.train_manifest_filepath, 'train'),
#     ):
#         split_path = os.path.join(data_path, split)
#         audio_path = os.path.join(split_path, "audio")
#         os.makedirs(split_path, exist_ok=True)
#         rttm_path = os.path.join(split_path, "rttm")
#         uem_path = os.path.join(split_path, "uem")

#         os.system(f"wget -P {split_path} {list_url.format(split)}")
#         with open(os.path.join(split_path, f"{split}.meetings.txt")) as f:
#             ids = f.read().strip().split('\n')
#         for id in [file_id for file_id in ids if file_id not in not_found_ids]:
#             for audio_type in audio_types:
#                 audio_type_path = os.path.join(audio_path, audio_type)
#                 os.makedirs(audio_type_path, exist_ok=True)
#                 os.system(
#                     f"wget -P {audio_type_path} https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{id}/audio/{id}.{audio_type}.wav"
#                 )
#             rttm_download = rttm_url.format(split, id)
#             os.system(f"wget -P {rttm_path} {rttm_download}")
#             uem_download = uem_url.format(split, id)
#             os.system(f"wget -P {uem_path} {uem_download}")

#         rttm_files_path = os.path.join(split_path, 'rttm_files.txt')
#         with open(rttm_files_path, 'w') as f:
#             f.write('\n'.join(os.path.join(rttm_path, p) for p in os.listdir(rttm_path)))
#         uem_files_path = os.path.join(split_path, 'uem_files.txt')
#         with open(uem_files_path, 'w') as f:
#             f.write('\n'.join(os.path.join(uem_path, p) for p in os.listdir(uem_path)))
#         for audio_type in audio_types:
#             audio_type_path = os.path.join(audio_path, audio_type)
#             audio_files_path = os.path.join(split_path, f'audio_files_{audio_type}.txt')
#             with open(audio_files_path, 'w') as f:
#                 f.write('\n'.join(os.path.join(audio_type_path, p) for p in os.listdir(audio_type_path)))
#             audio_type_manifest_path = manifest_path.replace('.json', f'.{audio_type}.json')
#             create_manifest(
#                 audio_files_path, audio_type_manifest_path, rttm_path=rttm_files_path, uem_path=uem_files_path
#             )