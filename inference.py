import subprocess
import nemo.collections.asr as nemo_asr
from webvtt import WebVTT, Caption
from datetime import timedelta
import torch
import click


def format_timestamp(timestamp):
    '''Get float output from parakeet into pretty-printed form for WebVTT'''
    time_object = timedelta(seconds=timestamp)
    if time_object.microseconds > 0:
        formatted = str(time_object)[:-3]
    else:
        formatted = str(time_object) + '.000'
    return formatted


def generate_vtt(output, path):
    '''Generates a VTT file at the designated path from timestamped parakeet
    transcription.
    '''
    vtt = WebVTT()
    for caption in output[0].timestamp['segment']:
        start = format_timestamp(caption['start'])
        end = format_timestamp(caption['end'])
        text_line = caption['segment']
        caption = Caption(start, end, text_line)
        vtt.captions.append(caption)

    with open(path, 'w', encoding='utf-8') as f:
        vtt.write(f)


def convert_file(file_base):
    '''Convert file from mp4 to wav file required by parakeet'''
    command = f'ffmpeg -i {file_base}.mp4 -ar 16000 -ac 1 {file_base}.wav'
    subprocess.call(command, shell=True)


def transcribe_file(file_base, low_attention):
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')

    if low_attention:# limit attention to reduce memory used
        asr_model.change_attention_model("rel_pos_local_attn", [256, 256])
        asr_model.change_subsampling_conv_chunking_factor(1)
        asr_model.to(torch.bfloat16)

    output = asr_model.transcribe([f'{file_base}.wav'], timestamps=True)

    return output


@click.command()
@click.option('--low-attention', is_flag=True, default=False)
@click.argument('file_base')
def transcribe(file_base, low_attention):
    convert_file(file_base)
    output = transcribe_file(file_base, low_attention=low_attention)
    generate_vtt(output, f'{file_base}.vtt')


if __name__ == '__main__':
    transcribe()