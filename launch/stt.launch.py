from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='k9_stt_pkg',
            executable='k9_stt',
            name='k9_stt',
            output='screen',
            parameters=[{
                'audio_topic': '/audio/raw',
                'effective_state_topic': '/audio/effective_state',
                'text_topic': '/speech_to_text/text',
                'state_topic': '/speech_to_text/state',

                'sample_rate': 16000,
                'silence_timeout': 1.0,
                'pre_roll_ms': 300,
                'min_speech_ms': 160,
                'max_speech_s': 30.0,

                'model_path':
                    '/home/hopkira/whisper.cpp/models/ggml-large-v3-turbo.bin',

                'vad_model_path':
                    '/home/hopkira/whisper.cpp/models/ggml-silero-v6.2.0.bin',

                'use_gpu': True,
                'gpu_device': 0,
                'flash_attn': False,
                'whisper_threads': 4,
                'beam_size': 5,
                'language': 'en',
                'vad_threads': 2,

                'publish_stale_results': False,
            }]
        )
    ])
