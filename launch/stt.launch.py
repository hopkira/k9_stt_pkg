from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    vad_start_threshold = LaunchConfiguration('vad_start_threshold')
    vad_end_threshold = LaunchConfiguration('vad_end_threshold')
    silence_timeout = LaunchConfiguration('silence_timeout')
    vad_diagnostics = LaunchConfiguration('vad_diagnostics')
    vad_log_interval_ms = LaunchConfiguration('vad_log_interval_ms')

    return LaunchDescription([
        DeclareLaunchArgument(
            'vad_start_threshold',
            default_value='0.60',
            description='Silero probability required to start an utterance'),
        DeclareLaunchArgument(
            'vad_end_threshold',
            default_value='0.35',
            description='Silero probability below which audio counts as trailing silence'),
        DeclareLaunchArgument(
            'silence_timeout',
            default_value='0.6',
            description='Continuous trailing silence required to end an utterance, seconds'),
        DeclareLaunchArgument(
            'vad_diagnostics',
            default_value='true',
            description='Log VAD probability, RMS and peak while listening'),
        DeclareLaunchArgument(
            'vad_log_interval_ms',
            default_value='500',
            description='Minimum interval between VAD diagnostic log messages'),

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
                'silence_timeout': ParameterValue(silence_timeout, value_type=float),
                'pre_roll_ms': 300,
                'min_speech_ms': 160,
                'max_speech_s': 30.0,

                'model_path':
                    '/home/hopkira/whisper.cpp/models/ggml-large-v3-turbo.bin',
                'vad_model_path':
                    '/home/hopkira/whisper.cpp/models/ggml-silero-v6.2.0.bin',

                'use_gpu': True,
                'gpu_device': 0,
                'flash_attn': True,
                'whisper_threads': 4,
                'beam_size': 1,
                'language': 'en',
                'vad_threads': 2,

                'vad_start_threshold': ParameterValue(
                    vad_start_threshold, value_type=float),
                'vad_end_threshold': ParameterValue(
                    vad_end_threshold, value_type=float),
                'vad_diagnostics': ParameterValue(
                    vad_diagnostics, value_type=bool),
                'vad_log_interval_ms': ParameterValue(
                    vad_log_interval_ms, value_type=int),

                'publish_stale_results': False,
            }]
        )
    ])
