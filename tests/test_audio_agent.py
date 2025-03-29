from dimos.stream.audio.sound_processing.node_microphone import SounddeviceAudioSource
from dimos.stream.audio.sound_processing.node_output import SounddeviceAudioOutput
from dimos.stream.audio.sound_processing.node_volume_monitor import monitor
from dimos.stream.audio.sound_processing.node_normalizer import AudioNormalizer
from dimos.stream.audio.sound_processing.node_key_recorder import KeyTriggeredAudioRecorder
from dimos.stream.audio.text.node_stdout import TextPrinterNode
from dimos.stream.audio.tts.node_openai import OpenAITTSNode
from dimos.stream.audio.utils import keepalive
from dimos.stream.audio.stt.node_whisper import WhisperNode
import threading
from dimos.utils.threadpool import get_scheduler
from dimos.agents.agent import OpenAIAgent

# Global variable to store whisper_node
whisper_node_ready = threading.Event()

def run_audio_processing():
    global whisper_node
    # Create microphone source, recorder, and audio output
    mic = SounddeviceAudioSource()
    normalizer = AudioNormalizer()
    recorder = KeyTriggeredAudioRecorder()
    whisper_node = WhisperNode()  # Assign to global variable
    output = SounddeviceAudioOutput(sample_rate=24000)

    # Connect audio processing pipeline
    normalizer.consume_audio(mic.emit_audio())
    recorder.consume_audio(normalizer.emit_audio())
    monitor(recorder.emit_audio())
    whisper_node.consume_audio(recorder.emit_recording())

    # Create and connect the text printer node
    text_printer = TextPrinterNode(prefix="USER: ")
    text_printer.consume_text(whisper_node.emit_text())

    # Signal that whisper_node is ready
    whisper_node_ready.set()

    # Keep the audio processing running
    keepalive()

def main():
    global whisper_node

    # Start audio processing in a separate thread
    audio_thread = threading.Thread(target=run_audio_processing, daemon=True)
    audio_thread.start()

    # Wait for whisper_node to be initialized
    whisper_node_ready.wait()

    # Initialize agent with main thread pool scheduler
    agent = OpenAIAgent(
                dev_name="UnitreeExecutionAgent",
                input_query_stream=whisper_node.emit_text(),
                system_query="You are a helpful robot named daneel that does my bidding",
                pool_scheduler=get_scheduler(),
            )

    # Set up TTS for agent responses
    tts_node = OpenAITTSNode()
    tts_node.consume_text(agent.get_response_observable())
    
    # Create audio output for TTS
    response_output = SounddeviceAudioOutput(sample_rate=24000)
    response_output.consume_audio(tts_node.emit_audio())

    # Keep the main thread alive
    keepalive()

if __name__ == "__main__":
    main()


