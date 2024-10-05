import gradio as gr
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
import logging
import joblib 
import datetime
import os

models_dict = {
    "Support Vector Machine (SVM)": "./models/SVM.pkl",
    "Naive Bayes": "./models/NaiveBayes.pkl",
    "Logistic Regression": "./models/LR.pkl"
}

logging.basicConfig(level=logging.INFO)

# Interface-related methods
def audio_to_segment(file_path):
    """
    Helper function to convert file path to AudioSegment.

    Args:
        file_path (str): Path to uploaded audio file.
    """
    try:
        return AudioSegment.from_file(file_path)
    except Exception as e:
        logging.error(f"Error converting audio to AudioSegment: {e}")
        return None

def trim_silence(file_path, state, silence_thresh=-40, min_silence_len=500, keep_silence=100):
    """
    Removes silence from an audio file and saves the processed audio.
    
    Args:
        file_path (str): Path to uploaded audio file.
        state (Gradio State): State for logs.
        silence_thresh (int): Silence threshold in dB. Default is -40 dB.
        min_silence_len (int): Minimum silence length (in ms) to consider as silence. Default is 500ms.
        keep_silence (int): Amount of silence (in ms) to leave at the beginning and end of each chunk. Default is 100ms.
    """
    try:
        state = state or []
        # Extract file name and extension from the file path
        original_file_name, original_extension = os.path.splitext(os.path.basename(file_path))

        # Convert audio file path to AudioSegment
        audio_segment = audio_to_segment(file_path)
        original_duration = len(audio_segment)

        if audio_segment is None:
            return None  # Error handled

        # Split audio based on silence
        chunks = split_on_silence(
            audio_segment,
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            keep_silence=keep_silence
        )
        # Combine chunks back to one segment
        trimmed_audio = AudioSegment.silent(duration=0)
        for chunk in chunks:
            trimmed_audio += chunk
        
        processed_duration = len(trimmed_audio)
        time_deducted = (original_duration - processed_duration) / 1000.0
        current_time = datetime.datetime.now().strftime("%H:%M")

        message = f"[{current_time}]: Time shaved off due to silence removal: {time_deducted} seconds. Original Length was {'' if ((original_duration / 1000)<60) else int((original_duration / 1000)/60)}{'' if ((original_duration / 1000)<60) else (' minute ' if ((original_duration / 1000)<120) else ' minutes ')}{int((original_duration / 1000) % 60)} seconds."
        state.append(message)
        history = "\n".join(state)

        # Save the trimmed audio to a new file with the original name and extension
        trimmed_file_path = f"{original_file_name}_trimmed{original_extension}"
        trimmed_audio.export(trimmed_file_path, format=original_extension[1:])  # Remove the dot from the extension
        
        return trimmed_file_path, history, state
    except Exception as e:
        logging.error(f"Error in trimming silence: {e}")
        return None

def resample_audio(file_path, target_sample_rate=16000):
    """
    Resamples audio sample rate.

    Args:
        file_path (str): Path to uploaded audio file.
        target_sample_rate (int): Target sample rate. 
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        resampled_data = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        
        # Save the resampled audio to a new file
        resampled_file_path = "resampled_output.wav"
        sf.write(resampled_file_path, resampled_data, target_sample_rate)
        
        return resampled_file_path
    except Exception as e:
        logging.error(f"Error in resampling audio: {e}")
        return None

def convert_audio_format(file_path, file_name="converted audio", target_format="wav"):
    """
    Convert audio file format.

    Args:
        file_path (str): Path to uploaded audio file.
        target_format (str): Accepted audio file formats. Default is wav.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        output_file = f"{file_name}.{target_format}"
        
        # Save the converted file in the desired format
        audio.export(output_file, format=target_format)
        
        return output_file
    except Exception as e:
        logging.error(f"Error in converting audio format: {e}")
        return None

def detect(file_path, model_name, state):
    state = state or []
    current_time = datetime.datetime.now().strftime("%H:%M")

    # Load and process the audio file for model prediction
    # (Placeholder logic for model loading)
    print(f"Using {model_name}")
    
    result = f"[{current_time}] {model_name}: File name is Bonafide"
    state.append(result)
    history = "\n".join(state)

    return history, state

# Interface
with gr.Blocks(title="Audio deepfake detection UI") as demo:
    gr.Markdown("## Audio Deepfake Classifier")
    gr.Markdown("<br /> Description:<br /> Here lies lorem ipsum")

    with gr.Tabs():
        with gr.TabItem("wav2vec + ML"):
            results_St = gr.State([])

            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(label="Upload Audio file")
                        model_names = [k for k, _ in models_dict.items()]   
                        selected_model = gr.Dropdown(label="Downstream Classifiers model", choices=model_names, value=model_names[0])
                    with gr.Column():
                        history = gr.Textbox(label="History", interactive=False, value="...", elem_classes="fill_height")
                
            predict_btn = gr.Button("Detect", variant="primary")

            predict_btn.click(
                fn=detect,
                inputs=[file_upload, selected_model, results_St],
                outputs=[history, results_St],
            )
        with gr.TabItem("SpecRNet"):
            gr.Audio()
            
        with gr.TabItem("Pre-processing"):
            gr.Markdown("Use the available methods to possibly increase the detection accuracy.")

            with gr.TabItem("Trim Silence"):
                logs_St = gr.State([])

                with gr.Row():
                    file_upload = gr.File(label="Upload Audio file")
                    trim_silence_btn = gr.Button("Trim Silence", variant="primary")
                    with gr.Column():
                        trimmed_audio = gr.Audio(interactive=False)
                        logs = gr.Textbox(interactive=False, label="Logs:", lines=3)
                with gr.Row():
                    with gr.Accordion("Advanced settings", open=False):
                        gr.Markdown("Trim silence saves the audio file as <b>WAV</b> format.")
                        silence_threshold_slider = gr.Slider(
                            minimum=-100, maximum=0, label="Silence Threshold (dB)", value=-40, step=1)
                        min_silence_len_slider = gr.Slider(
                            minimum=10, maximum=1000, label="Minimum silence length", value=500, step=10)
                        keep_silence_slider = gr.Slider(
                            minimum=0, maximum=1000, label="Keep Silence", value=100, step=10)

                trim_silence_btn.click(
                    fn=trim_silence,
                    inputs=[file_upload, logs_St, silence_threshold_slider, min_silence_len_slider, keep_silence_slider],
                    outputs=[trimmed_audio, logs, logs_St]
                )

            with gr.TabItem("Resample Audio"):
                with gr.Row():
                    file_upload = gr.File(label="Upload Audio file")
                    
                    with gr.Column():
                        with gr.Group():
                            sample_rate_view = gr.Number(value=0, label="Input file's sample rate:", interactive=False)
                            check_sample_rate = gr.Button("Check Sample Rate", variant="primary")
                        with gr.Group():
                            target_sample_rate = gr.Slider(
                            minimum=16000, maximum=48000, label="Target Sample Rate (SR)", value=16000, step=100)
                            resample_audio_btn = gr.Button("Resample Audio", variant="primary")
                    resampled_audio = gr.Audio(interactive=False)

                check_sample_rate.click(
                    fn=lambda file_path: librosa.get_samplerate(file_path) if file_path else 0,  # Get sample rate
                    inputs=file_upload,
                    outputs=sample_rate_view
                )

                resample_audio_btn.click(
                    fn=resample_audio,
                    inputs=[file_upload, target_sample_rate],
                    outputs=resampled_audio
                )

            with gr.TabItem("Convert Audio format"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            file_upload = gr.File(label="Upload Audio file")
                            file_name = gr.Textbox(label="Output file name:", placeholder="converted audio")
                    with gr.Column():
                        supported_formats_radio = gr.Radio(
                            label="Format conversion options:",
                            choices=["wav", "flac", "mp3"], value="wav")
                        gr.Markdown("""
                            * WAV - uncompressed & lossless at the cost of large file size. <br />
                            * FLAC - high audio quality and reduced file size. <br />
                            * MP3 - smallest file size at the cost of artifacts and loss of audio quality.
                        """)
                        convert_audio_format_btn = gr.Button("Convert Audio format", variant="primary")
                    converted_audio = gr.Audio(label="Converted audio",interactive=False)
                    

                convert_audio_format_btn.click(
                    fn=convert_audio_format,
                    inputs=[file_upload, file_name, supported_formats_radio],
                    outputs=converted_audio
                )

demo.launch(debug=True)
