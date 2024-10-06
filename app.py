import gradio as gr
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import logging
import joblib 
import datetime
import os
import torch

models_dict = {
    "Support Vector Machine (SVM)": "./models/SVM.pkl",
    "Naive Bayes": "./models/NaiveBayes.pkl",
    "Logistic Regression": "./models/LR.pkl"
}

max_single_audio_duration = 15000 # 15 seconds

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

# wav2vec + ml pipeline
class EmbedModel:
    """ 
    Feature representation from pre-trained SSL wav2vec model.

    Args:
        processor : wav2vec 960h base processor.
        model : wav2vec 960h base model.
        intermediate_embedding_layer (bool) : representation layers. Default is True.
        layer_index (int) : specific intermediate layer to use. Default is 3.
    """
    def __init__(self, processor, model, intermediate_embedding_layer=True, layer_index=3, single_audio_limit = max_single_audio_duration):
        self.processor = processor
        self.model = model
        self.intermediate_embedding_layer = intermediate_embedding_layer
        self.layer_index = layer_index 
        self.single_audio_limit = single_audio_limit # 15 seconds

    def audio_representation(self, audio):
        data, sr = sf.read(audio)
        # Multiple channels
        if len(data.shape) > 1 and data.shape[1] > 1:
            print("Multiple channels, converting to mono!")
            data = data.mean(axis=1)

        float_arr = data.astype(float)
        print(f"Audio file is converted to float tensors of shape {float_arr.shape}")
        input_values = self.processor(float_arr, return_tensors="pt").input_values # Batch size 1

        if self.intermediate_embedding_layer:
            try:
                hidden_state = self.model(input_values).hidden_states[self.layer_index]
            except Exception as e:
                print(f"-x-x-x-x-x- Check hidden layer index -x-x-x-x-x-")
                exit(1)
        else:
            hidden_state = self.model(input_values).last_hidden_state

        return hidden_state
         
    def complete_embedding(self, file_path):
        try:
            if os.path.isfile(file_path):
                # Single file processing
                audio = AudioSegment.from_file(file_path)
                audio_duration = len(audio)
                print(f"{file_path}: {audio_duration} seconds")
                
                if (audio_duration > self.single_audio_limit):
                    raise Exception(f"Audio file is longer than max single duration of {self.single_audio_limit / 1000} seconds. Audio is around {audio_duration / 1000} seconds!")
                
                # Initialize DataFrame only if the audio file passes the check
                representation_layers = self.audio_representation(file_path)

                # Compute the mean of representation layers and create a row
                representation_layers = torch.mean(representation_layers[0], dim=0)
                row = pd.DataFrame(representation_layers.detach().numpy().reshape(1, -1), 
                                columns=[f"feature_{i}" for i in range(self.model.config.output_hidden_size)])

                return row
            else:
                files_in_dir = os.listdir(file_path)
                count = 0

                file_names = [file for file in files_in_dir if os.path.isfile(os.path.join(file_path, file))]

                df = pd.DataFrame(columns=[f"feature_{i}" for i in range(self.model.config.output_hidden_size)])

                for file_name in file_names:
                    count +=1
                    print(f"-x-x-x-x-x- ID{count} file is processing -x-x-x-x-x-")
                    representation_layers = self.audio_representation(file_name)

                    representation_layers = torch.mean(representation_layers[0], dim=0) 
                    row = pd.DataFrame(representation_layers.detach().numpy().reshape(1, -1), columns=df.columns) #representation vectors stored as X
                    df = pd.concat([df, row], ignore_index=True)

                return df

        except Exception as e:
            print(f"Failed to get feature representation due to {e}")
            return None

class AudioSplit:
    pass

# Instantiation of embedModel.
wav2vecP = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")  #code source https://huggingface.co/transformers/v4.6.0/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html,  forward function
wav2vecM = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_attentions=True, output_hidden_states=True)

embedModel = EmbedModel(processor=wav2vecP, model=wav2vecM)

def detect(file_path, model_name, state):
    try:
        state = state or []
        current_time = datetime.datetime.now().strftime("%H:%M")
        # Split files first
        if file_path == None:
            raise Exception("No file uploaded!")
            
        # Audio representation
        df = embedModel.complete_embedding(file_path)
        if df is None:
            raise Exception("issue regarding file_path or audio file duration. Check logs!")

        df.to_csv("features.csv",index=False)
        # Load Model & Predict
        print(f"Using {model_name}")
        
        result = f"[{current_time}] {model_name}: File name is Bonafide"
        state.append(result)
        history = "\n".join(state)

        # Once done, remove the generated files.
        os.remove("features.csv")
        return history, state

    except Exception as e:
        logging.error(f"Error in detection audio deepfake: {e}")
        state = state or []
        current_time = datetime.datetime.now().strftime("%H:%M")
        result = f"[{current_time}] Failed attempt: {e}"
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
                    file_upload = gr.File(label=f"Upload Audio file max of {max_single_audio_duration}")
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
                    file_upload = gr.File(label="Upload Audio file")
                            
                    with gr.Column():
                        supported_formats_radio = gr.Radio(
                            label="Format conversion options:",
                            choices=["wav", "flac", "mp3"], value="wav")
                        gr.Markdown("""
                            * WAV - uncompressed & lossless at the cost of large file size. <br />
                            * FLAC - high audio quality and reduced file size. <br />
                            * MP3 - smallest file size at the cost of artifacts and loss of audio quality.
                        """)
                        file_name = gr.Textbox(label="Output file name:", placeholder="converted audio")
                    converted_audio = gr.Audio(label="Converted audio",interactive=False)
                    
                convert_audio_format_btn = gr.Button("Convert Audio format", variant="primary")
                convert_audio_format_btn.click(
                    fn=convert_audio_format,
                    inputs=[file_upload, file_name, supported_formats_radio],
                    outputs=converted_audio
                )

demo.launch(debug=True)
