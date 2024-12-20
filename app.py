import gradio as gr
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
import logging
import joblib 
import datetime
import os
import torch
import torchaudio
import tempfile
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from src.specrnet.specrnet import FrontendSpecRNet
from src.specrnet.frontends import prepare_lfcc_double_delta, prepare_mfcc_double_delta



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
    def __init__(self, processor, model, intermediate_embedding_layer=True, layer_index=3, single_audio_limit=15000):
        self.processor = processor
        self.model = model
        self.intermediate_embedding_layer = intermediate_embedding_layer
        self.layer_index = layer_index 
        self.single_audio_limit = single_audio_limit  # Max 15 seconds in milliseconds, change when there process is killed during audio_representation!

    def audio_representation(self, file_path):
        data, sr = sf.read(file_path)
        
        # If multiple channels, convert to mono
        if len(data.shape) > 1 and data.shape[1] > 1:
            print("Multiple channels detected, converting to mono!")
            data = data.mean(axis=1)

        float_arr = data.astype(float)
        print(f"Audio file is converted to float tensors of shape {float_arr.shape}")
        input_values = self.processor(float_arr, return_tensors="pt").input_values  # Batch size 1

        if self.intermediate_embedding_layer:
            try:
                hidden_state = self.model(input_values).hidden_states[self.layer_index]
            except Exception as e:
                print(f"-x-x-x-x-x- Check hidden layer index -x-x-x-x-x-")
                exit(1)
        else:
            hidden_state = self.model(input_values).last_hidden_state

        return hidden_state
        
    def split_audio(self, audio, file_path, split_seconds=12, min_duration=3500):
        try:
            print("Splitting Audio into smaller segments!")
            split_duration = split_seconds * 1000
            df = pd.DataFrame(columns=[f"feature_{i}" for i in range(self.model.config.output_hidden_size)])
            
            # Determine the format from the file extension
            file_format = os.path.splitext(file_path)[1][1:]  # Get the file extension without the dot
            
            for i in range(0, len(audio), split_duration):
                audio_split = audio[i:i + split_duration]

                if len(audio_split) < min_duration:
                    print(f"Skipping segment due to short duration")  # Technically only occurs at the end.
                    continue

                representation_layers = None

                # Create a temporary file with the same format as the original audio
                with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=True) as tmp_file:
                    audio_split.export(tmp_file.name, format=file_format)
                    representation_layers = self.audio_representation(tmp_file.name)

                representation_layers = torch.mean(representation_layers[0], dim=0)
                if representation_layers is None:
                    raise Exception("Failed to get audio representation!")

                row = pd.DataFrame(representation_layers.detach().numpy().reshape(1, -1), columns=df.columns)  # representation vectors stored as X
                df = pd.concat([df, row], ignore_index=True)
                print(f"Splitting {i} times.")
            print("Successfully obtained audio splits features!")

            return df

        except Exception as e:
            print(f"Error loading / splitting audio file: {e}")
            return None 

    def complete_embedding(self, file_path, split_seconds=12, min_duration=3500):
        try:
            if os.path.isfile(file_path):
                # Load audio using Pydub
                audio = AudioSegment.from_file(file_path)
                audio_duration = len(audio)
                print(f"{file_path}: {audio_duration / 1000} seconds")

                if audio_duration > self.single_audio_limit:
                    return self.split_audio(audio, file_path, split_seconds, min_duration)

                # Process the audio directly if it's shorter than the limit
                representation_layers = self.audio_representation(file_path)

                # Compute the mean of representation layers and create a row
                representation_layers = torch.mean(representation_layers[0], dim=0)
                row = pd.DataFrame(representation_layers.detach().numpy().reshape(1, -1),
                                   columns=[f"feature_{i}" for i in range(self.model.config.output_hidden_size)])

                return row

        except Exception as e:
            print(f"Failed to get feature representation due to {e}")
            return None

# Instantiation of embedModel.
wav2vecP = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")  #code source https://huggingface.co/transformers/v4.6.0/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html,  forward function
wav2vecM = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_attentions=True, output_hidden_states=True)

embedModel = EmbedModel(processor=wav2vecP, model=wav2vecM)

def wav2vec_ml_detect(file_path, model_name, state, threshold=60, basic_output=False, min_duration=3500, split_seconds=12):
    try:
        basic_output = basic_output == "true"
        # Naive bayes requires all 768 features but if SVM / LR it only uses 10 important features
        features = ['feature_122', 'feature_283', 'feature_663', 'feature_276', 'feature_349', 
                    'feature_343', 'feature_129', 'feature_566', 'feature_102', 'feature_494']
        
        state = state or []
        current_time = datetime.datetime.now().strftime("%H:%M")
        
        # Check if file_path is provided
        if file_path is None:
            raise Exception("No file uploaded!")
            
        # Get audio representation
        X_pred = embedModel.complete_embedding(file_path, split_seconds=split_seconds, min_duration=min_duration)
        if X_pred is None:
            raise Exception("Error in getting feature representation!")

        # Load Model & Predict
        print(f"Loading {model_name}...")
        model = joblib.load(models_dict.get(model_name))
        if model is None:
            raise Exception(f"Failed to load {model_name}.")
        
        if model_name != "Naive Bayes":
            print("Using only 10 features.")
            X_pred = X_pred[features]

        # Make predictions
        result = model.predict(X_pred)
        final_result = None
        # Check for edge cases
        if len(result) == 0:
            raise Exception("No predictions available.")
        elif len(result) == 1:
            confidence = 100 if result[0] == 1 else 0
            if basic_output:
                final_result = "bonafide" if result[0] == 1 else "spoof"
            else:
                final_result = f"{confidence}% bonafide" if result[0] == 1 else f"{confidence}% spoof"
        else:
            # Aggregate results based on threshold
            num_ones = np.sum(result)
            num_zeros = len(result) - num_ones
            result_percentage = (num_ones / len(result)) * 100  # Calculate percentage of 1s

            # Determine final result based on counts and threshold
            if basic_output:
                result_percentage = np.mean(result) * 100
                if result_percentage >= threshold:
                    final_result = "bonafide"
                else:
                    final_result = "spoof"
            else:
                if num_ones > num_zeros:
                    final_result = f"{result_percentage:.2f}% bonafide"
                elif num_ones == num_zeros:
                    final_result = "undetermined"
                else:
                    final_result = f"{100 - result_percentage:.2f}% spoof" # Equal numbers of 1s and 0s

        if final_result is None:
            raise Exception("Error in getting prediction result!")
        print(f"Result: {final_result}")
        result_message = f"[{current_time}] {model_name}: audio file is {final_result}"
        state.append(result_message)
        history = "\n".join(state)

        return history, state

    except Exception as e:
        print(f"Error during detection: {e}")
        return None, state

# Loaded models for wav2vec
models_dict = {
    "Support Vector Machine (SVM)": "./models/SVM.pkl",
    "Naive Bayes": "./models/NaiveBayes.pkl",
    "Logistic Regression": "./models/LR.pkl"
}

# Examples (Reserved for wav2vec)
w2m_examples = [
    ["./examples/long-bonafide.mp3", "Support Vector Machine (SVM)", 60, "false", 3500, 12],  
    ["./examples/long-spoof.flac", "Naive Bayes", 70, "true", 3000, 7], 
    ["./examples/short-spoof.wav", "Support Vector Machine (SVM)", 80, "true", 4000, 10], 
    ["./examples/short-bonafide.mp3", "Logistic Regression", 85, "false", 2500, 4], 
]


# Model for specrnet
srn_model_dict = {
    "SpecRNet": "./models/SpecRNet/ckpt.pth"
}

devices = [
    "cuda",
    "cpu"
]

def initialize_specrnet(model_path, input_channels, device, frontend_algorithm):
    global model  # Make model a global variable
    model = FrontendSpecRNet(input_channels=input_channels, device=device, frontend_algorithm=frontend_algorithm).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

def preprocess_audio(file_path, device):
    print(f"Processing file: {file_path}")
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"Original waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)
        print(f"Resampled waveform shape: {waveform.shape}")

    # Move waveform to device (GPU or CPU)
    waveform = waveform.to(device)

    return waveform

def predict(file_path, device):
    # Preprocess the audio
    audio = preprocess_audio(file_path, device)

    print(f"Input to model shape: {audio.shape}")

    # Perform prediction
    with torch.no_grad():
        output = model(audio)  # Use the global model variable
        prediction = torch.sigmoid(output).item()

    print(f"Raw prediction value: {prediction}")

    # Adjust the threshold for spoof classification
    threshold = 0.7  # Experiment with different thresholds, e.g., 0.6, 0.7, 0.8, etc.

    # Return the prediction as 'Spoof' or 'Bonafide' based on the adjusted threshold
    return "Bonafide" if prediction > threshold else "Spoof"


# SpecRNet detection function
def specrnet_detect(file_path, model_name, input_channels, device, frontend_algorithm, state):
    # Get the model path from the dictionary using the selected model name
    model_path = srn_model_dict.get(model_name)
    if model_path is None:
        raise ValueError(f"Model {model_name} not found in the model dictionary.")

    # Initialize the model
    initialize_specrnet(model_path, input_channels, device, frontend_algorithm)
    print(f"Model loaded from {model_path}")

    state = state or []

    try:
        # Get prediction using the SpecRNet model
        prediction = predict(file_path, device)
        current_time = datetime.datetime.now().strftime("%H:%M")
        result_message = f"[{current_time}] SpecRNet: audio file is {prediction}"
        state.append(result_message)
        history = "\n".join(state)

        return history, state
    except Exception as e:
        logging.error(f"Error during SpecRNet detection: {e}")
        state.append(f"Error: {e}")
        history = "\n".join(state)
        return history, state




# Interface
# Gradio Interface Setup
with gr.Blocks(title="Audio Deepfake Detection UI") as demo:
    gr.Markdown("## Audio Deepfake Classifier")
    gr.Markdown("<br /> Description:<br /> This interface allows you to detect audio deepfakes using Wav2Vec2 + ML models or SpecRNet neural networks.")

    with gr.Tabs():
        # Wav2Vec2 + ML Tab
        with gr.TabItem("wav2vec + ML"):
            results_St = gr.State([])

            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(label="Audio files only!")
                    model_names = [k for k, _ in models_dict.items()]   
                    selected_model = gr.Dropdown(label="Downstream Classifiers model", choices=model_names, value=model_names[0])
                    with gr.Accordion("Advanced settings", open=False):
                        gr.Markdown("Modifies the pipeline process of the model's prediction.")
                        with gr.Row():
                            gr.Markdown("This section is for handling audio files larger than the maximum limit of 15 seconds.")
                            split_seconds_slider = gr.Slider(minimum=2, maximum=15, label="Split into seconds", value=12, step=1, interactive=True)
                            min_duration_slider = gr.Slider(minimum=0, maximum=5000, label="Minimum duration in Milliseconds", value=3500, step=100, interactive=True)
                        with gr.Row():
                            use_nuanced_result = gr.Radio(label="Use binary result?", choices=["true", "false"], value="false", interactive=True)
                            threshold_activation_slider = gr.Slider(minimum=60, maximum=100, label="Audio Deepfake Classification Threshold", value=60, step=1, interactive=True)
                with gr.Row():
                    history = gr.Textbox(label="History", interactive=False, value="...", info="No history will be logged when an unexpected file is uploaded!", elem_classes="fill_height")
                
            predict_btn = gr.Button("Detect", variant="primary")

            predict_btn.click(
                fn=wav2vec_ml_detect,
                inputs=[file_upload, selected_model, results_St, threshold_activation_slider, use_nuanced_result, min_duration_slider, split_seconds_slider],
                outputs=[history, results_St],
            )

            gr.Examples(
                label="Examples",
                examples=w2m_examples, 
                cache_examples=False,
                inputs=[file_upload, selected_model, threshold_activation_slider, use_nuanced_result, min_duration_slider, split_seconds_slider],
            )

        # SpecRNet Tab
        # file_path, model_path, input_channels, device, frontend_algorithm,
        with gr.TabItem("SpecRNet"):
            srn_results_St = gr.State([])

            with gr.Row():
                with gr.Column():
                    srn_file_upload = gr.File(label="Audio files only!")
                    srn_model_names = [l for l, _ in srn_model_dict.items()]
                    srn_selected_model = gr.Dropdown(label="Neural Network Classifier Model", choices=srn_model_names, value=srn_model_names[0])
                    with gr.Accordion("View Model Configuration", open=False):
                        gr.Markdown("Modifies the pipeline process of the model's prediction.")
                        with gr.Row():
                            input_channel_config = gr.Radio(label="Input Channel Used For Processing with LFCC", choices=[1], value=1, interactive=True)
                        with gr.Row():
                            # Choose device based on CUDA availability
                            device_dropdown = gr.Dropdown(
                                label="Device to use", 
                                choices=devices, 
                                value=devices[0] if torch.cuda.is_available() else devices[1]
                            )
                        with gr.Row():
                            use_nuanced_result = gr.Radio(label="Use binary result?", choices=["true", "false"], value="false", interactive=True)
                        with gr.Row():
                            front_end_model_config = gr.Radio(label="Frontend Model Used", choices=["lfcc", "mfcc"], value="lfcc", interactive=False)

                with gr.Row():
                    srn_history = gr.Textbox(label="History", interactive=False, value="...", info="No history will be logged when an unexpected file is uploaded!", elem_classes="fill_height")

            srn_predict_btn = gr.Button("Detect", variant="primary")

            # Update srn_predict_btn.click to match the specrnet_detect function signature
            srn_predict_btn.click(
                fn=specrnet_detect,
                inputs=[
                    srn_file_upload,        # The uploaded file path
                    srn_selected_model,     # The selected model name, used to get the path
                    input_channel_config,   # Input channels for SpecRNet
                    device_dropdown,        # The device to use (CUDA or CPU)
                    front_end_model_config, # Frontend processing algorithm (e.g., "lfcc")
                    srn_results_St          # Gradio State to keep track of history
                ],
                outputs=[srn_history, srn_results_St]
            )


            gr.Examples(
                label="Examples",
                examples=[
                    ["./examples/long-bonafide.mp3"],  
                    ["./examples/long-spoof.flac"], 
                    ["./examples/short-spoof.wav"], 
                    ["./examples/short-bonafide.mp3"]
                ], 
                cache_examples=False,
                inputs=[srn_file_upload],
            )
                
            
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
