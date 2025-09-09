import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile
from pydub import AudioSegment
from tensorflow.image import resize
from IPython.display import Audio

## Function
def load_model():
    model = tf.keras.models.load_model("C:/Users/Administrator/Downloads/archive (25)/Data/music_genre_classification_model.h5")
    return model

## Loading and preprocessing a single audio file for prediction
#torchaudio.set_audio_backend("soundfile")
def load_and_preprocess_file(file_path,target_shape=(148,148)):
    try:
        audio=AudioSegment.from_file(file_path)
        #y,sr=librosa.load(file_path,sr=None) ## waveform shape: (num_channels,num_samples)
        #y=waveform.numpy().squeeze() ## Convert to 1D numpy array
        y=np.array(audio.get_array_of_samples(),dtype=np.float32)
        sr=audio.frame_rate
        
        ## if stereo, convert to mono
        if audio.channels==2:
            y=y.reshape((-1,2))
            y=y.mean(axis=1)
            print("Converted to mono")
            
    except Exception as e:
        print("Error loading file:",file_path,"| Error:",e)
        return None, None
    
    chunk_duration=4
    overlap_duration=2
    chunk_samples=chunk_duration*sr
    overlap_samples=overlap_duration*sr
    num_chunks=int(np.ceil((len(y)-chunk_samples)/(chunk_samples-overlap_samples)))+1
    
    data=[]
    raw_specs=[]
    
    for i in range(num_chunks):
        start=i*(chunk_samples-overlap_samples)
        end=start+chunk_samples
        chunk=y[start:end]
        
        if len(chunk)==0:
            continue
        
        mel_spectrogram=librosa.feature.melspectrogram(y=chunk,sr=sr)
        mel_spectrogram_resized=resize(np.expand_dims(mel_spectrogram,axis=-1),target_shape)
        
        data.append(mel_spectrogram_resized)
        raw_specs.append(mel_spectrogram)

    return np.array(data), raw_specs

def model_prediction(model,X_test):
    y_pred=model.predict(X_test)
    predicted_categories=np.argmax(y_pred,axis=1)
    unique,counts=np.unique(predicted_categories,return_counts=True)
    max_count=np.max(counts)
    max_elements=unique[counts==max_count]
    return predicted_categories,int(max_count),int(max_elements[0])

### Streamlit App UI
st.sidebar.title("Dashboard")

app_mode=st.sidebar.selectbox("Choose the App Mode",["Home","About App","Predict Music Genre"])

## Home Page
if app_mode=="Home":
   st.markdown(
    """
    <style>
    .stApp {
        background-color: #181646;  /* Blue background */
        color: white;
    }
    h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

   st.markdown(''' ## Welcome to the,\n
   ## Music Genre Classification System! 🎶🎧''')
   image_path = "music_genre_home.png"
   st.image(image_path, use_container_width=True)
   
   st.markdown("""
*The main goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and the system will analyze it to detect its genre. Discover the power of AI in music analysis!*

### How It Works
1. *Upload Audio:* Go to the *Genre Classification* page and upload an audio file.
2. *Analysis:* This system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. *Results:* View the predicted genre along with related information.

### Why Choose This?
- *Accuracy:* This system leverages state-of-the-art deep learning models for accurate genre prediction.
- *User-Friendly:* Simple and intuitive interface for a smooth user experience.
- *Fast and Efficient:* Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the *Genre Classification* page in the sidebar to upload an audio file and explore the magic of my Music Genre Classification System!

### About Me
Learn more about the project, me, and my mission on the *About* page.
""")

## About Page   
elif app_mode=="About App":
    st.markdown("""
                ### About Project
                Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

                This data hopefully can give the opportunity to do just that.

                ### About Dataset
                #### Content
                1. *genres original* - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
                2. *List of Genres* - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
                3. *images original* - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. *2 CSV files* - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.
                """)

### Prediction Page
elif app_mode=="Predict Music Genre":
    st.header("Music Genre Classification System 🎵")
    test_mp3=st.file_uploader("Upload an audio file",type=["mp3"])
    
    if test_mp3 is not None:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False,suffix=".mp3") as temp_file:
            temp_file.write(test_mp3.read())
            file_path=temp_file.name
        
    ## Play Audio
        st.audio(test_mp3)
        
    ## Prediction Button
        with st.spinner("Loading Model..."):
            X_test,raw_specs=load_and_preprocess_file(file_path)
            
            if X_test is None:
                st.error("❌ Could not process the audio file. Please try another one.")
            
            else:
                model=load_model()
                st.success("✅ Model Loaded Successfully!")
                predictions, _, result_index=model_prediction(model,X_test)
                
                labels = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
                
                predicted_genre=labels[result_index].upper()
                st.success(f":blue[Predicted Music Genre:] *:red[{predicted_genre}]*")
                
                # --- Visualization & Debugging Section ---
        st.write("---")
        st.markdown("#### Prediction Confidence")
        
        # Confidence scores display
        for i, label_name in enumerate(labels):
            st.write(f"{label_name.capitalize()}: `{predictions[i]:.2%}`")

        # Spectrogram display
        if raw_specs and len(raw_specs) > 0:
            st.markdown("#### Mel Spectrogram (Visualization)")
            spec_for_display = raw_specs[0]
            display_img = (spec_for_display - spec_for_display.min()) / (spec_for_display.max() - spec_for_display.min() + 1e-6)
            st.image(display_img, use_column_width=True, caption="Mel Spectrogram of the first 4-second chunk")