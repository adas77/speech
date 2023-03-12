from server.utils.audio import SpeechToText
st = SpeechToText('data', 'trained.hdf5')
st.train()