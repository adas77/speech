# Download lib
Linux: `pip install -r requirements.txt`\
Linux - upgrade libs: `pip install -U -r requirements.txt`\
Windows: `py -m pip install -U -r requirements.txt`
- Docker 
    - TODO => Docker (flask, uwsgi, npm | jupyter notebook)

# Analiza audio
Jupyter Notebook: `notebook.ipynb`
- Użyte metody - plik `audio.py`:
    - DONE => klasa `Audio` analiza, wykrsy, itp, itd
    - DONE => dataset `data` mały przykładowy
    - DONE => klasa `SpeechToText` klasyfikuje nagranie (nazwa folderu to jedno słowo, czas uczenia u mnie ok 40s dwa wyrazy ok 4k plików audio ) na słowo (do lekkiej poprawy)
    - TODO => poprawa `SpeechToText` + dodanie tłumaczenia całych zdań
    - TODO => `TextToSpeech`
    - TODO => `SpeechToSpeech`

# Run Flask server
`./run.sh`
- Port `7777`
    - DONE => Dodawanie plików audio - folder `audio`
    - TODO => autoryzacja
    - TODO => CRUD (SQL_Alchemy) dla plików audio
    - TODO => użycie `SpeechToText` i `TextToSpeech`

# Run Frontend
`cd frontend && npm run dev`
Za pierwszym razem: `npm i`
- Port `5173`
    - DONE => Dodawanie plików audio
    - DONE => Śledzenie stanu plików
    - TODO => Tailwind CSS

## Źródła
[Fajna strona](https://musicinformationretrieval.com/index.html)
# Optymalizacja przy użyciu GPU
[Nvidia Cuda](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)\
[how-to-resolve-unmet-dependencies-error-when-upgrading-depends-nvidia-kernel-c](https://askubuntu.com/questions/1436506/how-to-resolve-unmet-dependencies-error-when-upgrading-depends-nvidia-kernel-c)\
[python gpu](https://www.geeksforgeeks.org/running-python-script-on-gpu/)
# Przetwarzanie tekstu
[Prosodic - A metrical-phonological parser](http://prosodic.stanford.edu/)\
[prosodic - python lib](https://pypi.org/project/prosodic/)

[Natural Language Toolkit](https://www.nltk.org/)
