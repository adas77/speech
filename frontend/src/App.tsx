import useRecorder from './hooks/useRecorder';
import Recorder from './components/Recorder';
import Recordings from './components/Recordings';
import { UseRecorder } from './types/recorder';

function App() {
  const { recorderState, ...handlers }: UseRecorder = useRecorder();
  const { audio } = recorderState;
  console.log(audio)
  return (
    <div >
      <button>A</button>
      <Recorder recorderState={recorderState} handlers={handlers}/>
      <Recordings audio={audio}/>
      <button>B</button>
    </div>
  )
}

export default App
