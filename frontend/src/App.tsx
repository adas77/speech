import DragDrop from './components/DragDrop'
import Recorder from './components/Recorder'
import Recordings from './components/Recordings'
import useRecorder from './hooks/useRecorder'
import { UseRecorder } from './types/recorder'

function App() {
  const { recorderState, ...handlers }: UseRecorder = useRecorder()
  const { audio } = recorderState
  console.log(audio)
  return (
    <div className='mx-auto flex flex-wrap justify-center gap-64 mt-32' >
      <div>
        <b>Record Video</b>
        <br />
        <br />
        <Recorder recorderState={recorderState} handlers={handlers} />
        <Recordings />
      </div>
      <div>
        <b>Upload Video</b>
        <br />
        <br />

        <DragDrop />
      </div>
    </div>
  )
}

export default App
