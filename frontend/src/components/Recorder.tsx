import { RecorderControlsProps } from "../types/recorder"


const Recorder = ({ recorderState, handlers }: RecorderControlsProps) => {
    const { recordingMinutes, recordingSeconds, initRecording } = recorderState
    const { startRecording, saveRecording, cancelRecording } = handlers

    return (
        <div >
            <div >
                <div >
                    {initRecording && <div className="recording-indicator"></div>}
                    {recordingMinutes}
                    <span>:</span>
                    {recordingSeconds}
                </div>
                {initRecording && (
                    <div >
                        <button title="Cancel recording" onClick={cancelRecording}>
                            cancel
                        </button>
                    </div>
                )}
            </div>
            <div >
                {initRecording ? (
                    <button

                        title="Save recording"
                        disabled={recordingSeconds === 0}
                        onClick={saveRecording}
                    >
                        save
                    </button>
                ) : (
                    <button title="Start recording" onClick={startRecording}>
                        start Rec
                    </button>
                )}
            </div>
        </div>
    )
}

export default Recorder