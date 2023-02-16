import useRecordingsList from "./hooks/useRecordings";
import { RecordingsListProps } from "./types/recorder";

const Recordings = ({ audio }: RecordingsListProps) => {
    const { recordings, deleteAudio } = useRecordingsList(audio);

    return (
        <div >
            {recordings.length > 0 ? (
                <>
                    <h1>Your recordings</h1>
                    <div >
                        {recordings.map((record) => (
                            <div key={record.key}>
                                <audio controls src={record.audio} />
                                <div >
                                    <button
                                        title="Delete this audio"
                                        onClick={() => deleteAudio(record.key)}
                                    >
                                        del
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </>
            ) : (
                <div >
                    <span>no rec</span>
                </div>
            )}
        </div>
    );
}
export default Recordings