import { useMutation, useQuery } from "react-query";
import { deleteAudio, getAudios, predictAudio } from "../api";
import { Audio, RecordingsListProps } from "../types/recorder";
import { str_to_blob } from "../utils/audio";
// import "../../../audio"

const Recordings = ({ audio }: RecordingsListProps) => {
    const { isLoading, refetch, data: recordings } = useQuery({
        queryKey: 'query-audio',
        queryFn: getAudios

        // onSuccess:async(recodings)=>{
        //     QueryCache.setQueryData('forecast', recodings);
    })
    const del = useMutation((uuid: string) => {
        return deleteAudio(uuid)
    })
    // const delAudio=(uuid:string) => useQuery({
    //     queryKey: 'delete-audio',
    //     queryFn: deleteAudio
    //     // onSuccess:async(recodings)=>{
    //     //     QueryCache.setQueryData('forecast', recodings);
    // })
    // const { recordings, deleteAudio } = useRecordingsList(audio);

    return (
        <div >
            {/* { recordings.length > 0 ? ( */}
            {recordings ? (
                <>
                    <h1>Your recordings</h1>
                    <div >
                        {recordings.map((record: Audio) => (
                            <div key={record.key}>
                                {/* <audio controls src={`${import.meta.env.VITE_API}/audio/${record.key}.wav`} /> */}
                                <audio controls src={`${import.meta.env.VITE_API}/audio/${record.key}.wav`} />
                                <div >
                                    <button
                                        title="Delete this audio"
                                        onClick={() => del.mutate(record.key)}
                                    >
                                        del
                                    </button>
                                    <button
                                        title="Predict"
                                        onClick={() => predictAudio(record.key)}
                                    >
                                        predict
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