import { useState } from "react"
import { useMutation, useQuery } from "react-query"
import { deleteAudio, getAudios } from "../api"
import { Audio } from "../types/recorder"
import Error from "./Error"
import Spinner from "./Spinner"

const Recordings = () => {
    const [audios, setAudios] = useState<Audio[]>([])
    const { isLoading, isError, isRefetching } = useQuery("queryAudio", () =>
        getAudios(), {
        onSuccess(audios) {
            setAudios(audios)
        },

    })

    const { mutateAsync: del } = useMutation("deleteAudio", (uuid: string) =>
        deleteAudio(uuid)
        , {
            onSuccess(data) {
                setAudios(prev => prev.filter(a => a.key !== data?.data.uuid))
            },
        })

    return (
        <div >
            {(isLoading || isRefetching) && <Spinner />}
            {isError && <Error msg={"Error while fetching..."} />}
            {audios ? (
                <>
                    <h1>Your recordings</h1>
                    <div >
                        {audios.map((record: Audio) => (
                            <div key={record.key}>
                                <audio controls src={`${import.meta.env.VITE_API}/audio/${record.key}.wav`} />
                                <div >
                                    <button
                                        title="Delete this audio"
                                        onClick={() => del(record.key)}
                                    >
                                        del
                                    </button>
                                    <br />
                                    <button
                                        title="Predict"
                                        onClick={() => {
                                        }}
                                    >
                                        Predicted: {record.predicted}
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
    )
}
export default Recordings