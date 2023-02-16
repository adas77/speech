import { useEffect, useState } from "react";
import { v4 as uuidv4 } from 'uuid';
import { deleteAudio } from "../handlers/recordings";
import { Audio } from "../types/recorder";


export default function useRecordingsList(audio: string | null) {
    const [recordings, setRecordings] = useState<Audio[]>([]);

    useEffect(() => {
        if (audio)
            setRecordings((prevState: Audio[]) => {
                return [...prevState, { key: uuidv4(), audio }];
            });
    }, [audio]);

    return {
        recordings,
        deleteAudio: (audioKey: string) => deleteAudio(audioKey, setRecordings),
    };
}