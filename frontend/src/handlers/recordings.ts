import { deleteAudio as apiDeleteAudio } from "../api";
import { SetRecordings } from "../types/recorder";

export function deleteAudio(audioKey: string, setRecordings: SetRecordings) {
  // setRecordings((prevState) => prevState.filter((record) => record.key !== audioKey));
  console.log('audioKey', audioKey)
  apiDeleteAudio(audioKey)
}