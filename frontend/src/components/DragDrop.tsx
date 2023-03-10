import { useState } from 'react';
import { FileUploader } from 'react-drag-drop-files';
import { AudioExtension } from '../types/recorder';
import { saveAudio } from '../api';
import { useQueryClient } from 'react-query';


type Props = {}

const DragDrop = (props: Props) => {
    const queryClient = useQueryClient()

    // const [files, setFiles] = useState<string[]>([]);
    const [file, setFile] = useState<any>();
    // const mp3: AudioExtension = 'mp3'
    const wav: AudioExtension = 'wav'
    const handleChange = (newFile: any) => {
        // setFiles([...files, newFile]);
        setFile(newFile);
        console.log(newFile);
    };
    const send = () => {
        saveAudio(file)
        queryClient.fetchQuery("queryAudio")
    }
    return (
        <>
            <div>DragDrop</div>
            <FileUploader handleChange={handleChange} name="file" types={[wav]} label={'Drag & Drop file/s to upload.'} />
            <br />
            <br />
            <div>
                <p>{file ? `File name: ${file.name}` : "no files uploaded yet"}</p>
            </div>
            <button onClick={send}>Send</button>
        </>
    )
}

export default DragDrop